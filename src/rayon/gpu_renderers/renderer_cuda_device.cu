// Host-side CUDA renderer implementation.
// Device-side kernels and ray tracing logic live in gpu_renderers/shaders/*.cu(h).

#include "cuda_raytracer.cuh"
#include "cuda_scene.cuh"
#include "cuda_utils.cuh"

#include "shaders/render_acc_kernel.cuh"

#include <cstdio>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <vector>

//==================== HOST MODIFIED PARAMETERS ====================
// Global device constants (single definition here)
__constant__ float g_light_intensity = 1.0f;
__constant__ float g_background_intensity = 1.0f;
__constant__ float g_metal_fuzziness = 0.8f;
__constant__ float g_glass_refraction_index = 1.5f; // Default glass index

// Depth of Field constants
__constant__ bool g_dof_enabled = false;
__constant__ float g_dof_aperture = 0.1f;
__constant__ float g_dof_focus_distance = 10.0f;

//==================== CUDA STREAM & PINNED MEMORY ====================
// Display stream: gamma-correct kernel + async D2H copy pipeline.
// Non-blocking so it does not implicitly synchronize with the default stream.
static cudaStream_t s_display_stream = nullptr;
static unsigned char *s_pinned_display = nullptr;
static size_t s_pinned_display_size = 0;

extern "C" void initCudaStreams()
{
   if (s_display_stream == nullptr)
      cudaStreamCreateWithFlags(&s_display_stream, cudaStreamNonBlocking);
}

extern "C" void cleanupCudaStreams()
{
   if (s_display_stream != nullptr)
   {
      cudaStreamDestroy(s_display_stream);
      s_display_stream = nullptr;
   }
   if (s_pinned_display != nullptr)
   {
      cudaFreeHost(s_pinned_display);
      s_pinned_display = nullptr;
      s_pinned_display_size = 0;
   }
}

//==================== HOST INTERFACE FUNCTIONS ====================
extern "C" void setLightIntensity(float intensity) { cudaMemcpyToSymbol(g_light_intensity, &intensity, sizeof(float)); }

extern "C" void setBackgroundIntensity(float intensity)
{
   cudaMemcpyToSymbol(g_background_intensity, &intensity, sizeof(float));
}

extern "C" void setMetalFuzziness(float fuzziness) { cudaMemcpyToSymbol(g_metal_fuzziness, &fuzziness, sizeof(float)); }

extern "C" void setGlassRefractionIndex(float index)
{
   cudaMemcpyToSymbol(g_glass_refraction_index, &index, sizeof(float));
}

// Depth of Field setters
extern "C" void setDOFEnabled(bool enabled) { cudaMemcpyToSymbol(g_dof_enabled, &enabled, sizeof(bool)); }

extern "C" void setDOFAperture(float aperture) { cudaMemcpyToSymbol(g_dof_aperture, &aperture, sizeof(float)); }

extern "C" void setDOFFocusDistance(float distance)
{
   cudaMemcpyToSymbol(g_dof_focus_distance, &distance, sizeof(float));
}

//==================== DEVICE MEMORY MANAGEMENT ====================

/**
 * @brief Calculate optimal thread block configuration for 2D image rendering
 * Caches the result after first query to avoid repeated cudaGetDeviceProperties calls.
 */
static dim3 getOptimalBlockSize(int width, int height)
{
   static dim3 cached_block_size(0, 0);
   if (cached_block_size.x == 0)
   {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, 0);

      int blockSizeX = 32;
      int blockSizeY = 8;

      if (prop.maxThreadsPerBlock < 256)
      {
         blockSizeX = 16;
         blockSizeY = 16;
      }
      cached_block_size = dim3(blockSizeX, blockSizeY);
   }
   return cached_block_size;
}

extern "C" void freeDeviceRandomStates(void *d_rand_states)
{
   if (d_rand_states != nullptr)
   {
      cudaFree(d_rand_states);
   }
}

extern "C" void freeDeviceAccumBuffer(void *d_accum_buffer)
{
   if (d_accum_buffer != nullptr)
   {
      cudaFree(d_accum_buffer);
   }
}

extern "C" void resetDeviceAccumBuffer(void *d_accum_buffer, int num_pixels)
{
   if (d_accum_buffer != nullptr)
   {
      cudaMemset(d_accum_buffer, 0, (size_t)num_pixels * sizeof(float4));
   }
}
/**
 * @brief Accumulative rendering function that adds new samples to existing accumulated color buffer
 *
 * This function allows progressive refinement by accumulating samples over multiple calls.
 * The accumulation buffer stays on the GPU for maximum performance - no copying back and forth.
 *
 * IMPORTANT: The 'image' parameter is IGNORED - the kernel only updates the accumulation buffer.
 * The caller must apply gamma correction separately using convertAccumBufferToImage().
 *
 * @param image UNUSED - kernel ignores this and sets device image buffer to 0
 * @param accum_buffer Host accumulation buffer (float RGB) - copied to/from device
 * @param scene Scene structure with device pointers
 * @param width Image width
 * @param height Image height
 * @param cam_center_x,y,z Camera center position
 * @param pixel00_x,y,z Top-left pixel center
 * @param delta_u_x,y,z Pixel step in U direction
 * @param delta_v_x,y,z Pixel step in V direction
 * @param samples_to_add Number of new samples to add this iteration
 * @param total_samples_so_far Total samples accumulated INCLUDING this batch
 * @param max_depth Maximum ray bounce depth
 * @param d_rand_states_ptr Persistent device random states pointer
 * @param d_accum_buffer_ptr Persistent device accumulation buffer pointer (stays on device!)
 * @return Number of rays traced
 */
// Progressive accumulative rendering host wrapper (used by interactive SDL renderer)
extern "C" unsigned long long renderPixelsCUDAAccumulative(
    unsigned char *image, float *accum_buffer, CudaScene::Scene *scene, int width, int height, double cam_center_x,
    double cam_center_y, double cam_center_z, double pixel00_x, double pixel00_y, double pixel00_z, double delta_u_x,
    double delta_u_y, double delta_u_z, double delta_v_x, double delta_v_y, double delta_v_z, int samples_to_add,
    int total_samples_so_far, int max_depth, void **d_rand_states_ptr, void **d_accum_buffer_ptr, double cam_u_x,
    double cam_u_y, double cam_u_z, double cam_v_x, double cam_v_y, double cam_v_z,
    void *d_pixel_sample_counts, int min_adaptive_samples, float adaptive_threshold)
{
#ifdef DIAGS
   static bool first_call = true;
   if (first_call)
   {
      int device;
      cudaGetDevice(&device);
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, device);
      printf("- GPU: %s (SM %d.%d, %d multiprocessors)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
      printf("- Max threads per block: %d\n", prop.maxThreadsPerBlock);
      first_call = false;
   }
#endif

   if (!scene)
      return 0ULL;

   float4 *d_accum = nullptr;
   curandState *d_rand_states = nullptr;

   size_t accum_size = (size_t)width * height * sizeof(float4);
   int num_pixels = width * height;

#ifdef DIAGS
   unsigned long long *d_ray_count = nullptr;
   cudaMalloc(&d_ray_count, sizeof(unsigned long long));
#else
   // When DIAGS is off, pass a dummy pointer — kernel won't write to it
   unsigned long long *d_ray_count = nullptr;
#endif

   bool need_rand_init = false;

   if (*d_rand_states_ptr == nullptr)
   {
      cudaMalloc(&d_rand_states, num_pixels * sizeof(curandState));
      *d_rand_states_ptr = d_rand_states;
      need_rand_init = true;
   }
   else
   {
      d_rand_states = static_cast<curandState *>(*d_rand_states_ptr);
   }

   if (*d_accum_buffer_ptr == nullptr)
   {
      cudaMalloc(&d_accum, accum_size);
      *d_accum_buffer_ptr = d_accum;

      if (accum_buffer != nullptr)
      {
         // Batch renderer path: upload host float3 buffer as float4
         float4 *host_f4 = (float4 *)malloc(accum_size);
         for (int i = 0; i < num_pixels; ++i)
         {
            host_f4[i] = make_float4(accum_buffer[i * 3], accum_buffer[i * 3 + 1], accum_buffer[i * 3 + 2], 0.0f);
         }
         cudaMemcpy(d_accum, host_f4, accum_size, cudaMemcpyHostToDevice);
         free(host_f4);
      }
      else
      {
         // Progressive renderer path: start from zero, no host buffer involved
         cudaMemset(d_accum, 0, accum_size);
      }
   }
   else
   {
      d_accum = static_cast<float4 *>(*d_accum_buffer_ptr);
   }

#ifdef DIAGS
   cudaMemset(d_ray_count, 0, sizeof(unsigned long long));
#endif

   // Use optimized thread block configuration
   dim3 threads = getOptimalBlockSize(width, height);
   dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

   // Verify block size on first call
   static bool block_size_verified = false;
   if (!block_size_verified)
   {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, 0);
      int total_threads = threads.x * threads.y;
      if (total_threads > prop.maxThreadsPerBlock)
      {
         printf("❌ ERROR: Block size %dx%d (%d threads) exceeds device limit %d\n", threads.x, threads.y,
                total_threads, prop.maxThreadsPerBlock);
      }
      block_size_verified = true;
   }

   if (need_rand_init)
   {
      init_random_states<<<blocks, threads>>>(d_rand_states, num_pixels, 1234ULL, width);
      cudaError_t err = cudaDeviceSynchronize();
      if (err != cudaSuccess)
      {
         printf("❌ RNG init failed: %s\n", cudaGetErrorString(err));
      }
      else
      {
#ifdef DIAGS
         // Verify RNG state was initialized
         unsigned int test_state;
         cudaMemcpy(&test_state, d_rand_states, sizeof(unsigned int), cudaMemcpyDeviceToHost);

         printf("- RNG initialized: first state value = %u\n", test_state);
#endif
      }
   }

   renderAccKernel<<<blocks, threads>>>(
       d_accum, scene, width, height, samples_to_add, total_samples_so_far, max_depth, (float)cam_center_x,
       (float)cam_center_y, (float)cam_center_z, (float)pixel00_x, (float)pixel00_y, (float)pixel00_z, (float)delta_u_x,
       (float)delta_u_y, (float)delta_u_z, (float)delta_v_x, (float)delta_v_y, (float)delta_v_z, d_ray_count,
       d_rand_states, (float)cam_u_x, (float)cam_u_y, (float)cam_u_z, (float)cam_v_x, (float)cam_v_y, (float)cam_v_z,
       static_cast<int *>(d_pixel_sample_counts), min_adaptive_samples, adaptive_threshold);

   cudaError_t kernel_err = cudaGetLastError();
   if (kernel_err != cudaSuccess)
   {
      printf("❌ Kernel launch error: %s\n", cudaGetErrorString(kernel_err));
   }

   cudaError_t sync_err = cudaDeviceSynchronize();
   if (sync_err != cudaSuccess)
   {
      printf("❌ Kernel execution error: %s\n", cudaGetErrorString(sync_err));
   }

   // Only copy accum buffer back to host when not using GPU-side display conversion.
   // The progressive renderer uses convertAccumToDisplayCUDA() instead, avoiding this copy.
   if (accum_buffer != nullptr)
   {
      float4 *host_f4 = (float4 *)malloc(accum_size);
      cudaMemcpy(host_f4, d_accum, accum_size, cudaMemcpyDeviceToHost);
      for (int i = 0; i < num_pixels; ++i)
      {
         accum_buffer[i * 3] = host_f4[i].x;
         accum_buffer[i * 3 + 1] = host_f4[i].y;
         accum_buffer[i * 3 + 2] = host_f4[i].z;
      }
      free(host_f4);
   }

#ifdef DIAGS
   // Exact ray count from kernel (includes all bounces)
   unsigned long long host_ray_count = 0ULL;
   cudaMemcpy(&host_ray_count, d_ray_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
   cudaFree(d_ray_count);
   return host_ray_count;
#else
   // Estimate: primary rays only (width * height * samples)
   return (unsigned long long)num_pixels * samples_to_add;
#endif
}

/**
 * @brief GPU-side gamma correction and display image conversion
 *
 * Reads the device accumulation buffer, applies gamma correction on GPU,
 * and copies only the small uint8 display image back to host.
 * This avoids the expensive float4 D2H copy + host-side conversion.
 *
 * @param d_accum_buffer Device accumulation buffer (float4, persistent)
 * @param display_image Host display image buffer (uint8, width*height*channels)
 * @param width Image width
 * @param height Image height
 * @param channels Number of color channels (3 or 4)
 * @param num_samples Total accumulated samples for normalization
 * @param gamma Gamma correction value
 */
extern "C" void convertAccumToDisplayCUDA(void *d_accum_buffer, unsigned char *display_image, int width, int height,
                                          int channels, int num_samples, float gamma, void *d_pixel_sample_counts)
{
   if (d_accum_buffer == nullptr || display_image == nullptr || num_samples <= 0)
      return;

   size_t display_size = (size_t)width * height * channels * sizeof(unsigned char);

   // Allocate device display buffer (reuse across calls)
   static unsigned char *d_display = nullptr;
   static size_t d_display_size = 0;

   if (d_display == nullptr || d_display_size != display_size)
   {
      if (d_display != nullptr)
         cudaFree(d_display);
      cudaMalloc(&d_display, display_size);
      d_display_size = display_size;
   }

   // Allocate/resize pinned host staging buffer for async D2H copy
   if (s_display_stream && (s_pinned_display == nullptr || s_pinned_display_size != display_size))
   {
      if (s_pinned_display != nullptr)
         cudaFreeHost(s_pinned_display);
      cudaMallocHost(&s_pinned_display, display_size);
      s_pinned_display_size = display_size;
   }

   dim3 threads = getOptimalBlockSize(width, height);
   dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

   // Use display stream: kernel and async copy are pipelined with no CPU gap between them.
   // Falls back to default stream if initCudaStreams() was not called.
   cudaStream_t stream = s_display_stream ? s_display_stream : 0;

   gammaCorrectKernel<<<blocks, threads, 0, stream>>>(
       static_cast<float4 *>(d_accum_buffer), d_display, width, height, num_samples,
       channels, gamma, static_cast<const int *>(d_pixel_sample_counts));

   if (s_display_stream && s_pinned_display)
   {
      // Async path: kernel → DMA copy queued on same stream, single sync at end
      cudaMemcpyAsync(s_pinned_display, d_display, display_size, cudaMemcpyDeviceToHost, s_display_stream);
      cudaStreamSynchronize(s_display_stream);
      memcpy(display_image, s_pinned_display, display_size);
   }
   else
   {
      // Fallback synchronous path
      cudaDeviceSynchronize();
      cudaMemcpy(display_image, d_display, display_size, cudaMemcpyDeviceToHost);
   }
}

//==================== ADAPTIVE SAMPLING BUFFER MANAGEMENT ====================

extern "C" void allocateAdaptiveBuffer(void **d_pixel_sample_counts, int num_pixels)
{
   if (*d_pixel_sample_counts == nullptr)
   {
      cudaMalloc(d_pixel_sample_counts, (size_t)num_pixels * sizeof(int));
      cudaMemset(*d_pixel_sample_counts, 0, (size_t)num_pixels * sizeof(int));
   }
}

extern "C" void resetAdaptiveBuffer(void *d_pixel_sample_counts, int num_pixels)
{
   if (d_pixel_sample_counts != nullptr)
   {
      cudaMemset(d_pixel_sample_counts, 0, (size_t)num_pixels * sizeof(int));
   }
}

extern "C" void freeAdaptiveBuffer(void *d_pixel_sample_counts)
{
   if (d_pixel_sample_counts != nullptr)
   {
      cudaFree(d_pixel_sample_counts);
   }
}

extern "C" int countConvergedPixels(void *d_pixel_sample_counts, int num_pixels)
{
   if (d_pixel_sample_counts == nullptr)
      return 0;

   // Copy buffer to host and count negative values (converged pixels)
   std::vector<int> host_counts(num_pixels);
   cudaMemcpy(host_counts.data(), d_pixel_sample_counts, (size_t)num_pixels * sizeof(int), cudaMemcpyDeviceToHost);

   int converged = 0;
   for (int i = 0; i < num_pixels; ++i)
   {
      if (host_counts[i] < 0)
         ++converged;
   }
   return converged;
}

extern "C" void renderSampleHeatmapCUDA(void *d_pixel_sample_counts, unsigned char *display_image, int width, int height,
                                        int channels, int max_samples_for_scale)
{
   if (d_pixel_sample_counts == nullptr || display_image == nullptr)
      return;

   size_t display_size = (size_t)width * height * channels * sizeof(unsigned char);

   // Reuse the same static display buffer as convertAccumToDisplayCUDA
   static unsigned char *d_display_heatmap = nullptr;
   static size_t d_display_heatmap_size = 0;

   if (d_display_heatmap == nullptr || d_display_heatmap_size != display_size)
   {
      if (d_display_heatmap != nullptr)
         cudaFree(d_display_heatmap);
      cudaMalloc(&d_display_heatmap, display_size);
      d_display_heatmap_size = display_size;
   }

   dim3 threads = getOptimalBlockSize(width, height);
   dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

   sampleHeatmapKernel<<<blocks, threads, 0, s_display_stream ? s_display_stream : 0>>>(
       static_cast<const int *>(d_pixel_sample_counts), d_display_heatmap, width, height, channels,
       max_samples_for_scale);

   if (s_display_stream && s_pinned_display)
   {
      cudaMemcpyAsync(s_pinned_display, d_display_heatmap, display_size, cudaMemcpyDeviceToHost, s_display_stream);
      cudaStreamSynchronize(s_display_stream);
      memcpy(display_image, s_pinned_display, display_size);
   }
   else
   {
      cudaDeviceSynchronize();
      cudaMemcpy(display_image, d_display_heatmap, display_size, cudaMemcpyDeviceToHost);
   }
}

// Use renderPixelsCUDAAccumulative for all rendering (one-shot and progressive).
// For one-shot rendering: call with samples_to_add = total samples, total_samples_so_far = 0
