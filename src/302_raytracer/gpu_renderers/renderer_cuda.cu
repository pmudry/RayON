// Host-side CUDA renderer implementation.
// Device-side kernels and ray tracing logic live in gpu_renderers/shaders/*.cu(h).

#include "cuda_float3.cuh"
#include "cuda_scene.cuh"
#include "cuda_utils.cuh"
#include "shaders/shader_common.cuh"
#include "shaders/render_acc_kernel.cuh"
#include "shaders/render_scene_kernel.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <cstdio>

// Global device constants (single definition here)
__constant__ float g_light_intensity = 1.0f;
__constant__ float g_background_intensity = 1.0f;
__constant__ float g_metal_fuzziness = 0.8f;
__constant__ float g_glass_refraction_index = 1.5f;  // Default glass index

// Depth of Field constants
__constant__ bool g_dof_enabled = false;
__constant__ float g_dof_aperture = 0.1f;
__constant__ float g_dof_focus_distance = 10.0f;

//==================== HOST INTERFACE FUNCTIONS ====================
extern "C" void setLightIntensity(float intensity) { cudaMemcpyToSymbol(g_light_intensity, &intensity, sizeof(float)); }

extern "C" void setBackgroundIntensity(float intensity)
{
   cudaMemcpyToSymbol(g_background_intensity, &intensity, sizeof(float));
}

extern "C" void setMetalFuzziness(float fuzziness) { cudaMemcpyToSymbol(g_metal_fuzziness, &fuzziness, sizeof(float)); }

extern "C" void setGlassRefractionIndex(float index) { cudaMemcpyToSymbol(g_glass_refraction_index, &index, sizeof(float)); }

// Depth of Field setters
extern "C" void setDOFEnabled(bool enabled) { cudaMemcpyToSymbol(g_dof_enabled, &enabled, sizeof(bool)); }

extern "C" void setDOFAperture(float aperture) { cudaMemcpyToSymbol(g_dof_aperture, &aperture, sizeof(float)); }

extern "C" void setDOFFocusDistance(float distance) { cudaMemcpyToSymbol(g_dof_focus_distance, &distance, sizeof(float)); }

//==================== DEVICE MEMORY MANAGEMENT ====================

/**
 * @brief Calculate optimal thread block configuration for 2D image rendering
 * @param width Image width
 * @param height Image height
 * @return dim3 with optimal block dimensions
 */
static dim3 getOptimalBlockSize(int width, int height)
{
   // Query device properties
   cudaDeviceProp prop;
   cudaGetDeviceProperties(&prop, 0);
   
   // For modern GPUs, 256 threads per block is often optimal
   // Use rectangular blocks (32x8) for better memory coalescing
   // 32 threads in x-direction aligns with warp size for coalesced memory access
   int blockSizeX = 32;
   int blockSizeY = 8;
   
   // For older/smaller GPUs, fall back to 16x16
   if (prop.maxThreadsPerBlock < 256)
   {
      blockSizeX = 16;
      blockSizeY = 16;
   }
   
   return dim3(blockSizeX, blockSizeY);
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
/**
 * @brief Accumulative rendering function that adds new samples to existing accumulated color buffer
 *
 * This function allows progressive refinement by accumulating samples over multiple calls.
 * The accumulation buffer stays on the GPU for maximum performance - no copying back and forth.
 *
 * @param image Output RGB image buffer (byte array) - only copied to host when needed
 * @param accum_buffer Host accumulation buffer (float3 array) - only used for initialization/final readback
 * @param scene Scene structure with device pointers
 * @param width Image width
 * @param height Image height
 * @param cam_center_x,y,z Camera center position
 * @param pixel00_x,y,z Top-left pixel center
 * @param delta_u_x,y,z Pixel step in U direction
 * @param delta_v_x,y,z Pixel step in V direction
 * @param samples_to_add Number of new samples to add this iteration
 * @param total_samples_so_far Total samples accumulated (including these new ones)
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
    int total_samples_so_far, int max_depth, void **d_rand_states_ptr, void **d_accum_buffer_ptr,
    double cam_u_x, double cam_u_y, double cam_u_z, double cam_v_x, double cam_v_y, double cam_v_z)
{
   if (!scene) return 0ULL;

   unsigned char *d_image = nullptr;
   float *d_accum = nullptr;
   unsigned long long *d_ray_count = nullptr;
   curandState *d_rand_states = nullptr;

   size_t image_size = (size_t)width * height * 3 * sizeof(unsigned char);
   size_t accum_size = (size_t)width * height * 3 * sizeof(float);
   int num_pixels = width * height;

   cudaMalloc(&d_image, image_size);
   cudaMalloc(&d_ray_count, sizeof(unsigned long long));

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
      cudaMemcpy(d_accum, accum_buffer, accum_size, cudaMemcpyHostToDevice);
   }
   else
   {
      d_accum = static_cast<float *>(*d_accum_buffer_ptr);
   }

   cudaMemset(d_ray_count, 0, sizeof(unsigned long long));

   // Use optimized thread block configuration
   dim3 threads = getOptimalBlockSize(width, height);
   dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

   if (need_rand_init)
   {
      init_random_states<<<blocks, threads>>>(d_rand_states, num_pixels, 1234ULL, width);
      cudaDeviceSynchronize();
   }

   renderAccKernel<<<blocks, threads>>>(d_accum, d_image, scene, width, height, samples_to_add, total_samples_so_far,
                                        max_depth, (float)cam_center_x, (float)cam_center_y, (float)cam_center_z,
                                        (float)pixel00_x, (float)pixel00_y, (float)pixel00_z, (float)delta_u_x,
                                        (float)delta_u_y, (float)delta_u_z, (float)delta_v_x, (float)delta_v_y,
                                        (float)delta_v_z, d_ray_count, d_rand_states,
                                        (float)cam_u_x, (float)cam_u_y, (float)cam_u_z,
                                        (float)cam_v_x, (float)cam_v_y, (float)cam_v_z);
   cudaDeviceSynchronize();

   cudaMemcpy(accum_buffer, d_accum, accum_size, cudaMemcpyDeviceToHost);
   unsigned long long host_ray_count = 0ULL;
   cudaMemcpy(&host_ray_count, d_ray_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

   cudaFree(d_image);
   cudaFree(d_ray_count);
   return host_ray_count;
}

//==============================================================================
// NEW SCENE-BASED RENDERING (Phase 3)
//==============================================================================

/**
 * @brief CUDA kernel for rendering with scene description data structure
 * @param image Output image buffer
 * @param scene Scene data on device
 * @param width Image width
 * @param height Image height
 * @param samples_per_pixel Samples per pixel for anti-aliasing
 * @param max_depth Maximum ray bounce depth
 * @param cam_center_* Camera parameters
 * @param pixel00_* Pixel grid parameters
 * @param delta_u_* Pixel delta vectors
 * @param delta_v_* Pixel delta vectors
 * @param ray_count Ray counter
 * @param rand_states Random state per pixel
 */
// renderKernelWithScene is implemented in shaders/render_scene_kernel.cu

/**
 * @brief Host function to render using scene description
 * @param image Output image buffer on host
 * @param scene Scene structure with device pointers
 * @param width Image width
 * @param height Image height
 * @param cam_center_* Camera parameters
 * @param pixel00_* Pixel grid parameters
 * @param delta_u_* Pixel delta vectors
 * @param delta_v_* Pixel delta vectors
 * @param samples_per_pixel Samples per pixel
 * @param max_depth Maximum ray bounce depth
 * @return Total rays traced
 */
extern "C" unsigned long long renderPixelsCUDAWithScene(unsigned char *image, CudaScene::Scene *scene, int width,
                                                        int height, double cam_center_x, double cam_center_y,
                                                        double cam_center_z, double pixel00_x, double pixel00_y,
                                                        double pixel00_z, double delta_u_x, double delta_u_y,
                                                        double delta_u_z, double delta_v_x, double delta_v_y,
                                                        double delta_v_z, int samples_per_pixel, int max_depth)
{
   if (!scene)
      return 0;

   // Allocate device memory
   unsigned char *d_image;
   unsigned long long *d_ray_count;
   curandState *d_rand_states;

   int num_pixels = width * height;
   size_t image_size = num_pixels * 3 * sizeof(unsigned char);

   cudaMalloc(&d_image, image_size);
   cudaMalloc(&d_ray_count, sizeof(unsigned long long));
   cudaMalloc(&d_rand_states, num_pixels * sizeof(curandState));

   // Use optimized thread block configuration
   dim3 threads = getOptimalBlockSize(width, height);
   dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

   init_random_states<<<blocks, threads>>>(d_rand_states, num_pixels, 1234ULL, width);
   cudaDeviceSynchronize();

   // Initialize ray count
   cudaMemset(d_ray_count, 0, sizeof(unsigned long long));

   // Launch rendering kernel - pass scene pointer instead of dereferencing
   renderKernelWithScene<<<blocks, threads>>>(
       d_image, scene, width, height, samples_per_pixel, max_depth, (float)cam_center_x, (float)cam_center_y,
       (float)cam_center_z, (float)pixel00_x, (float)pixel00_y, (float)pixel00_z, (float)delta_u_x, (float)delta_u_y,
       (float)delta_u_z, (float)delta_v_x, (float)delta_v_y, (float)delta_v_z, d_ray_count, d_rand_states);

   cudaDeviceSynchronize();

   // Copy results back
   cudaMemcpy(image, d_image, image_size, cudaMemcpyDeviceToHost);

   unsigned long long host_ray_count = 0;
   cudaMemcpy(&host_ray_count, d_ray_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

   // Cleanup
   cudaFree(d_image);
   cudaFree(d_ray_count);
   cudaFree(d_rand_states);

   return host_ray_count;
}
