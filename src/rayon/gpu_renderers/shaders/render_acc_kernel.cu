#include "render_acc_kernel.cuh"

//==============================================================================
// GPU-side gamma correction kernel — converts float4 accum buffer to uint8 display
//==============================================================================
//==============================================================================
// GPU-side gamma correction kernel — converts float4 accum buffer to uint8 display.
// With adaptive sampling (pixel_sample_counts != nullptr), each pixel divides by
// its own sample count rather than the global num_samples.
//==============================================================================
__global__ void gammaCorrectKernel(const float4 *__restrict__ accum_buffer, unsigned char *display_image, int width,
                                   int height, int num_samples, int channels, float gamma,
                                   const int *pixel_sample_counts)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x >= width || y >= height)
      return;

   int pixel_idx = y * width + x;
   float4 acc = accum_buffer[pixel_idx];

   // Use per-pixel sample count when adaptive sampling is active.
   // Converged pixels store a negative count (sign bit = converged flag), so use abs().
   int samples = (pixel_sample_counts != nullptr) ? abs(pixel_sample_counts[pixel_idx]) : num_samples;
   if (samples <= 0)
      samples = 1;

   float inv_samples = 1.0f / (float)samples;
   float inv_gamma = 1.0f / gamma;

   // Gamma correction + tone mapping
   float r = fminf(powf(fmaxf(acc.x * inv_samples, 0.0f), inv_gamma), 0.999f);
   float g = fminf(powf(fmaxf(acc.y * inv_samples, 0.0f), inv_gamma), 0.999f);
   float b = fminf(powf(fmaxf(acc.z * inv_samples, 0.0f), inv_gamma), 0.999f);

   int image_idx = pixel_idx * channels;
   display_image[image_idx + 0] = (unsigned char)(256.0f * r);
   display_image[image_idx + 1] = (unsigned char)(256.0f * g);
   display_image[image_idx + 2] = (unsigned char)(256.0f * b);
   if (channels == 4)
      display_image[image_idx + 3] = 255;
}

//==============================================================================
// Plasma colormap lookup (from matplotlib/viridis).
// Maps t in [0,1] to RGB. 8 control points, linearly interpolated.
//==============================================================================
__device__ void plasmaColormap(float t, float &r, float &g, float &b)
{
   // Plasma colormap control points (dark purple → blue → pink → orange → yellow)
   const int N = 8;
   const float keys[N][4] = {
      {0.000f, 0.050f, 0.030f, 0.530f}, // dark purple
      {0.143f, 0.230f, 0.015f, 0.660f}, // indigo
      {0.286f, 0.450f, 0.005f, 0.660f}, // purple-red
      {0.429f, 0.650f, 0.060f, 0.540f}, // magenta
      {0.571f, 0.820f, 0.170f, 0.360f}, // orange-red
      {0.714f, 0.940f, 0.340f, 0.170f}, // orange
      {0.857f, 0.990f, 0.560f, 0.040f}, // yellow-orange
      {1.000f, 0.940f, 0.975f, 0.130f}, // bright yellow
   };

   t = fminf(fmaxf(t, 0.0f), 1.0f);

   // Find the two surrounding control points and interpolate
   for (int i = 0; i < N - 1; ++i)
   {
      if (t <= keys[i + 1][0])
      {
         float local_t = (t - keys[i][0]) / (keys[i + 1][0] - keys[i][0]);
         r = keys[i][1] + local_t * (keys[i + 1][1] - keys[i][1]);
         g = keys[i][2] + local_t * (keys[i + 1][2] - keys[i][2]);
         b = keys[i][3] + local_t * (keys[i + 1][3] - keys[i][3]);
         return;
      }
   }
   r = keys[N - 1][1];
   g = keys[N - 1][2];
   b = keys[N - 1][3];
}

//==============================================================================
// Sample count heatmap kernel.
// Visualizes where the renderer is spending its sample budget.
// Dark purple = converged early (few samples needed), bright yellow = many samples.
//==============================================================================
__global__ void sampleHeatmapKernel(const int *pixel_sample_counts, unsigned char *display_image, int width, int height,
                                    int channels, int max_samples_for_scale)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x >= width || y >= height)
      return;

   int pixel_idx = y * width + x;

   // abs() because converged pixels store negative sample counts
   int sample_count = abs(pixel_sample_counts[pixel_idx]);

   // Normalize to [0, 1] range for colormap lookup
   float t = (max_samples_for_scale > 0) ? fminf((float)sample_count / (float)max_samples_for_scale, 1.0f) : 0.0f;

   float r, g, b;
   plasmaColormap(t, r, g, b);

   int image_idx = pixel_idx * channels;
   display_image[image_idx + 0] = (unsigned char)(255.0f * r);
   display_image[image_idx + 1] = (unsigned char)(255.0f * g);
   display_image[image_idx + 2] = (unsigned char)(255.0f * b);
   if (channels == 4)
      display_image[image_idx + 3] = 255;
}

//==============================================================================
// Path tracing kernel with optional adaptive sampling.
//
// Adaptive sampling (when pixel_sample_counts != nullptr):
//   - Each pixel tracks its own sample count
//   - After min_adaptive_samples, we check if the pixel has converged by
//     comparing the luminance change from this batch against the running average
//   - Converged pixels skip ray tracing entirely, saving GPU work
//   - Convergence is measured as: |batch_luminance - average_luminance| / average
//     If this relative change is below adaptive_threshold, the pixel is done.
//==============================================================================
__global__ void renderAccKernel(float4 *accum_buffer, const CudaScene::Scene *__restrict__ scene, int width, int height,
                                int samples_to_add, int total_samples_so_far, int max_depth, float cam_center_x,
                                float cam_center_y, float cam_center_z, float pixel00_x, float pixel00_y,
                                float pixel00_z, float delta_u_x, float delta_u_y, float delta_u_z, float delta_v_x,
                                float delta_v_y, float delta_v_z, unsigned long long *ray_count,
                                curandState *rand_states, float cam_u_x, float cam_u_y, float cam_u_z, float cam_v_x,
                                float cam_v_y, float cam_v_z,
                                int *pixel_sample_counts, int min_adaptive_samples, float adaptive_threshold)
{
#ifdef DIAGS
   // Shared memory for block-level ray counting (only when diagnostics enabled)
   __shared__ unsigned long long block_ray_count;

   if (threadIdx.x == 0 && threadIdx.y == 0)
   {
      block_ray_count = 0;
   }
   __syncthreads();
#endif

   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x >= width || y >= height)
      return;

   int pixel_idx = y * width + x;

   if (pixel_idx >= width * height)
      return;

   // --- Adaptive sampling: skip converged pixels ---
   // A pixel is "converged" when its per-pixel sample count is negative (we use
   // the sign bit as a convergence flag to avoid an extra buffer).
   bool adaptive_enabled = (pixel_sample_counts != nullptr);
   int pixel_samples = 0;

   if (adaptive_enabled)
   {
      pixel_samples = pixel_sample_counts[pixel_idx];
      if (pixel_samples < 0)
         return; // Already converged — nothing to do
   }

   curandState *local_rand_state = &rand_states[pixel_idx];

   f3 camera_center(cam_center_x, cam_center_y, cam_center_z);
   f3 pixel00_loc(pixel00_x, pixel00_y, pixel00_z);
   f3 pixel_delta_u(delta_u_x, delta_u_y, delta_u_z);
   f3 pixel_delta_v(delta_v_x, delta_v_y, delta_v_z);
   f3 cam_u(cam_u_x, cam_u_y, cam_u_z);
   f3 cam_v(cam_v_x, cam_v_y, cam_v_z);

   // Single coalesced float4 read instead of 3 separate float reads
   float4 acc = accum_buffer[pixel_idx];
   f3 accumulated_color(acc.x, acc.y, acc.z);

   // Snapshot the accumulated luminance BEFORE this batch (for convergence check)
   float lum_before = 0.0f;
   if (adaptive_enabled && pixel_samples > 0)
   {
      float inv_s = 1.0f / (float)pixel_samples;
      lum_before = 0.299f * acc.x * inv_s + 0.587f * acc.y * inv_s + 0.114f * acc.z * inv_s;
   }

#ifdef DIAGS
   int local_ray_count = 0;
#endif

   // --- Trace new samples ---
   for (int s = 0; s < samples_to_add; s++)
   {
      float offset_u = rand_float(local_rand_state) - 0.5f;
      float offset_v = rand_float(local_rand_state) - 0.5f;

      f3 pixel_center = pixel00_loc + ((float)x + offset_u) * pixel_delta_u + ((float)y + offset_v) * pixel_delta_v;
      f3 ray_direction = pixel_center - camera_center;

      // Apply dof
      f3 ray_origin = camera_center;

      if (g_dof_enabled && g_dof_aperture > 0.0f)
      {
         // Find focus point along original ray direction
         f3 normalized_dir = normalize(ray_direction);
         f3 focus_point = camera_center + g_dof_focus_distance * normalized_dir;

         // Offset ray origin randomly on aperture disk
         f3 aperture_offset = sample_aperture_disk(cam_u, cam_v, local_rand_state);
         ray_origin = camera_center + aperture_offset;

         // Ray from offset origin to focus point
         ray_direction = focus_point - ray_origin;
      }

      ray_simple r(ray_origin, ray_direction);
      accumulated_color = accumulated_color + ray_color(r, *scene, local_rand_state, max_depth
#ifdef DIAGS
                                                        ,
                                                        local_ray_count
#endif
      );
   }

#ifdef DIAGS
   // Block-level atomic accumulation (only when diagnostics enabled)
   atomicAdd(&block_ray_count, (unsigned long long)local_ray_count);
#endif

   // Single coalesced float4 write instead of 3 separate float writes
   accum_buffer[pixel_idx] = make_float4(accumulated_color.x, accumulated_color.y, accumulated_color.z, 0.0f);

   // --- Adaptive sampling: check convergence after this batch ---
   if (adaptive_enabled)
   {
      int new_sample_count = pixel_samples + samples_to_add;

      if (new_sample_count >= min_adaptive_samples)
      {
         // Compare average luminance before and after this batch
         float inv_s = 1.0f / (float)new_sample_count;
         float lum_after = 0.299f * accumulated_color.x * inv_s
                         + 0.587f * accumulated_color.y * inv_s
                         + 0.114f * accumulated_color.z * inv_s;

         // Relative change in luminance — if adding more samples barely changes
         // the pixel's appearance, it has converged
         float relative_change = fabsf(lum_after - lum_before) / fmaxf(lum_after, 0.001f);

         if (relative_change < adaptive_threshold)
         {
            // Mark as converged: store negative sample count (sign bit = converged flag)
            pixel_sample_counts[pixel_idx] = -new_sample_count;
         }
         else
         {
            pixel_sample_counts[pixel_idx] = new_sample_count;
         }
      }
      else
      {
         pixel_sample_counts[pixel_idx] = new_sample_count;
      }
   }

#ifdef DIAGS
   __syncthreads();

   // Single global atomic per block
   if (threadIdx.x == 0 && threadIdx.y == 0)
   {
      atomicAdd(ray_count, block_ray_count);
   }
#endif
}
