#include "render_scene_kernel.cuh"

__global__ void renderKernel(unsigned char *image, const CudaScene::Scene *__restrict__ scene, int width, int height,
                             int samples_per_pixel, int max_depth, float cam_center_x, float cam_center_y,
                             float cam_center_z, float pixel00_x, float pixel00_y, float pixel00_z, float delta_u_x,
                             float delta_u_y, float delta_u_z, float delta_v_x, float delta_v_y, float delta_v_z,
                             unsigned long long *ray_count, curandState *rand_states)
{
   // Shared memory for block-level ray counting (reduces global atomics)
   __shared__ unsigned long long block_ray_count;

   // Initialize shared counter (only first thread)
   if (threadIdx.x == 0 && threadIdx.y == 0)
   {
      block_ray_count = 0;
   }

   __syncthreads();

   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x >= width || y >= height)
      return;

   int pixel_index = y * width + x;

   curandState *local_rand_state = &rand_states[pixel_index];

   f3 cam_center(cam_center_x, cam_center_y, cam_center_z);
   f3 pixel00(pixel00_x, pixel00_y, pixel00_z);
   f3 delta_u(delta_u_x, delta_u_y, delta_u_z);
   f3 delta_v(delta_v_x, delta_v_y, delta_v_z);
   f3 pixel_color(0, 0, 0);

   int local_ray_count = 0;

   // Sample multiple rays per pixel for anti-aliasing and accuracy
   for (int s = 0; s < samples_per_pixel; ++s)
   {
      // Generate random offsets within the pixel for stochastic sampling
      float offset_u = rand_float(local_rand_state) - 0.5f;
      float offset_v = rand_float(local_rand_state) - 0.5f;

      // Calculate the pixel center with jittered offsets for anti-aliasing
      // Combines pixel00 base position with x/y offsets scaled by delta_u/delta_v
      f3 pixel_center = pixel00 +
                        f3((x + offset_u) * delta_u.x, (x + offset_u) * delta_u.y, (x + offset_u) * delta_u.z) +
                        f3((y + offset_v) * delta_v.x, (y + offset_v) * delta_v.y, (y + offset_v) * delta_v.z);

      // Compute ray direction from camera center to the sampled pixel location
      f3 ray_direction =
          f3(pixel_center.x - cam_center.x, pixel_center.y - cam_center.y, pixel_center.z - cam_center.z);

      // Construct the ray and trace it through the scene
      ray_simple r(cam_center, ray_direction);

      // Accumulate color contribution from this sample (also updates local_ray_count)
      pixel_color = pixel_color + ray_color(r, *scene, local_rand_state, max_depth, local_ray_count);
   }

   float scale = 1.0f / samples_per_pixel;
   pixel_color = f3(pixel_color.x * scale, pixel_color.y * scale, pixel_color.z * scale);

   pixel_color.x = sqrtf(pixel_color.x);
   pixel_color.y = sqrtf(pixel_color.y);
   pixel_color.z = sqrtf(pixel_color.z);
   
   pixel_color.x = fminf(fmaxf(pixel_color.x, 0.0f), 1.0f);
   pixel_color.y = fminf(fmaxf(pixel_color.y, 0.0f), 1.0f);
   pixel_color.z = fminf(fmaxf(pixel_color.z, 0.0f), 1.0f);

   int idx = pixel_index * 3;
   image[idx + 0] = static_cast<unsigned char>(pixel_color.x * 255.0f);
   image[idx + 1] = static_cast<unsigned char>(pixel_color.y * 255.0f);
   image[idx + 2] = static_cast<unsigned char>(pixel_color.z * 255.0f);

   // Accumulate to block-level counter (shared memory atomic - much faster)
   atomicAdd(&block_ray_count, (unsigned long long)local_ray_count);
   __syncthreads();

   // Single thread per block writes to global memory
   if (threadIdx.x == 0 && threadIdx.y == 0)
   {
      atomicAdd(ray_count, block_ray_count);
   }
}