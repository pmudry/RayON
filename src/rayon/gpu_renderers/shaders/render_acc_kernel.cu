#include "render_acc_kernel.cuh"

__global__ void renderAccKernel(float4 *accum_buffer, const CudaScene::Scene *__restrict__ scene, int width, int height,
                                int samples_to_add, int total_samples_so_far, int max_depth, float cam_center_x,
                                float cam_center_y, float cam_center_z, float pixel00_x, float pixel00_y,
                                float pixel00_z, float delta_u_x, float delta_u_y, float delta_u_z, float delta_v_x,
                                float delta_v_y, float delta_v_z, unsigned long long *ray_count,
                                curandState *rand_states, float cam_u_x, float cam_u_y, float cam_u_z, float cam_v_x,
                                float cam_v_y, float cam_v_z)
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

#ifdef DIAGS
   int local_ray_count = 0;
#endif

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

#ifdef DIAGS
   __syncthreads();

   // Single global atomic per block
   if (threadIdx.x == 0 && threadIdx.y == 0)
   {
      atomicAdd(ray_count, block_ray_count);
   }
#endif
}
