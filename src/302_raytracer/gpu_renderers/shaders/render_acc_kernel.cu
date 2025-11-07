#include "render_acc_kernel.cuh"

__global__ void renderAccKernel(float *accum_buffer, unsigned char *image, const CudaScene::Scene * __restrict__ scene, int width,
                                int height, int samples_to_add, int total_samples_so_far, int max_depth,
                                float cam_center_x, float cam_center_y, float cam_center_z, float pixel00_x,
                                float pixel00_y, float pixel00_z, float delta_u_x, float delta_u_y, float delta_u_z,
                                float delta_v_x, float delta_v_y, float delta_v_z, unsigned long long *ray_count,
                                curandState *rand_states)
{
   // Shared memory for block-level ray counting
   __shared__ unsigned long long block_ray_count;
   
   if (threadIdx.x == 0 && threadIdx.y == 0) {
      block_ray_count = 0;
   }
   __syncthreads();
   
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   if (x >= width || y >= height) return;
   int pixel_idx = y * width + x;
   int base_idx = pixel_idx * 3;
   if (pixel_idx >= width * height || base_idx + 2 >= width * height * 3) return;

   curandState *local_rand_state = &rand_states[pixel_idx];
   float3_simple camera_center(cam_center_x, cam_center_y, cam_center_z);
   float3_simple pixel00_loc(pixel00_x, pixel00_y, pixel00_z);
   float3_simple pixel_delta_u(delta_u_x, delta_u_y, delta_u_z);
   float3_simple pixel_delta_v(delta_v_x, delta_v_y, delta_v_z);
   float3_simple accumulated_color(accum_buffer[base_idx], accum_buffer[base_idx + 1], accum_buffer[base_idx + 2]);
   int local_ray_count = 0;
   for (int s = 0; s < samples_to_add; s++)
   {
      float offset_u = rand_float(local_rand_state) - 0.5f;
      float offset_v = rand_float(local_rand_state) - 0.5f;
      float3_simple pixel_center = pixel00_loc + ((float)x + offset_u) * pixel_delta_u + ((float)y + offset_v) * pixel_delta_v;
      float3_simple ray_direction = pixel_center - camera_center;
      ray_simple r(camera_center, ray_direction);
      accumulated_color = accumulated_color + ray_color(r, *scene, local_rand_state, min(max_depth, 6), local_ray_count);
   }
   
   // Block-level atomic accumulation
   atomicAdd(&block_ray_count, (unsigned long long)local_ray_count);
   
   accum_buffer[base_idx] = accumulated_color.x;
   accum_buffer[base_idx + 1] = accumulated_color.y;
   accum_buffer[base_idx + 2] = accumulated_color.z;
   image[base_idx] = 0; image[base_idx + 1] = 0; image[base_idx + 2] = 0;
   
   __syncthreads();
   
   // Single global atomic per block
   if (threadIdx.x == 0 && threadIdx.y == 0) {
      atomicAdd(ray_count, block_ray_count);
   }
}
