#include "render_acc_kernel.cuh"

__global__ void renderAccKernel(float *accum_buffer, unsigned char *image, const CudaScene::Scene *__restrict__ scene,
                                int width, int height, int samples_to_add, int total_samples_so_far, int max_depth,
                                float cam_center_x, float cam_center_y, float cam_center_z, float pixel00_x,
                                float pixel00_y, float pixel00_z, float delta_u_x, float delta_u_y, float delta_u_z,
                                float delta_v_x, float delta_v_y, float delta_v_z, unsigned long long *ray_count,
                                curandState *rand_states, float cam_u_x, float cam_u_y, float cam_u_z, float cam_v_x,
                                float cam_v_y, float cam_v_z)
{
   // Calculate pixel coordinates
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;


   int pixel_idx = y * width + x;
   int base_idx = pixel_idx * 3; // a base is here a triplet of RGB values for each pixel

   int thread_ray_count = 0;
   bool valid_pixel = (x < width) && (y < height); // to calculate only once

   if (valid_pixel)
   {
      // Get local random state for this pixel
      curandState *local_rand_state = &rand_states[pixel_idx];

      // Camera parameters
      f3 camera_center(cam_center_x, cam_center_y, cam_center_z);
      f3 pixel00_loc(pixel00_x, pixel00_y, pixel00_z);
      f3 pixel_delta_u(delta_u_x, delta_u_y, delta_u_z);
      f3 pixel_delta_v(delta_v_x, delta_v_y, delta_v_z);
      f3 cam_u(cam_u_x, cam_u_y, cam_u_z);
      f3 cam_v(cam_v_x, cam_v_y, cam_v_z);
      f3 accumulated_color(accum_buffer[base_idx], accum_buffer[base_idx + 1], accum_buffer[base_idx + 2]);

      
      for (int s = 0; s < samples_to_add; s++) 
      {  
         //jitter within the pixel for anti-aliasing
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
         accumulated_color = accumulated_color + ray_color(r, *scene, local_rand_state, max_depth, thread_ray_count);
      }

      // writing back to global memory
      accum_buffer[base_idx] = accumulated_color.x;
      accum_buffer[base_idx + 1] = accumulated_color.y;
      accum_buffer[base_idx + 2] = accumulated_color.z;
      image[base_idx] = 0;
      image[base_idx + 1] = 0;
      image[base_idx + 2] = 0;
   }

   // warp shuffling optimization
   unsigned mask = 0xffffffff; // All threads are active
   int lane = threadIdx.x % 32; // Lane index within the warp

   // safety transfer
   int warp_sum = thread_ray_count;

   // "Butterfly" Reduction
   warp_sum += __shfl_down_sync(mask, warp_sum, 16);
   warp_sum += __shfl_down_sync(mask, warp_sum, 8);
   warp_sum += __shfl_down_sync(mask, warp_sum, 4);
   warp_sum += __shfl_down_sync(mask, warp_sum, 2);
   warp_sum += __shfl_down_sync(mask, warp_sum, 1);

   // One atomicAdd per warp
   if (lane == 0)
   {
      atomicAdd(ray_count, (unsigned long long)warp_sum);
   }
   
}
