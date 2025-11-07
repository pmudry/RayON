#include "render_scene_kernel.cuh"

__global__ void renderKernelWithScene(unsigned char *image, const CudaScene::Scene * __restrict__ scene, int width, int height,
                                      int samples_per_pixel, int max_depth, float cam_center_x, float cam_center_y,
                                      float cam_center_z, float pixel00_x, float pixel00_y, float pixel00_z,
                                      float delta_u_x, float delta_u_y, float delta_u_z, float delta_v_x,
                                      float delta_v_y, float delta_v_z, unsigned long long *ray_count,
                                      curandState *rand_states)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   if (x >= width || y >= height) return;
   int pixel_index = y * width + x;
   curandState *local_state = &rand_states[pixel_index];
   float3_simple cam_center(cam_center_x, cam_center_y, cam_center_z);
   float3_simple pixel00(pixel00_x, pixel00_y, pixel00_z);
   float3_simple delta_u(delta_u_x, delta_u_y, delta_u_z);
   float3_simple delta_v(delta_v_x, delta_v_y, delta_v_z);
   float3_simple pixel_color(0, 0, 0);
   int local_ray_count = 0;
   for (int s = 0; s < samples_per_pixel; ++s)
   {
      float offset_u = rand_float(local_state) - 0.5f;
      float offset_v = rand_float(local_state) - 0.5f;
      float3_simple pixel_center = pixel00 + float3_simple((x + offset_u) * delta_u.x, (x + offset_u) * delta_u.y, (x + offset_u) * delta_u.z) +
                                   float3_simple((y + offset_v) * delta_v.x, (y + offset_v) * delta_v.y, (y + offset_v) * delta_v.z);
      float3_simple ray_direction = float3_simple(pixel_center.x - cam_center.x, pixel_center.y - cam_center.y, pixel_center.z - cam_center.z);
      ray_simple r(cam_center, ray_direction);
      pixel_color = pixel_color + ray_color(r, *scene, local_state, min(max_depth, 6), local_ray_count);
   }
   float scale = 1.0f / samples_per_pixel;
   pixel_color = float3_simple(pixel_color.x * scale, pixel_color.y * scale, pixel_color.z * scale);
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
   atomicAdd(ray_count, (unsigned long long)local_ray_count);
}
