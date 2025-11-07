/**
 * @class RendererCUDA
 * @brief CUDA GPU renderer
 *
 * This renderer implements GPU-accelerated ray tracing using CUDA, significantly
 * speeding up rendering compared to CPU-based methods.
 */
#pragma once

#include "../camera/camera_base.h"
#include <vector>

#ifdef __cplusplus
extern "C"
{
#endif
   // Host function declaration for tile-based CUDA rendering (for real-time display)
   unsigned long long renderPixelsCUDA(unsigned char *image, int width, int height, double cam_center_x,
                                       double cam_center_y, double cam_center_z, double pixel00_x, double pixel00_y,
                                       double pixel00_z, double delta_u_x, double delta_u_y, double delta_u_z,
                                       double delta_v_x, double delta_v_y, double delta_v_z, int samples_per_pixel,
                                       int max_depth);

   // Host function for accumulative rendering (adds samples to existing buffer)
   // d_rand_states_ptr: Pass pointer to nullptr to initialize, pass existing pointer to reuse
   // d_accum_buffer_ptr: Pass pointer to nullptr to initialize, pass existing pointer to reuse (stays on device)
   unsigned long long renderPixelsSDLAccumulative(unsigned char *image, float *accum_buffer,
                                                    int width, int height,
                                                    double cam_center_x, double cam_center_y, double cam_center_z,
                                                    double pixel00_x, double pixel00_y, double pixel00_z,
                                                    double delta_u_x, double delta_u_y, double delta_u_z,
                                                    double delta_v_x, double delta_v_y, double delta_v_z,
                                                    int samples_to_add, int total_samples_so_far, int max_depth,
                                                    void **d_rand_states_ptr, void **d_accum_buffer_ptr);
   
   // Helper to free device random states
   void freeDeviceRandomStates(void *d_rand_states);
   
   // Helper to free device accumulation buffer
   void freeDeviceAccumBuffer(void *d_accum_buffer);
   
   // Set global light intensity (affects area light emission)
   void setLightIntensity(float intensity);
   
   // Set background gradient intensity (sky brightness)
   void setBackgroundIntensity(float intensity);
   
   // Set metal roughness/fuzziness multiplier
   void setMetalFuzziness(float fuzziness);
   
   // Set sampling strategy (0 = uniform, 1 = stratified)
   void setStratifiedSampling(int use_stratified);

   #ifdef __cplusplus
}
#endif

// Forward declaration of CUDA functions
extern "C" unsigned long long renderPixelsCUDA(unsigned char *image, int width, int height,
                                               double cam_center_x, double cam_center_y, double cam_center_z,
                                               double pixel00_x, double pixel00_y, double pixel00_z,
                                               double delta_u_x, double delta_u_y, double delta_u_z,
                                               double delta_v_x, double delta_v_y, double delta_v_z,
                                               int samples_per_pixel, int max_depth);

class RendererCUDA : virtual public CameraBase
{
 public:
   using CameraBase::CameraBase;

   /**
    * @brief Renders the image using CUDA for parallel processing.
    *
    * This method leverages CUDA to perform ray tracing computations on the GPU,
    * significantly accelerating the rendering process. It calculates the color
    * of each pixel in the image buffer and updates the ray count.
    *
    * @param image A vector of unsigned char representing the image buffer where
    *              the rendered pixel data will be stored. The buffer must be
    *              pre-allocated with a size of (image_width * image_height * image_channels).
    */
   void renderPixelsCUDA(vector<unsigned char> &image)
   {
      auto start_time = std::chrono::high_resolution_clock::now();
      printf("CUDA renderer starting: %dx%d, %d samples, max_depth=%d\n", image_width, image_height, samples_per_pixel,
             max_depth);

      // Call CUDA rendering function with expanded parameters and get ray count back
      unsigned long long cuda_ray_count = ::renderPixelsCUDA(
          image.data(), image_width, image_height, camera_center.x(), camera_center.y(), camera_center.z(),
          pixel00_loc.x(), pixel00_loc.y(), pixel00_loc.z(), pixel_delta_u.x(), pixel_delta_u.y(), pixel_delta_u.z(),
          pixel_delta_v.x(), pixel_delta_v.y(), pixel_delta_v.z(), samples_per_pixel, max_depth);

      // Add the CUDA ray count to our atomic counter
      n_rays.fetch_add(cuda_ray_count, std::memory_order_relaxed);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = end_time - start_time;
      cout << "CUDA rendering completed in " << timeStr(duration) << endl;
   }
};
