/**
 * @class RendererCUDA
 * @brief CUDA GPU renderer
 *
 * This renderer implements GPU-accelerated ray tracing using CUDA, significantly
 * speeding up rendering compared to CPU-based methods.
 */
#pragma once

#include "camera/camera_base.h"
#include "scene_description.h"
#include "scene_builder.h"
#include <vector>

// Forward declaration for CudaScene::Scene
namespace CudaScene {
    struct Scene;
}

// Forward declarations of CUDA functions
extern "C"
{
   // Host function for accumulative CUDA rendering (used for both one-shot and progressive rendering)
   // For one-shot rendering: call once with samples_to_add = total samples, total_samples_so_far = 0
   // For progressive rendering: call multiple times, incrementing total_samples_so_far each time
   unsigned long long renderPixelsCUDAAccumulative(
       unsigned char *image, float *accum_buffer, CudaScene::Scene* scene, int width, int height,
       double cam_center_x, double cam_center_y, double cam_center_z,
       double pixel00_x, double pixel00_y, double pixel00_z,
       double delta_u_x, double delta_u_y, double delta_u_z,
       double delta_v_x, double delta_v_y, double delta_v_z,
       int samples_to_add, int total_samples_so_far, int max_depth,
       void **d_rand_states_ptr, void **d_accum_buffer_ptr,
       double cam_u_x, double cam_u_y, double cam_u_z,
       double cam_v_x, double cam_v_y, double cam_v_z);
   
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
   
   // Set glass refraction index
   void setGlassRefractionIndex(float index);
   
   // Depth of Field controls
   void setDOFEnabled(bool enabled);
   void setDOFAperture(float aperture);
   void setDOFFocusDistance(float distance);
}

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
   void renderPixelsCUDA(const Scene::SceneDescription &scene, vector<unsigned char> &image)
   {
      using namespace Scene;
      
      auto start_time = std::chrono::high_resolution_clock::now();
      printf("CUDA renderer starting: %dx%d, %d samples, max_depth=%d\n", image_width, image_height, samples_per_pixel,
             max_depth);
      
      CudaScene::Scene* gpu_scene = CudaSceneBuilder::buildGPUScene(scene);

      // Allocate accumulation buffer for one-shot rendering
      // Uses renderPixelsCUDAAccumulative with samples_to_add = total samples, total_samples_so_far = 0
      // This provides unified rendering path with progressive renderer
      std::vector<float> accum_buffer(image_width * image_height * 3, 0.0f);
      void* d_rand_states = nullptr;
      void* d_accum_buffer = nullptr;

      // Call accumulative CUDA rendering function in one-shot mode
      // Note: First parameter (image) is unused by kernel - only accum_buffer is updated
      unsigned long long cuda_ray_count = ::renderPixelsCUDAAccumulative(
          nullptr, accum_buffer.data(), gpu_scene, image_width, image_height,
          camera_center.x(), camera_center.y(), camera_center.z(),
          pixel00_loc.x(), pixel00_loc.y(), pixel00_loc.z(),
          pixel_delta_u.x(), pixel_delta_u.y(), pixel_delta_u.z(),
          pixel_delta_v.x(), pixel_delta_v.y(), pixel_delta_v.z(),
          samples_per_pixel, samples_per_pixel, max_depth,
          &d_rand_states, &d_accum_buffer,
          u.x(), u.y(), u.z(),
          v.x(), v.y(), v.z());

      // Convert accumulation buffer to final image with gamma correction
      // The accumulative kernel only updates the accumulation buffer, not the final image
      convertAccumBufferToImage(image, accum_buffer, samples_per_pixel, 2.0f);

      // Cleanup device memory
      if (d_rand_states) freeDeviceRandomStates(d_rand_states);
      if (d_accum_buffer) freeDeviceAccumBuffer(d_accum_buffer);

      // Free GPU scene
      CudaSceneBuilder::freeGPUScene(gpu_scene);

      // Add the CUDA ray count to our atomic counter
      n_rays.fetch_add(cuda_ray_count, std::memory_order_relaxed);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = end_time - start_time;
      cout << "CUDA rendering completed in " << timeStr(duration) << endl;
   }

private:
   /**
    * @brief Create the default rendering scene - implemented in main.cc
    * Uses scene_file from CameraBase if set, otherwise creates default scene
    */
   Scene::SceneDescription createDefaultScene();
};
