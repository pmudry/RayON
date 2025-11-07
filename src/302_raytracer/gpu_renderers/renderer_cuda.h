/**
 * @class RendererCUDA
 * @brief CUDA GPU renderer
 *
 * This renderer implements GPU-accelerated ray tracing using CUDA, significantly
 * speeding up rendering compared to CPU-based methods.
 */
#pragma once

#include "../camera/camera_base.h"
#include "../scene_description.h"
#include "../scene_builder.h"
#include <vector>

// Forward declaration for CudaScene::Scene
namespace CudaScene {
    struct Scene;
}

// Forward declarations of CUDA functions
extern "C"
{
   // Host function for scene-based CUDA rendering
   unsigned long long renderPixelsCUDAWithScene(
       unsigned char *image, CudaScene::Scene* scene, int width, int height,
       double cam_center_x, double cam_center_y, double cam_center_z,
       double pixel00_x, double pixel00_y, double pixel00_z,
       double delta_u_x, double delta_u_y, double delta_u_z,
       double delta_v_x, double delta_v_y, double delta_v_z,
       int samples_per_pixel, int max_depth);
   
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
   void renderPixelsCUDA(vector<unsigned char> &image)
   {
      using namespace Scene;
      
      auto start_time = std::chrono::high_resolution_clock::now();
      printf("CUDA renderer starting: %dx%d, %d samples, max_depth=%d\n", image_width, image_height, samples_per_pixel,
             max_depth);

      // Build the scene description (create default scene)
      SceneDescription scene_desc = createDefaultScene();
      
      // Convert to GPU scene
      CudaScene::Scene* gpu_scene = CudaSceneBuilder::buildGPUScene(scene_desc);

      // Call new scene-based CUDA rendering function
      unsigned long long cuda_ray_count = ::renderPixelsCUDAWithScene(
          image.data(), gpu_scene, image_width, image_height,
          camera_center.x(), camera_center.y(), camera_center.z(),
          pixel00_loc.x(), pixel00_loc.y(), pixel00_loc.z(),
          pixel_delta_u.x(), pixel_delta_u.y(), pixel_delta_u.z(),
          pixel_delta_v.x(), pixel_delta_v.y(), pixel_delta_v.z(),
          samples_per_pixel, max_depth);

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
    */
   static Scene::SceneDescription createDefaultScene();
};
