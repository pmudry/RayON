/**
 * @class RendererCUDA
 * @brief CUDA GPU renderer
 *
 * This renderer implements GPU-accelerated ray tracing using CUDA, significantly
 * speeding up rendering compared to CPU-based methods.
 */
#pragma once

#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>

#include "render/render_utils.hpp"
#include "render/renderer_interface.hpp"
#include "renderer_cuda_device.cuh"
#include "scene_builder.hpp"

class RendererCUDA : public IRenderer
{
 public:
   void render(const RenderRequest &request, RenderContext &context) override
   {
      auto start_time = std::chrono::high_resolution_clock::now();
      const CameraFrame frame = request.camera.buildFrame();

      CudaScene::Scene *gpu_scene = Scene::CudaSceneBuilder::buildGPUScene(request.scene);

      std::vector<float> accum_buffer(frame.image_width * frame.image_height * 3, 0.0f);
      void *d_rand_states = nullptr;
      void *d_accum_buffer = nullptr;

      // Render progressively with accumulative samples to show progress
      const int samples_per_update = frame.samples_per_pixel / 25; // Update progress every 1% of samples
      int samples_completed = 0;
      unsigned long long total_rays = 0;
      
      while (samples_completed < frame.samples_per_pixel)
      {
         int samples_to_add = std::min(samples_per_update, frame.samples_per_pixel - samples_completed);
         samples_completed += samples_to_add;
         
         unsigned long long cuda_ray_count = ::renderPixelsCUDAAccumulative(
             nullptr, accum_buffer.data(), gpu_scene, frame.image_width, frame.image_height, frame.camera_center.x(),
             frame.camera_center.y(), frame.camera_center.z(), frame.pixel00_loc.x(), frame.pixel00_loc.y(),
             frame.pixel00_loc.z(), frame.pixel_delta_u.x(), frame.pixel_delta_u.y(), frame.pixel_delta_u.z(),
             frame.pixel_delta_v.x(), frame.pixel_delta_v.y(), frame.pixel_delta_v.z(), samples_to_add,
             samples_completed, frame.max_depth, &d_rand_states, &d_accum_buffer, frame.u.x(), frame.u.y(),
             frame.u.z(), frame.v.x(), frame.v.y(), frame.v.z());
         
         total_rays += cuda_ray_count;
         
         // Show progress based on samples completed
         float progress = (float)samples_completed / frame.samples_per_pixel;
         int progress_steps = (int)(progress * frame.image_height);
         render::showProgress(progress_steps, frame.image_height);
      }

      render::convertAccumBufferToImage(request.target, accum_buffer, frame.samples_per_pixel, context.gamma);

      if (d_rand_states)
         freeDeviceRandomStates(d_rand_states);
      if (d_accum_buffer)
         freeDeviceAccumBuffer(d_accum_buffer);

      Scene::CudaSceneBuilder::freeGPUScene(gpu_scene);

      context.ray_counter.fetch_add(total_rays, std::memory_order_relaxed);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = end_time - start_time;
      std::cout << "\nCUDA rendering completed in " << render::timeStr(duration) << "\n";
   }
};
