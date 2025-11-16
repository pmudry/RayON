/**
 * @class RendererCPUParallel
 * @brief Multi-threaded CPU renderer
 *
 * This renderer implements parallel rendering across multiple CPU cores, dividing
 * the image into horizontal chunks and processing them concurrently.
 */
#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "cpu_ray_tracer.hpp"
#include "render/render_utils.hpp"
#include "render/renderer_interface.hpp"
#include "scene_builder.hpp"

class RendererCPUParallel : public IRenderer
{
 public:
   void render(const RenderRequest &request, RenderContext &context) override
   {
      Hittable_list cpu_scene = Scene::CPUSceneBuilder::buildCPUScene(request.scene);
      const CameraFrame frame = request.camera.buildFrame();

      const unsigned hw_threads = std::thread::hardware_concurrency();
      const int num_threads = static_cast<int>(std::max(1u, hw_threads));
      std::vector<std::thread> threads(num_threads);
      std::mutex progress_mutex;
      int completed_rows = 0;

      auto start_time = std::chrono::high_resolution_clock::now();

      auto render_chunk = [&](int start_y, int end_y)
      {
         for (int y = start_y; y < end_y; ++y)
         {
            for (int x = 0; x < frame.image_width; ++x)
            {
               const Color pixel_color = CPURayTracer::computePixelColor(frame, cpu_scene, x, y, context.ray_counter);
               render::writePixel(request.target, x, y, pixel_color, context.gamma);
            }

            {
               std::lock_guard<std::mutex> lock(progress_mutex);
               completed_rows++;
               render::showProgress(completed_rows - 1, frame.image_height);
            }
         }
      };

      const int rows_per_thread = std::max(1, frame.image_height / num_threads);
      for (int t = 0; t < num_threads; ++t)
      {
         int start_y = t * rows_per_thread;
         int end_y = (t == num_threads - 1) ? frame.image_height : std::min(frame.image_height, start_y + rows_per_thread);
         threads[t] = std::thread(render_chunk, start_y, end_y);
      }

      for (auto &thread : threads)
      {
         thread.join();
      }

      auto end_time = std::chrono::high_resolution_clock::now();

      std::cout << std::endl;
      std::cout << "Parallel rendering (using " << num_threads << " threads) completed in "
                << render::timeStr(end_time - start_time) << std::endl;
   }
};
