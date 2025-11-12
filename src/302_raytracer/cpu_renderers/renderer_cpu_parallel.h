/**
 * @class RendererCPUParallel
 * @brief Multi-threaded CPU renderer
 *
 * This renderer implements parallel rendering across multiple CPU cores, dividing
 * the image into horizontal chunks and processing them concurrently.
 */
#pragma once

#include "cpu_ray_tracer.h"
#include "scene_description.h"
#include "scene_builder.h"

#include <mutex>
#include <thread>

class RendererCPUParallel : virtual public CPURayTracer
{
 public:
   using CPURayTracer::CPURayTracer;

   /**
    * @brief Renders the entire image using parallel processing for improved performance
    *
    * This method divides the image into horizontal chunks and processes them concurrently
    * using multiple threads. Each thread renders a subset of rows independently, with
    * thread-safe progress tracking using a mutex.
    *
    * @param scene The hittable scene object containing all geometry to render
    * @param image Vector buffer to store the rendered RGB pixel data (modified in-place)
    */
   void renderPixelsParallel(const Scene::SceneDescription &scene, vector<unsigned char> &image)
   {
      Hittable_list cpu_scene = Scene::CPUSceneBuilder::buildCPUScene(scene);

      const int num_threads = std::thread::hardware_concurrency(); // Get the number of available hardware threads
      std::vector<std::thread> threads(num_threads);
      std::mutex progress_mutex;
      int completed_rows = 0; // Track globally completed rows

      auto start_time = std::chrono::high_resolution_clock::now();

      auto render_chunk = [&](int start_y, int end_y)
      {
         for (int y = start_y; y < end_y; ++y)
         {
            for (int x = 0; x < image_width; ++x)
            {
               // Compute the color for this pixel using the shared helper method
               Color pixel_color = computePixelColor(cpu_scene, x, y);

               // Store the computed color in the image buffer
               setPixel(image, x, y, pixel_color);
            }

            // Update progress - increment global counter and show progress
            {
               std::lock_guard<std::mutex> lock(progress_mutex);
               completed_rows++;
               showProgress(completed_rows - 1, image_height);
            }
         }
      };

      int chunk_size = image_height / num_threads;
      for (int t = 0; t < num_threads; ++t)
      {
         int start_y = t * chunk_size;
         int end_y = (t == num_threads - 1) ? image_height : start_y + chunk_size;
         threads[t] = std::thread(render_chunk, start_y, end_y);
      }

      for (auto &thread : threads)
      {
         thread.join();
      }

      showProgress(image_height - 1, image_height);

      auto end_time = std::chrono::high_resolution_clock::now();

      cout << endl;
      cout << "Parallel rendering (using " << num_threads << " threads) completed in " << timeStr(end_time - start_time)
           << endl;
   }
};
