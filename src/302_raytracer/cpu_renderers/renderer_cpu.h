/**
 * @class RendererCPU
 * @brief Single-threaded CPU renderer
 *
 * This renderer implements sequential pixel-by-pixel rendering on a single CPU core.
 * It's the simplest rendering method, useful for debugging or small images.
 */
#pragma once

#include "cpu_ray_tracer.h"
#include "scene_description.h"
#include "scene_builder.h"

class RendererCPU : virtual public CPURayTracer
{
 public:
   using CPURayTracer::CPURayTracer;

   /**
    * @brief Renders the entire image sequentially pixel by pixel using ray tracing
    *
    * This method performs sequential rendering by iterating through each pixel in the image
    * from top-left to bottom-right.
    *
    * @param scene The hittable scene object containing all geometry to render
    * @param image Vector buffer to store the rendered RGB pixel data (modified in-place)
    */
   void renderPixels(const Scene::SceneDescription &scene, vector<unsigned char> &image)
   {
      auto start_time = std::chrono::high_resolution_clock::now();

      Hittable_list cpu_scene = Scene::CPUSceneBuilder::buildCPUScene(scene);

      // Render each pixel in the image sequentially
      for (int y = 0; y < image_height; ++y)
      {
         for (int x = 0; x < image_width; ++x)
         {
            // Compute the color for this pixel using ray tracing with anti-aliasing
            Color pixel_color = computePixelColor(cpu_scene, x, y);

            // Store the computed color in the image buffer
            setPixel(image, x, y, pixel_color);
         }

         // Show progress after completing each row
         showProgress(y, image_height);
      }

      showProgress(image_height - 1, image_height);

      auto end_time = std::chrono::high_resolution_clock::now();

      cout << endl;
      cout << "CPU single thread rendering completed in " << timeStr(end_time - start_time) << endl;
   }
};
