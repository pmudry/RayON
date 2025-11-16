/**
 * @class RendererCPU
 * @brief Single-threaded CPU renderer
 *
 * This renderer implements sequential pixel-by-pixel rendering on a single CPU core.
 * It's the simplest rendering method, useful for debugging or small images.
 */
#pragma once

#include <chrono>
#include <iostream>

#include "cpu_ray_tracer.hpp"
#include "render/render_utils.hpp"
#include "render/renderer_interface.hpp"
#include "scene_builder.hpp"

class RendererCPU : public IRenderer
{
 public:
   void render(const RenderRequest &request, RenderContext &context) override
   {
      auto start_time = std::chrono::high_resolution_clock::now();

      const CameraFrame frame = request.camera.buildFrame();
      Hittable_list cpu_scene = Scene::CPUSceneBuilder::buildCPUScene(request.scene);

      for (int y = 0; y < frame.image_height; ++y)
      {
         for (int x = 0; x < frame.image_width; ++x)
         {
            const Color pixel_color = CPURayTracer::computePixelColor(frame, cpu_scene, x, y, context.ray_counter);
            render::writePixel(request.target, x, y, pixel_color, context.gamma);
         }

         render::showProgress(y, frame.image_height);
      }

      auto end_time = std::chrono::high_resolution_clock::now();

      std::cout << std::endl;
      std::cout << "CPU single thread rendering completed in "
                << render::timeStr(end_time - start_time) << std::endl;
   }
};
