// OptiX GPU renderer — IRenderer implementation
// Uses hardware RT cores for BVH traversal and intersection
#pragma once

#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

#include "render/render_utils.hpp"
#include "render/renderer_interface.hpp"

namespace Scene
{
class SceneDescription;
}

// Forward declarations of OptiX host functions (defined in optix/optix_renderer.cu)
extern "C"
{
   void optixRendererInit();
   void optixRendererBuildScene(const Scene::SceneDescription &scene);
   void optixRendererResetAccum(int width, int height);
   unsigned long long optixRendererLaunch(int width, int height, int num_materials, int samples_to_add,
                                          int total_samples_so_far, int max_depth, float cam_cx, float cam_cy,
                                          float cam_cz, float p00x, float p00y, float p00z, float dux, float duy,
                                          float duz, float dvx, float dvy, float dvz, float cam_ux, float cam_uy,
                                          float cam_uz, float cam_vx, float cam_vy, float cam_vz,
                                          float bg_intensity, bool dof_enabled, float dof_aperture,
                                          float dof_focus_dist);
   void optixRendererDownloadAccum(float *host_accum_buffer, int width, int height);
   void optixRendererCleanup();
}

class RendererOptiX : public IRenderer
{
 public:
   void render(const RenderRequest &request, RenderContext &context) override
   {
      auto start_time = std::chrono::high_resolution_clock::now();
      const CameraFrame frame = request.camera.buildFrame();

      // Initialize OptiX pipeline (idempotent — only runs once)
      optixRendererInit();

      // Build acceleration structure from scene
      optixRendererBuildScene(request.scene);

      // Zero the device accumulation buffer — no host↔device transfer needed
      optixRendererResetAccum(frame.image_width, frame.image_height);

      int num_materials = static_cast<int>(request.scene.materials.size());

      // Render in batches with progress updates.
      // Key optimization: accum buffer stays entirely on GPU across batches.
      // Only a small launch-params struct (~200 bytes) is uploaded per batch.
      // Use fewer, larger batches to minimize kernel launch overhead.
      const int num_batches = 2; // 2 batches: minimal overhead while still showing progress
      const int samples_per_update = std::max(1, frame.samples_per_pixel / num_batches);
      int samples_completed = 0;
      unsigned long long total_rays = 0;

      while (samples_completed < frame.samples_per_pixel)
      {
         int samples_to_add = std::min(samples_per_update, frame.samples_per_pixel - samples_completed);
         samples_completed += samples_to_add;

         unsigned long long ray_count = optixRendererLaunch(
             frame.image_width, frame.image_height, num_materials, samples_to_add, samples_completed, frame.max_depth,
             static_cast<float>(frame.camera_center.x()), static_cast<float>(frame.camera_center.y()),
             static_cast<float>(frame.camera_center.z()), static_cast<float>(frame.pixel00_loc.x()),
             static_cast<float>(frame.pixel00_loc.y()), static_cast<float>(frame.pixel00_loc.z()),
             static_cast<float>(frame.pixel_delta_u.x()), static_cast<float>(frame.pixel_delta_u.y()),
             static_cast<float>(frame.pixel_delta_u.z()), static_cast<float>(frame.pixel_delta_v.x()),
             static_cast<float>(frame.pixel_delta_v.y()), static_cast<float>(frame.pixel_delta_v.z()),
             static_cast<float>(frame.u.x()), static_cast<float>(frame.u.y()), static_cast<float>(frame.u.z()),
             static_cast<float>(frame.v.x()), static_cast<float>(frame.v.y()), static_cast<float>(frame.v.z()),
             1.0f,    // background_intensity
             false,   // dof_enabled
             0.0f,    // dof_aperture
             10.0f);  // dof_focus_distance

         total_rays += ray_count;

         float progress = (float)samples_completed / frame.samples_per_pixel;
         int progress_steps = (int)(progress * frame.image_height);
         render::showProgress(progress_steps, frame.image_height);
      }

      // Download accumulated results from device — single transfer at the end
      // (previously this happened every batch: 5 round-trips → 1)
      std::vector<float> accum_buffer(frame.image_width * frame.image_height * 3);
      optixRendererDownloadAccum(accum_buffer.data(), frame.image_width, frame.image_height);

      render::convertAccumBufferToImage(request.target, accum_buffer, frame.samples_per_pixel, context.gamma);

      context.ray_counter.fetch_add(total_rays, std::memory_order_relaxed);

      // NOTE: We intentionally do NOT call optixRendererCleanup() here.
      // The OptiX pipeline, GAS, and SBT persist in g_state for potential reuse.
      // Cleanup happens at program exit via static destructor or OS process teardown.

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = end_time - start_time;
      std::cout << "\nOptiX rendering completed in " << render::timeStr(duration) << "\n";
   }
};
