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
#include <optional>

#include "render/render_utils.hpp"
#include "render/renderer_interface.hpp"
#include "render/benchmark_config.hpp" // Added include
#include "renderer_cuda_device.cuh"
#include "scene_builder.hpp"
#include "cuda_metrics.hpp"

class RendererCUDA : public IRenderer
{
 public:
   void setBenchmarkConfig(const Rayon::BenchmarkConfig& config) override {
       benchmark_config_ = config;
   }

   void render(const RenderRequest &request, RenderContext &context) override
   {
      auto start_time = std::chrono::high_resolution_clock::now();
      const CameraFrame frame = request.camera.buildFrame();

      CudaScene::Scene *gpu_scene = Scene::CudaSceneBuilder::buildGPUScene(request.scene);

      // Apply scene settings
      ::setBackgroundIntensity(request.scene.ambient_light);
      ::setLightIntensity(request.scene.light_intensity);
      ::setMetalFuzziness(request.scene.global_metal_fuzziness);
      ::setGlassRefractionIndex(request.scene.global_glass_ior);
      ::setDOFEnabled(request.scene.dof_enabled);
      ::setDOFAperture(request.scene.dof_aperture);
      ::setDOFFocusDistance(request.scene.dof_focus_distance);

      // Gather GPU Metrics
      getCudaDeviceMetrics(context.device_name, context.vram_usage_bytes);

      std::vector<float> accum_buffer(frame.image_width * frame.image_height * 3, 0.0f);
      void *d_rand_states = nullptr;
      void *d_accum_buffer = nullptr;

      // Determine rendering goals
      int total_samples_goal = frame.samples_per_pixel;
      float time_limit_sec = 0.0f;
      
      if (benchmark_config_.has_value()) {
          total_samples_goal = benchmark_config_->target_samples;
          time_limit_sec = benchmark_config_->max_time_seconds;
          std::cout << "Starting benchmark: Target Samples=" << total_samples_goal 
                    << ", Time Limit=" << time_limit_sec << "s\n";
      }

      // Render progressively with accumulative samples to show progress
      const int samples_per_update = std::max(1, total_samples_goal / 20); // Update progress roughly every 5%
      int samples_completed = 0;
      unsigned long long total_rays = 0;
      
      while (samples_completed < total_samples_goal)
      {
         int samples_to_add = std::min(samples_per_update, total_samples_goal - samples_completed);
         
         // In benchmark mode, ensure we don't overshoot if possible, but strict blocks are fine
         
         unsigned long long cuda_ray_count = ::renderPixelsCUDAAccumulative(
             nullptr, accum_buffer.data(), gpu_scene, frame.image_width, frame.image_height, frame.camera_center.x(),
             frame.camera_center.y(), frame.camera_center.z(), frame.pixel00_loc.x(), frame.pixel00_loc.y(),
             frame.pixel00_loc.z(), frame.pixel_delta_u.x(), frame.pixel_delta_u.y(), frame.pixel_delta_u.z(),
             frame.pixel_delta_v.x(), frame.pixel_delta_v.y(), frame.pixel_delta_v.z(), samples_to_add,
             samples_completed, frame.max_depth, &d_rand_states, &d_accum_buffer, frame.u.x(), frame.u.y(),
             frame.u.z(), frame.v.x(), frame.v.y(), frame.v.z());
         
         total_rays += cuda_ray_count;
         samples_completed += samples_to_add; // Update after render
         
         // Show progress based on samples completed
         if (!benchmark_config_.has_value()) {
             float progress = (float)samples_completed / total_samples_goal;
             int progress_steps = (int)(progress * frame.image_height);
             render::showProgress(progress_steps, frame.image_height);
         } else {
             // Simple progress for benchmark
             auto current_time = std::chrono::high_resolution_clock::now();
             double elapsed_s = std::chrono::duration<double>(current_time - start_time).count();
             std::cout << "\rBenchmark Progress: " << samples_completed << "/" << total_samples_goal 
                       << " samples | Time: " << std::fixed << std::setprecision(1) << elapsed_s << "s" << std::flush;
             
             // Time limit check
             if (time_limit_sec > 0.0f && elapsed_s >= time_limit_sec) {
                 std::cout << "\nBenchmark time limit reached (" << time_limit_sec << "s). Stopping.\n";
                 break;
             }
         }
      }

      render::convertAccumBufferToImage(request.target, accum_buffer, samples_completed, context.gamma);

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

 private:
   std::optional<Rayon::BenchmarkConfig> benchmark_config_;
};
