/**
 * @class RendererOptiXProgressive
 * @brief Interactive SDL renderer with progressive sample accumulation using OptiX hardware RT cores.
 *
 * Mirrors RendererCUDAProgressive but drives OptiX instead of the CUDA path-tracer.
 * All SDL/GUI/camera-control infrastructure is shared verbatim; only renderBatch()
 * and device-memory management differ.
 *
 * Key differences from the CUDA progressive renderer:
 *  - No d_rand_states / d_accum_buffer / gpu_scene pointers — OptiX manages its own device memory.
 *  - Scene and BVH are built once via optixRendererBuildScene() and persist in g_state.
 *  - Camera change reset → optixRendererResetAccum() instead of freeing/reallocating CUDA buffers.
 *  - Per-frame render → optixRendererLaunch() + optixRendererDownloadAccum().
 *  - light_intensity / metal_fuzziness / glass_refraction_index sliders are visible in the UI
 *    but have no effect (OptiX bakes those into the SBT at scene-build time).
 */
#pragma once

#if defined(SDL2_FOUND) && defined(OPTIX_FOUND)

#include <SDL.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "render/render_utils.hpp"
#include "render/renderer_interface.hpp"
#include "sdl_gui_controls.hpp"
#include "sdl_gui_handler.hpp"

// Forward declarations of OptiX host functions (implemented in optix/optix_renderer.cu)
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

class RendererOptiXProgressive : public IRenderer
{
 public:
   struct Settings
   {
      int samples_per_batch = 8;
      bool auto_accumulate = true;
      int target_fps = 60;
      bool adaptive_depth = false;
   };

   RendererOptiXProgressive() = default;
   explicit RendererOptiXProgressive(Settings settings) : settings_(settings) {}

   void setSettings(const Settings &settings) { settings_ = settings; }

   void render(const RenderRequest &request, RenderContext &context) override
   {
      int samples_per_batch = settings_.samples_per_batch;
      bool auto_accumulate = settings_.auto_accumulate;
      int target_fps = settings_.target_fps;
      bool adaptive_depth = settings_.adaptive_depth;

      auto &camera = request.camera;
      auto &scene = request.scene;
      RenderTargetView target = request.target;

      Point3 &look_from = camera.look_from;
      Point3 &look_at = camera.look_at;
      Vec3 &vup = camera.vup;
      CameraFrame frame = camera.buildFrame();
      Vec3 basis_w = frame.w;

      const int image_width = target.width;
      const int image_height = target.height;
      const int image_channels = target.channels;
      const int num_materials = static_cast<int>(scene.materials.size());

      auto refreshCameraFrame = [&]()
      {
         camera.updateFrame();
         frame = camera.buildFrame();
         basis_w = frame.w;
      };

      refreshCameraFrame();

      // Initialize GUI
      SDLGuiHandler gui(target.width, target.height);
      if (!gui.initialize())
         return;
      int max_samples = camera.samples_per_pixel;

      gui.printControls(samples_per_batch, max_samples, auto_accumulate);

      // Initialize camera controls
      CameraControlHandler camera_control;
      camera_control.initializeCameraControls(look_from, look_at);

      // Ray-tracing state
      bool running = true;
      bool camera_changed = true;
      bool accumulation_enabled = auto_accumulate;
      int current_samples = 0;
      float gamma = 2.0f;
      float light_intensity = 1.0f;       // UI only for OptiX — not forwarded to launch
      float background_intensity = 1.0f;
      float metal_fuzziness = 1.0f;       // UI only for OptiX — not forwarded to launch
      float glass_refraction_index = 1.5f; // UI only for OptiX — not forwarded to launch
      bool dof_enabled = false;
      float dof_aperture = 0.1f;
      float dof_focus_distance = 10.0f;
      bool needs_rerender = false;
      bool force_immediate_render = false;
      float samples_per_batch_float = static_cast<float>(samples_per_batch);

      // Motion detection
      bool is_camera_moving = false;
      auto last_camera_change_time = std::chrono::high_resolution_clock::now();
      const float motion_cooldown_seconds = 0.5f;

      int adaptive_samples_per_batch = samples_per_batch;
      int user_samples_per_batch = samples_per_batch;
      const float target_frame_time_ms = 1000.0f / target_fps;
      (void)target_frame_time_ms; // suppress unused warning

      auto syncSamplesFromSlider = [&]()
      { samples_per_batch = std::max(1, static_cast<int>(samples_per_batch_float)); };

      // No-op for light/fuzz/ior: those are baked into the OptiX SBT at scene-build time.
      // DOF and background are passed per-launch below.
      auto propagateAccumulationToggle = [&]()
      {
         if (accumulation_enabled != auto_accumulate)
            auto_accumulate = accumulation_enabled;
      };

      // UI slider state
      SliderBounds samples_slider_bounds = {0, 0, 0, 0, 1.0f, 256.0f, &samples_per_batch_float};
      SliderBounds intensity_slider_bounds = {0, 0, 0, 0, 0.1f, 3.0f, &light_intensity};
      SliderBounds background_slider_bounds = {0, 0, 0, 0, 0.0f, 3.0f, &background_intensity};
      SliderBounds fuzziness_slider_bounds = {0, 0, 0, 0, 0.0f, 5.0f, &metal_fuzziness};
      SliderBounds glass_ior_slider_bounds = {0, 0, 0, 0, 1.0f, 2.5f, &glass_refraction_index};
      SliderBounds dof_aperture_slider_bounds = {0, 0, 0, 0, 0.0f, 1.0f, &dof_aperture};
      SliderBounds dof_focus_slider_bounds = {0, 0, 0, 0, 1.0f, 50.0f, &dof_focus_distance};
      SDL_Rect toggle_button_rect = {0, 0, 0, 0};
      SDL_Rect orbit_button_rect = {0, 0, 0, 0};
      SDL_Rect dof_button_rect = {0, 0, 0, 0};
      bool dragging_slider = false;
      SliderBounds *active_slider = nullptr;

      // Rendering buffers
      SDL_Event event;
      std::vector<unsigned char> display_image(image_width * image_height * image_channels);
      std::vector<float> accum_buffer(image_width * image_height * image_channels, 0.0f);
      RenderTargetView display_view{&display_image, image_width, image_height, image_channels};

      // Initialize OptiX pipeline and build scene (idempotent — safe to call once)
      optixRendererInit();
      optixRendererBuildScene(scene);

      auto last_frame_time = std::chrono::high_resolution_clock::now();
      auto total_start = std::chrono::high_resolution_clock::now();

      // ─── Main loop ───────────────────────────────────────────────────────────
      while (running)
      {
         while (gui.pollEvent(event))
         {
            if (event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE))
            {
               running = false;
            }
            else if (event.type == SDL_KEYDOWN)
            {
               if (event.key.keysym.sym == SDLK_h)
               {
                  gui.toggleControls();
               }
               else if (camera_control.handleKeyDown(event, accumulation_enabled, samples_per_batch_float,
                                                     light_intensity, background_intensity, needs_rerender,
                                                     camera_changed))
               {
                  syncSamplesFromSlider();
                  propagateAccumulationToggle();
               }
            }
            else if (event.type == SDL_MOUSEBUTTONDOWN)
            {
               syncSamplesFromSlider();
               if (camera_control.handleMouseButtonDown(
                       event, dragging_slider, active_slider, samples_slider_bounds, intensity_slider_bounds,
                       background_slider_bounds, fuzziness_slider_bounds, glass_ior_slider_bounds,
                       dof_aperture_slider_bounds, dof_focus_slider_bounds, toggle_button_rect, orbit_button_rect,
                       dof_button_rect, accumulation_enabled, dof_enabled, samples_per_batch_float, light_intensity,
                       background_intensity, metal_fuzziness, glass_refraction_index, dof_aperture, dof_focus_distance,
                       needs_rerender, camera_changed, gui.getShowControls()))
               {
                  syncSamplesFromSlider();
                  propagateAccumulationToggle();
               }
            }
            else if (event.type == SDL_MOUSEBUTTONUP)
            {
               camera_control.handleMouseButtonUp(event, dragging_slider, active_slider);
            }
            else if (event.type == SDL_MOUSEMOTION)
            {
               if (camera_control.handleMouseMotion(
                       event, dragging_slider, active_slider, samples_slider_bounds, intensity_slider_bounds,
                       background_slider_bounds, fuzziness_slider_bounds, glass_ior_slider_bounds,
                       dof_aperture_slider_bounds, dof_focus_slider_bounds, samples_per_batch_float, light_intensity,
                       background_intensity, metal_fuzziness, glass_refraction_index, dof_aperture, dof_focus_distance,
                       needs_rerender, camera_changed, look_from, look_at, vup, basis_w, gui.getShowControls()))
               {
                  syncSamplesFromSlider();
                  camera_changed = true;
               }
            }
            else if (event.type == SDL_MOUSEWHEEL)
            {
               if (camera_control.handleMouseWheel(event, look_from, look_at))
                  camera_changed = true;
            }
         }

         // Auto-orbit
         auto current_frame_time = std::chrono::high_resolution_clock::now();
         std::chrono::duration<float> delta = current_frame_time - last_frame_time;
         last_frame_time = current_frame_time;

         if (camera_control.updateAutoOrbit(look_from, look_at, delta.count()))
            camera_changed = true;

         // Motion detection
         auto now = std::chrono::high_resolution_clock::now();
         std::chrono::duration<float> time_since_last_change = now - last_camera_change_time;
         is_camera_moving = (time_since_last_change.count() < motion_cooldown_seconds);

         // Camera changed → reset accumulation
         if (camera_changed)
         {
            camera_changed = false;
            current_samples = 0;
            force_immediate_render = true;
            std::fill(accum_buffer.begin(), accum_buffer.end(), 0.0f);
            last_camera_change_time = now;
            is_camera_moving = true;

            // Tell OptiX to zero its device accumulation buffer
            optixRendererResetAccum(image_width, image_height);

            refreshCameraFrame();
         }

         // Redisplay after settings change without adding new samples
         if (needs_rerender && current_samples > 0)
         {
            render::convertAccumBufferToImage(display_view, accum_buffer, current_samples, gamma);
            displayFrame(gui, display_image, current_samples, adaptive_samples_per_batch, light_intensity,
                         background_intensity, metal_fuzziness, glass_refraction_index, accumulation_enabled,
                         camera_control.isAutoOrbitEnabled(), dof_enabled, dof_aperture, dof_focus_distance,
                         samples_slider_bounds, intensity_slider_bounds, background_slider_bounds,
                         fuzziness_slider_bounds, glass_ior_slider_bounds, dof_aperture_slider_bounds,
                         dof_focus_slider_bounds, toggle_button_rect, orbit_button_rect, dof_button_rect,
                         image_channels);
            if (target.pixels) *target.pixels = display_image;
            needs_rerender = false;
         }

         bool should_render = (current_samples < max_samples && !camera_changed && running) || force_immediate_render;
         bool needs_initial_render = current_samples == 0 && !accumulation_enabled;

         if (should_render && (accumulation_enabled || needs_initial_render || force_immediate_render))
         {
            force_immediate_render = false;
            syncSamplesFromSlider();
            user_samples_per_batch = samples_per_batch;

            adaptive_samples_per_batch = is_camera_moving ? std::max(5, adaptive_samples_per_batch)
                                                           : user_samples_per_batch;

            auto frame_start = std::chrono::high_resolution_clock::now();

            renderBatch(frame, accum_buffer, display_view, current_samples, max_samples,
                        adaptive_samples_per_batch, gamma, is_camera_moving, adaptive_depth, context,
                        num_materials, background_intensity, dof_enabled, dof_aperture, dof_focus_distance);

            auto frame_end = std::chrono::high_resolution_clock::now();
            (void)(frame_end - frame_start); // timing available for future adaptive tuning

            if (is_camera_moving)
               adaptive_samples_per_batch = 5;

            displayFrame(gui, display_image, current_samples, adaptive_samples_per_batch, light_intensity,
                         background_intensity, metal_fuzziness, glass_refraction_index, accumulation_enabled,
                         camera_control.isAutoOrbitEnabled(), dof_enabled, dof_aperture, dof_focus_distance,
                         samples_slider_bounds, intensity_slider_bounds, background_slider_bounds,
                         fuzziness_slider_bounds, glass_ior_slider_bounds, dof_aperture_slider_bounds,
                         dof_focus_slider_bounds, toggle_button_rect, orbit_button_rect, dof_button_rect,
                         image_channels);

            if (target.pixels) *target.pixels = display_image;
         }
      }

      auto total_end = std::chrono::high_resolution_clock::now();
      std::cout << "\nTotal session time: " << render::timeStr(total_end - total_start) << std::endl;

      optixRendererCleanup();
      gui.cleanup();
   }

 private:
   Settings settings_{};

   int calculateProgressiveMaxDepth(int current_samples, bool is_moving, int max_depth) const
   {
      if (is_moving) return 3;
      if (current_samples <= 4)   return 4;
      if (current_samples <= 16)  return 5;
      if (current_samples <= 32)  return 6;
      if (current_samples <= 64)  return 7;
      if (current_samples <= 128) return 8;
      if (current_samples <= 256) return 16;
      if (current_samples <= 512) return 16;
      if (current_samples <= 1024) return 24;
      return std::min(512, max_depth);
   }

   void renderBatch(const CameraFrame &frame, std::vector<float> &accum_buffer, RenderTargetView display_target,
                    int &current_samples, int max_samples, int samples_per_batch, float gamma,
                    bool is_moving, bool adaptive_depth, RenderContext &context, int num_materials,
                    float background_intensity, bool dof_enabled, float dof_aperture, float dof_focus_distance)
   {
      // If we've already reached or exceeded the maximum, do not render more samples.
      if (current_samples >= max_samples) return;

      const int remaining       = max_samples - current_samples;
      const int actual_samples  = std::min(samples_per_batch, remaining);
      const int new_total_samples = current_samples + actual_samples;

      const int depth = adaptive_depth
                            ? calculateProgressiveMaxDepth(new_total_samples, is_moving, frame.max_depth)
                            : frame.max_depth;

      unsigned long long ray_count = optixRendererLaunch(
          frame.image_width, frame.image_height, num_materials,
          actual_samples, new_total_samples, depth,
          static_cast<float>(frame.camera_center.x()), static_cast<float>(frame.camera_center.y()),
          static_cast<float>(frame.camera_center.z()),
          static_cast<float>(frame.pixel00_loc.x()), static_cast<float>(frame.pixel00_loc.y()),
          static_cast<float>(frame.pixel00_loc.z()),
          static_cast<float>(frame.pixel_delta_u.x()), static_cast<float>(frame.pixel_delta_u.y()),
          static_cast<float>(frame.pixel_delta_u.z()),
          static_cast<float>(frame.pixel_delta_v.x()), static_cast<float>(frame.pixel_delta_v.y()),
          static_cast<float>(frame.pixel_delta_v.z()),
          static_cast<float>(frame.u.x()), static_cast<float>(frame.u.y()), static_cast<float>(frame.u.z()),
          static_cast<float>(frame.v.x()), static_cast<float>(frame.v.y()), static_cast<float>(frame.v.z()),
          background_intensity, dof_enabled, dof_aperture, dof_focus_distance);

      // Download GPU accumulation buffer to host for display
      optixRendererDownloadAccum(accum_buffer.data(), frame.image_width, frame.image_height);

      context.ray_counter.fetch_add(ray_count, std::memory_order_relaxed);

      current_samples = new_total_samples;
      render::convertAccumBufferToImage(display_target, accum_buffer, current_samples, gamma);
   }

   void displayFrame(SDLGuiHandler &gui, const std::vector<unsigned char> &display_image, int current_samples,
                     int samples_per_batch, float light_intensity, float background_intensity, float metal_fuzziness,
                     float glass_refraction_index, bool accumulation_enabled, bool auto_orbit_enabled, bool dof_enabled,
                     float dof_aperture, float dof_focus_distance, SliderBounds &samples_slider_bounds,
                     SliderBounds &intensity_slider_bounds, SliderBounds &background_slider_bounds,
                     SliderBounds &fuzziness_slider_bounds, SliderBounds &glass_ior_slider_bounds,
                     SliderBounds &dof_aperture_slider_bounds, SliderBounds &dof_focus_slider_bounds,
                     SDL_Rect &toggle_button_rect, SDL_Rect &orbit_button_rect, SDL_Rect &dof_button_rect,
                     int image_channels)
   {
      gui.updateDisplay(display_image, image_channels);
      gui.drawLogo();
      gui.drawSampleCountText(current_samples);
      gui.drawUIControls(samples_per_batch, light_intensity, background_intensity, metal_fuzziness,
                         glass_refraction_index, accumulation_enabled, auto_orbit_enabled, samples_slider_bounds,
                         intensity_slider_bounds, background_slider_bounds, fuzziness_slider_bounds,
                         glass_ior_slider_bounds, toggle_button_rect, orbit_button_rect);
      gui.drawEffectsPanel(dof_enabled, dof_aperture, dof_focus_distance, dof_aperture_slider_bounds,
                           dof_focus_slider_bounds, dof_button_rect);
      gui.present();
   }
};

#endif // SDL2_FOUND && OPTIX_FOUND
