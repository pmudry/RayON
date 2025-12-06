/**
 * @class RendererProgressiveSDL
 * @brief Interactive SDL renderer with progressive sample accumulation in CUDA
 *
 * This renderer focuses on ray-tracing logic with progressive quality improvement.
 * GUI and camera control are delegated to separate handler classes.
 */
#pragma once

#ifdef SDL2_FOUND

#include "render/render_utils.hpp"
#include "render/renderer_interface.hpp"
#include "renderer_cuda_host.hpp"
#include "scene_builder.hpp"
#include "sdl_gui_controls.hpp"
#include "sdl_gui_handler.hpp"

#include <SDL.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

class RendererCUDAProgressive : public IRenderer
{
 public:
   struct Settings
   {
      int samples_per_batch = 8;
      bool auto_accumulate = true;
      int target_fps = 60;
      bool adaptive_depth = false;
   };

   RendererCUDAProgressive() = default;
   explicit RendererCUDAProgressive(Settings settings) : settings_(settings) {}

   void setSettings(const Settings &settings) { settings_ = settings; }

   /**
    * @brief Interactive SDL rendering with continuous sample accumulation
    *
    * This method implements the core ray-tracing loop with progressive sampling.
    * GUI and camera controls are handled by separate dedicated classes.
    *
    * @param image The final image buffer to store the render
    * @param max_samples Maximum total samples to accumulate (default: 4096)
    * @param samples_per_batch Number of samples to add per batch (default: 8)
    * @param auto_accumulate Enable automatic sample accumulation (default: true)
    * @param target_fps Target frame rate for interactive rendering (default: 60)
    * @param adaptive_depth Enable adaptive depth (progressively increases max depth) (default: false)
    */
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

      // Initialize camera controls
      CameraControlHandler camera_control;
      camera_control.initializeCameraControls(look_from, look_at);

      // Ray-tracing state
      bool running = true;
      bool camera_changed = true;
      bool accumulation_enabled = auto_accumulate;
      int current_samples = 0;
      float gamma = 2.0f; // Fixed gamma value
      float light_intensity = 1.0f;
      float background_intensity = 1.0f;
      float metal_fuzziness = 1.0f;
      float glass_refraction_index = 1.5f; // Default glass index
      bool dof_enabled = false;
      float dof_aperture = 0.1f;
      float dof_focus_distance = 10.0f;
      float fov = static_cast<float>(camera.vfov);
      float initial_fov = fov;
      bool needs_rerender = false;
      bool force_immediate_render = false; // Flag to force rendering immediately after state change
      float samples_per_batch_float = static_cast<float>(samples_per_batch); // Float version for slider
      float current_sps = 0.0f;           // Track Samples Per Second for UI
      float current_ms_per_sample = 0.0f; // Track Time Per Sample for UI

      // Motion detection for adaptive quality
      bool is_camera_moving = false;
      auto last_camera_change_time = std::chrono::high_resolution_clock::now();
      const float motion_cooldown_seconds = 0.5f; // Wait 0.5s after last input before considering stopped

      // Adaptive sample rate for smooth target FPS
      int adaptive_samples_per_batch = samples_per_batch;      // Actual samples to render (adapts during motion)
      int user_samples_per_batch = samples_per_batch;          // User's preferred samples (from UI slider)
      const float target_frame_time_ms = 1000.0f / target_fps; // Calculate target frame time from FPS
      const float adaptive_speed = 0.2f;                       // How quickly to adapt sample rate (lower = smoother)

      auto syncSamplesFromSlider = [&]()
      { samples_per_batch = std::max(1, static_cast<int>(samples_per_batch_float)); };

      auto applySceneSettings = [&]()
      {
         ::setLightIntensity(light_intensity);
         ::setBackgroundIntensity(background_intensity);
         ::setMetalFuzziness(metal_fuzziness);
         ::setGlassRefractionIndex(glass_refraction_index);
         ::setDOFEnabled(dof_enabled);
         ::setDOFAperture(dof_aperture);
         ::setDOFFocusDistance(dof_focus_distance);
      };

      auto propagateAccumulationToggle = [&]()
      {
         if (accumulation_enabled != auto_accumulate)
            auto_accumulate = accumulation_enabled;
      };

      applySceneSettings();

      // Rendering buffers
      SDL_Event event;
      vector<unsigned char> display_image(image_width * image_height * image_channels);
      vector<float> accum_buffer(image_width * image_height * image_channels, 0.0f);
      RenderTargetView display_view{&display_image, image_width, image_height, image_channels};

      void *d_rand_states = nullptr;
      void *d_accum_buffer = nullptr; // Persistent device accumulation buffer

      // Build scene once
      CudaScene::Scene *gpu_scene = Scene::CudaSceneBuilder::buildGPUScene(scene);

      // Timing for auto-orbit
      auto last_frame_time = std::chrono::high_resolution_clock::now();

      auto total_start = std::chrono::high_resolution_clock::now();

      // Main rendering loop
      while (running)
      {
         // Handle events
         while (gui.pollEvent(event))
         {
            if (event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE))
            {
               running = false;
            }

            // Prevent camera/scene interaction if ImGui is using inputs
            ImGuiIO& io = ImGui::GetIO();
            if (io.WantCaptureMouse)
            {
               if (event.type == SDL_MOUSEBUTTONDOWN || event.type == SDL_MOUSEBUTTONUP ||
                   event.type == SDL_MOUSEMOTION || event.type == SDL_MOUSEWHEEL)
                  continue;
            }
            if (io.WantCaptureKeyboard)
            {
               if (event.type == SDL_KEYDOWN || event.type == SDL_KEYUP)
                  continue;
            }

            if (event.type == SDL_KEYDOWN)
            {
               // Handle 'h' key to toggle GUI controls visibility
               if (event.key.keysym.sym == SDLK_h)
               {
                  gui.toggleControls();
               }
               else if (event.key.keysym.sym == SDLK_c)
               {
                  gui.toggleHeaderCollapse();
               }
               else if (event.key.keysym.sym == SDLK_r)
               {
                  // Reset to default state
                  light_intensity = 1.0f;
                  background_intensity = 1.0f;
                  metal_fuzziness = 1.0f;
                  glass_refraction_index = 1.5f;
                  dof_enabled = false;
                  dof_aperture = 0.1f;
                  dof_focus_distance = 10.0f;
                  fov = initial_fov;
                  samples_per_batch_float = static_cast<float>(settings_.samples_per_batch);
                  camera_control.setAutoOrbit(false);

                  camera.vfov = fov;
                  refreshCameraFrame();

                  camera_changed = true;
                  applySceneSettings();
               }
               else if (camera_control.handleKeyDown(event, accumulation_enabled, samples_per_batch_float,
                                                     light_intensity, background_intensity, needs_rerender,
                                                     camera_changed))
               {
                  syncSamplesFromSlider();
                  if (camera_changed)
                     applySceneSettings();
                  propagateAccumulationToggle();
               }
            }
            else if (event.type == SDL_MOUSEBUTTONDOWN)
            {
               syncSamplesFromSlider();

               if (camera_control.handleMouseButtonDown(event))
               {
                  syncSamplesFromSlider();

                  if (camera_changed)
                     applySceneSettings();
                  propagateAccumulationToggle();
               }
            }
            else if (event.type == SDL_MOUSEBUTTONUP)
            {
               camera_control.handleMouseButtonUp(event);
            }
            else if (event.type == SDL_MOUSEMOTION)
            {
               if (camera_control.handleMouseMotion(
                       event, look_from, look_at, vup, basis_w))
               {
                  syncSamplesFromSlider();

                  if (camera_changed)
                     applySceneSettings();
                  camera_changed = true;
               }
            }
            else if (event.type == SDL_MOUSEWHEEL)
            {
               if (camera_control.handleMouseWheel(event, look_from, look_at))
               {
                  camera_changed = true;
               }
            }
         }

         // Update auto-orbit if enabled (before handling camera_changed)
         auto current_frame_time = std::chrono::high_resolution_clock::now();
         std::chrono::duration<float> delta = current_frame_time - last_frame_time;
         last_frame_time = current_frame_time;

         if (camera_control.updateAutoOrbit(look_from, look_at, delta.count()))
         {
            camera_changed = true;
         }

         // Update motion detection
         auto now = std::chrono::high_resolution_clock::now();
         std::chrono::duration<float> time_since_last_change = now - last_camera_change_time;
         is_camera_moving = (time_since_last_change.count() < motion_cooldown_seconds);

         // Handle camera changes - restart rendering
         if (camera_changed)
         {
            camera_changed = false;
            current_samples = 0;
            force_immediate_render = true; // Force rendering after camera/settings change
            std::fill(accum_buffer.begin(), accum_buffer.end(), 0.0f);

            // Mark camera as moving
            last_camera_change_time = now;
            is_camera_moving = true;

            // Reset only the accumulation buffer; keep RNG states alive to preserve jitter sequences
            if (d_accum_buffer != nullptr)
            {
               freeDeviceAccumBuffer(d_accum_buffer);
               d_accum_buffer = nullptr;
            }

            refreshCameraFrame();
         }

         // Only happens once at start or after toggling accumulation
         if (needs_rerender && current_samples > 0)
         {
            render::convertAccumBufferToImage(display_view, accum_buffer, current_samples, gamma);
            
            // Note: displayFrame is now called unconditionally below
            if (target.pixels)
               *target.pixels = display_image;
            needs_rerender = false;
         }

         // Render logic: accumulate if enabled, or render once if auto-accumulation is off
         bool should_render = (current_samples < max_samples && !camera_changed && running) || force_immediate_render;
         bool needs_initial_render =
             current_samples == 0 && !accumulation_enabled; // Render at least once when auto-accumulation is off

         if (should_render && (accumulation_enabled || needs_initial_render || force_immediate_render))
         {
            force_immediate_render = false; // Reset flag after rendering

            syncSamplesFromSlider();
            user_samples_per_batch = samples_per_batch;

            // Adaptive sample rate: use fewer samples during motion for smooth 60 FPS
            if (is_camera_moving)
            {
               adaptive_samples_per_batch = std::max(5, adaptive_samples_per_batch);
            }
            else
            {
               // When camera stops, gradually ramp up to user's preferred sample count
               adaptive_samples_per_batch = user_samples_per_batch;
            }

            // Start timing this frame
            auto frame_start = std::chrono::high_resolution_clock::now();

            renderBatch(frame, accum_buffer, display_view, current_samples, max_samples, adaptive_samples_per_batch,
                        gamma, d_rand_states, d_accum_buffer, gpu_scene, is_camera_moving, adaptive_depth, context);

            // Measure frame time and adapt sample rate for next frame
            auto frame_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> frame_time = frame_end - frame_start;
            
            if (frame_time.count() > 0.0f)
            {
               current_sps = (static_cast<float>(adaptive_samples_per_batch) * 1000.0f) / frame_time.count();
               current_ms_per_sample = frame_time.count() / static_cast<float>(adaptive_samples_per_batch);
            }

            // Adaptive adjustment only during motion
            if (is_camera_moving)
            {
               adaptive_samples_per_batch = 5; // Fixed low sample count during motion for simplicity
            }

            if (target.pixels)
               *target.pixels = display_image;
         }
         else
         {
            // Cap CPU usage when not rendering
             SDL_Delay(16); 
         }

         // Always display the frame and UI, regardless of whether we rendered a new batch
         bool old_dof = dof_enabled;
         float old_aperture = dof_aperture;
         float old_focus = dof_focus_distance;
         float old_fov = fov;
         float old_light = light_intensity;
         float old_fuzz = metal_fuzziness;
         float old_ior = glass_refraction_index;

         displayFrame(
             gui, display_image, current_samples, &samples_per_batch_float, &light_intensity, background_intensity,
             &metal_fuzziness, &glass_refraction_index, &accumulation_enabled, camera_control, &dof_enabled,
             &dof_aperture, &dof_focus_distance, &fov, 
             image_channels, current_sps, current_ms_per_sample);

         // Check for changes from UI
         if (dof_enabled != old_dof || dof_aperture != old_aperture || dof_focus_distance != old_focus || fov != old_fov ||
             light_intensity != old_light || metal_fuzziness != old_fuzz || glass_refraction_index != old_ior)
         {
            camera_changed = true;
            applySceneSettings();
            if (fov != old_fov)
            {
               camera.vfov = fov;
               refreshCameraFrame();
            }
         }
      }

      auto total_end = std::chrono::high_resolution_clock::now();
      std::cout << "\nTotal session time: " << render::timeStr(total_end - total_start) << std::endl;

      // Cleanup device resources
      if (d_rand_states != nullptr)
      {
         freeDeviceRandomStates(d_rand_states);
      }
      if (d_accum_buffer != nullptr)
      {
         freeDeviceAccumBuffer(d_accum_buffer);
      }

      // Cleanup scene
      Scene::CudaSceneBuilder::freeGPUScene(gpu_scene);

      gui.cleanup();
   }

 private:
   Settings settings_{};

   // Lightweight view describing which buffer SDL should read from after gamma correction.
   struct DisplayBufferView
   {
      const vector<unsigned char> *buffer;
      int channels;
   };

   /**
    * @brief Calculate progressive max depth based on accumulated samples
    * Starts at depth 1, gradually increases to max 256 for final quality
    */
   int calculateProgressiveMaxDepth(int current_samples, bool is_moving, int max_depth) const
   {
      // During camera motion, use reduced depth for faster preview
      if (is_moving)
         return 3; // Fast preview during motion

      // Progressive depth schedule for smooth quality ramp-up
      if (current_samples <= 4) {
         return 4; // First few samples: depth 1 (fastest preview)
      } else if (current_samples <= 16) {
         return 5; // Quick preview: depth 2
      } else if (current_samples <= 32) {
         return 6; // Early quality: depth 3
      } else if (current_samples <= 64) {
         return 7; // Building detail: depth 4
      } else if (current_samples <= 128) {
         return 8; // Good quality: depth 6
      } else if (current_samples <= 256) {
         return 16; // High quality: depth 8
      } else if (current_samples <= 512) {
         return 16; // Very high quality: depth 12
      } else if (current_samples <= 1024) {
         return 24; // Excellent quality: depth 16
      } else {
         return std::min(512, max_depth); // Final quality: depth up to 256
      }
   }

   /**
    * @brief Render a batch of samples using CUDA
    */
   void renderBatch(const CameraFrame &frame, vector<float> &accum_buffer, RenderTargetView display_target,
                    int &current_samples, int max_samples, int samples_per_batch, float gamma, void *&d_rand_states,
                    void *&d_accum_buffer, CudaScene::Scene *gpu_scene, bool is_moving, bool adaptive_depth,
                    RenderContext &context)
   {
      current_samples += samples_per_batch;

      if (current_samples > max_samples)
         current_samples = max_samples;

      int actual_samples_to_add = samples_per_batch;
      if (current_samples > max_samples)
         actual_samples_to_add = max_samples - (current_samples - samples_per_batch);

      const int progressive_depth =
          adaptive_depth ? calculateProgressiveMaxDepth(current_samples, is_moving, frame.max_depth)
                         : frame.max_depth;

      // Call CUDA to render and accumulate samples with progressive depth
      // Note: First parameter (image) is unused by the kernel - it only updates accum_buffer
      unsigned long long cuda_ray_count = ::renderPixelsCUDAAccumulative(
          nullptr, accum_buffer.data(), gpu_scene, frame.image_width, frame.image_height, frame.camera_center.x(),
          frame.camera_center.y(), frame.camera_center.z(), frame.pixel00_loc.x(), frame.pixel00_loc.y(),
          frame.pixel00_loc.z(), frame.pixel_delta_u.x(), frame.pixel_delta_u.y(), frame.pixel_delta_u.z(),
          frame.pixel_delta_v.x(), frame.pixel_delta_v.y(), frame.pixel_delta_v.z(), actual_samples_to_add,
          current_samples, progressive_depth, &d_rand_states, &d_accum_buffer, frame.u.x(), frame.u.y(), frame.u.z(),
          frame.v.x(), frame.v.y(), frame.v.z());

      context.ray_counter.fetch_add(cuda_ray_count, std::memory_order_relaxed);

      render::convertAccumBufferToImage(display_target, accum_buffer, current_samples,
                                        gamma); // Keep display + disk paths identical.
   }

   /**
    * @brief Update the display with current frame and UI
    */
   void displayFrame(SDLGuiHandler &gui, const vector<unsigned char> &display_image, int current_samples,
                     float* samples_per_batch, float* light_intensity, float background_intensity, float* metal_fuzziness,
                     float* glass_refraction_index, bool* accumulation_enabled, CameraControlHandler& camera_control, bool* dof_enabled,
                     float* dof_aperture, float* dof_focus_distance, float* fov,
                     int image_channels, float sps, float ms_per_sample)
   {
      bool is_orbiting = camera_control.isAutoOrbitEnabled();

      gui.updateDisplay(display_image, image_channels, sps, ms_per_sample, current_samples,
                        dof_enabled, dof_aperture, dof_focus_distance, fov,
                        light_intensity, metal_fuzziness, glass_refraction_index,
                        samples_per_batch, accumulation_enabled, &is_orbiting);
      
      if (is_orbiting != camera_control.isAutoOrbitEnabled())
      {
         camera_control.setAutoOrbit(is_orbiting);
      }

      gui.drawLogo();
      gui.present();
   }
};

#endif // SDL2_FOUND
