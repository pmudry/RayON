/**
 * @class RendererProgressiveSDL
 * @brief Interactive SDL renderer with progressive sample accumulation in CUDA
 *
 * This renderer focuses on ray-tracing logic with progressive quality improvement.
 * GUI and camera control are delegated to separate handler classes.
 */
#pragma once

#ifdef SDL2_FOUND

#include "camera/camera_base.h"
#include "gpu_renderers/renderer_cuda.h"
#include "interval.h"
#include "sdl_gui_controls.h"
#include "sdl_gui_handler.h"

#include "../scene_builder.h"
#include "../scene_description.h"
#include <SDL.h>
#include <algorithm>
#include <chrono>

// Forward declarations of CUDA functions
extern "C"
{
   void setLightIntensity(float intensity);
   void setBackgroundIntensity(float intensity);
   void setMetalFuzziness(float fuzziness);
   void setDOFEnabled(bool enabled);
   void setDOFAperture(float aperture);
   void setDOFFocusDistance(float distance);
   unsigned long long renderPixelsCUDAAccumulative(unsigned char *image, float *accum_buffer, CudaScene::Scene *scene,
                                                   int width, int height, double cam_center_x, double cam_center_y,
                                                   double cam_center_z, double pixel00_x, double pixel00_y,
                                                   double pixel00_z, double delta_u_x, double delta_u_y,
                                                   double delta_u_z, double delta_v_x, double delta_v_y,
                                                   double delta_v_z, int samples_to_add, int total_samples_so_far,
                                                   int max_depth, void **d_rand_states_ptr, void **d_accum_buffer_ptr,
                                                   double cam_u_x, double cam_u_y, double cam_u_z,
                                                   double cam_v_x, double cam_v_y, double cam_v_z);
   void freeDeviceRandomStates(void *d_rand_states);
   void freeDeviceAccumBuffer(void *d_accum_buffer);
}

class RendererCUDAProgressive : virtual public CameraBase
{
 public:
   using CameraBase::CameraBase;

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
   void renderPixelsSDLContinuous(vector<unsigned char> &image, int samples_per_batch = 8, bool auto_accumulate = true, int target_fps = 60, bool adaptive_depth = false)
   {
      // Initialize GUI
      SDLGuiHandler gui(image_width, image_height);
      if (!gui.initialize())
         return;

      int max_samples = samples_per_pixel;

      gui.printControls(samples_per_batch, max_samples, auto_accumulate);

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
      float glass_refraction_index = 1.5f;  // Default glass index
      bool dof_enabled = false;
      float dof_aperture = 0.1f;
      float dof_focus_distance = 10.0f;
      bool needs_rerender = false;
      bool force_immediate_render = false; // Flag to force rendering immediately after state change
      float samples_per_batch_float = static_cast<float>(samples_per_batch); // Float version for slider
      
      // Motion detection for adaptive quality
      bool is_camera_moving = false;
      auto last_camera_change_time = std::chrono::high_resolution_clock::now();
      const float motion_cooldown_seconds = 0.5f; // Wait 0.5s after last input before considering stopped
      
      // Adaptive sample rate for smooth target FPS
      int adaptive_samples_per_batch = samples_per_batch; // Actual samples to render (adapts during motion)
      int user_samples_per_batch = samples_per_batch;     // User's preferred samples (from UI slider)
      const float target_frame_time_ms = 1000.0f / target_fps; // Calculate target frame time from FPS
      const float adaptive_speed = 0.2f; // How quickly to adapt sample rate (lower = smoother)

      // Set initial rendering parameters
      ::setLightIntensity(light_intensity);
      ::setBackgroundIntensity(background_intensity);
      ::setMetalFuzziness(metal_fuzziness);
      ::setGlassRefractionIndex(glass_refraction_index);
      ::setDOFEnabled(dof_enabled);
      ::setDOFAperture(dof_aperture);
      ::setDOFFocusDistance(dof_focus_distance);

      // UI state
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
      vector<unsigned char> display_image(image_width * image_height * image_channels);
      vector<float> accum_buffer(image_width * image_height * image_channels, 0.0f);
      void *d_rand_states = nullptr;
      void *d_accum_buffer = nullptr; // Persistent device accumulation buffer

      // Build scene once
      Scene::SceneDescription scene_desc = createDefaultScene();
      CudaScene::Scene *gpu_scene = Scene::CudaSceneBuilder::buildGPUScene(scene_desc);

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
            else if (event.type == SDL_KEYDOWN)
            {
               // Handle 'h' key to toggle GUI controls visibility
               if (event.key.keysym.sym == SDLK_h)
               {
                  gui.toggleControls();
               }
               else if (camera_control.handleKeyDown(event, accumulation_enabled, samples_per_batch_float,
                                                     light_intensity, background_intensity, needs_rerender,
                                                     camera_changed))
               {
                  // Sync samples_per_batch from float slider value
                  samples_per_batch = static_cast<int>(samples_per_batch_float);

                  if (camera_changed)
                  {
                     ::setLightIntensity(light_intensity);
                     ::setBackgroundIntensity(background_intensity);
                  }
                  if (accumulation_enabled != auto_accumulate)
                  {
                     auto_accumulate = accumulation_enabled;
                  }
               }
            }
            else if (event.type == SDL_MOUSEBUTTONDOWN)
            {
               // Sync samples_per_batch from float slider value
               samples_per_batch = static_cast<int>(samples_per_batch_float);

               if (camera_control.handleMouseButtonDown(
                       event, dragging_slider, active_slider, samples_slider_bounds, intensity_slider_bounds,
                       background_slider_bounds, fuzziness_slider_bounds, glass_ior_slider_bounds,
                       dof_aperture_slider_bounds, dof_focus_slider_bounds, toggle_button_rect, orbit_button_rect, dof_button_rect,
                       accumulation_enabled, dof_enabled, samples_per_batch_float, light_intensity, background_intensity,
                       metal_fuzziness, glass_refraction_index, dof_aperture, dof_focus_distance, needs_rerender, camera_changed, gui.getShowControls()))
               {
                  // Sync samples_per_batch from float slider value after modification
                  samples_per_batch = static_cast<int>(samples_per_batch_float);

                  if (camera_changed)
                  {
                     ::setLightIntensity(light_intensity);
                     ::setBackgroundIntensity(background_intensity);
                     ::setMetalFuzziness(metal_fuzziness);
                     ::setGlassRefractionIndex(glass_refraction_index);
                     ::setDOFEnabled(dof_enabled);
                     ::setDOFAperture(dof_aperture);
                     ::setDOFFocusDistance(dof_focus_distance);
                  }
                  if (accumulation_enabled != auto_accumulate)
                  {
                     auto_accumulate = accumulation_enabled;
                  }
               }
            }
            else if (event.type == SDL_MOUSEBUTTONUP)
            {
               camera_control.handleMouseButtonUp(event, dragging_slider, active_slider);
            }
            else if (event.type == SDL_MOUSEMOTION)
            {
               if (camera_control.handleMouseMotion(event, dragging_slider, active_slider, samples_slider_bounds,
                                                    intensity_slider_bounds, background_slider_bounds,
                                                    fuzziness_slider_bounds, glass_ior_slider_bounds,
                                                    dof_aperture_slider_bounds, dof_focus_slider_bounds, samples_per_batch_float, light_intensity,
                                                    background_intensity, metal_fuzziness, glass_refraction_index, dof_aperture,
                                                    dof_focus_distance, needs_rerender,
                                                    camera_changed, look_from, look_at, vup, w, gui.getShowControls()))
               {
                  // Sync samples_per_batch from float slider value
                  samples_per_batch = static_cast<int>(samples_per_batch_float);

                  if (camera_changed)
                  {
                     ::setLightIntensity(light_intensity);
                     ::setBackgroundIntensity(background_intensity);
                     ::setMetalFuzziness(metal_fuzziness);
                     ::setGlassRefractionIndex(glass_refraction_index);
                     ::setDOFEnabled(dof_enabled);
                     ::setDOFAperture(dof_aperture);
                     ::setDOFFocusDistance(dof_focus_distance);
                  }
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

            // Free and reset device buffers on camera change
            if (d_rand_states != nullptr)
            {
               freeDeviceRandomStates(d_rand_states);
               d_rand_states = nullptr;
            }
            if (d_accum_buffer != nullptr)
            {
               freeDeviceAccumBuffer(d_accum_buffer);
               d_accum_buffer = nullptr;
            }

            initialize(); // Recalculate camera parameters
         }

         // Reprocess with new gamma if needed (without re-rendering)
         if (needs_rerender && current_samples > 0)
         {
            applyGammaCorrection(display_image, accum_buffer, current_samples, gamma);
            displayFrame(gui, display_image, current_samples, adaptive_samples_per_batch, light_intensity, background_intensity,
                         metal_fuzziness, glass_refraction_index, accumulation_enabled, camera_control.isAutoOrbitEnabled(),
                         dof_enabled, dof_aperture, dof_focus_distance,
                         samples_slider_bounds, intensity_slider_bounds, background_slider_bounds,
                         fuzziness_slider_bounds, glass_ior_slider_bounds, dof_aperture_slider_bounds, dof_focus_slider_bounds,
                         toggle_button_rect, orbit_button_rect, dof_button_rect);
            image = display_image;
            needs_rerender = false;
         }

         // Render logic: accumulate if enabled, or render once if auto-accumulation is off
         bool should_render = (current_samples < max_samples && !camera_changed && running) || force_immediate_render;
         bool needs_initial_render =
             current_samples == 0 && !accumulation_enabled; // Render at least once when auto-accumulation is off

         if (should_render && (accumulation_enabled || needs_initial_render || force_immediate_render))
         {
            force_immediate_render = false; // Reset flag after rendering
            
            // Sync user preference from slider
            user_samples_per_batch = static_cast<int>(samples_per_batch_float);
            
            // Adaptive sample rate: use fewer samples during motion for smooth 60 FPS
            if (is_camera_moving)
            {
               // During motion, use adaptive sample count for target frame rate
               adaptive_samples_per_batch = max(1, adaptive_samples_per_batch);
            }
            else
            {
               // When camera stops, gradually ramp up to user's preferred sample count
               adaptive_samples_per_batch = user_samples_per_batch;
            }
            
            // Start timing this frame
            auto frame_start = std::chrono::high_resolution_clock::now();
            
            renderBatch(display_image, accum_buffer, current_samples, max_samples, adaptive_samples_per_batch, gamma,
                        d_rand_states, d_accum_buffer, gpu_scene, is_camera_moving, adaptive_depth);
            
            // Measure frame time and adapt sample rate for next frame
            auto frame_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> frame_time = frame_end - frame_start;
            
            // Adaptive adjustment only during motion
            if (is_camera_moving)
            {
               float time_ratio = target_frame_time_ms / frame_time.count();
               
               // Adjust sample count to hit target FPS
               // Use exponential smoothing for gradual adjustment
               float target_samples = adaptive_samples_per_batch * time_ratio;
               adaptive_samples_per_batch = max(1, static_cast<int>(
                  adaptive_samples_per_batch * (1.0f - adaptive_speed) + target_samples * adaptive_speed
               ));
               
               // Clamp to reasonable range: 1 to user preference
               adaptive_samples_per_batch = max(1, min(adaptive_samples_per_batch, user_samples_per_batch));
            }

            displayFrame(gui, display_image, current_samples, adaptive_samples_per_batch, light_intensity, background_intensity,
                         metal_fuzziness, glass_refraction_index, accumulation_enabled, camera_control.isAutoOrbitEnabled(),
                         dof_enabled, dof_aperture, dof_focus_distance,
                         samples_slider_bounds, intensity_slider_bounds,
                         background_slider_bounds, fuzziness_slider_bounds, glass_ior_slider_bounds, dof_aperture_slider_bounds,
                         dof_focus_slider_bounds, toggle_button_rect, orbit_button_rect, dof_button_rect);

            image = display_image;
         }
         else if (current_samples >= max_samples && !camera_changed)
         {
            // Refresh display even when idle to show logo and UI
            displayFrame(gui, display_image, current_samples, adaptive_samples_per_batch, light_intensity, background_intensity,
                         metal_fuzziness, glass_refraction_index, accumulation_enabled, camera_control.isAutoOrbitEnabled(),
                         dof_enabled, dof_aperture, dof_focus_distance,
                         samples_slider_bounds, intensity_slider_bounds,
                         background_slider_bounds, fuzziness_slider_bounds, glass_ior_slider_bounds, dof_aperture_slider_bounds,
                         dof_focus_slider_bounds, toggle_button_rect, orbit_button_rect, dof_button_rect);
            SDL_Delay(8); // ~60 FPS event polling
         }
         else if (!accumulation_enabled && current_samples > 0 && !camera_changed)
         {
            // Refresh display even when idle to show logo and UI
            displayFrame(gui, display_image, current_samples, adaptive_samples_per_batch, light_intensity, background_intensity,
                         metal_fuzziness, glass_refraction_index, accumulation_enabled, camera_control.isAutoOrbitEnabled(),
                         dof_enabled, dof_aperture, dof_focus_distance,
                         samples_slider_bounds, intensity_slider_bounds,
                         background_slider_bounds, fuzziness_slider_bounds, glass_ior_slider_bounds, dof_aperture_slider_bounds,
                         dof_focus_slider_bounds, toggle_button_rect, orbit_button_rect, dof_button_rect);
            SDL_Delay(8); // ~60 FPS event polling (already rendered once, waiting for user input)
         }
      }

      auto total_end = std::chrono::high_resolution_clock::now();
      cout << "\nTotal session time: " << timeStr(total_end - total_start) << endl;

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
   /**
    * @brief Calculate progressive max depth based on accumulated samples
    * Starts at depth 1, gradually increases to max 256 for final quality
    */
   int calculateProgressiveMaxDepth(int current_samples, int max_samples, bool is_moving)
   {
      // During camera motion, use reduced depth for faster preview
      if (is_moving)
      {
         return 3; // Fast preview during motion
      }
      
      // Progressive depth schedule for smooth quality ramp-up
      if (current_samples <= 4)
         return 4;  // First few samples: depth 1 (fastest preview)
      else if (current_samples <= 16)
         return 5;  // Quick preview: depth 2
      else if (current_samples <= 32)
         return 6;  // Early quality: depth 3
      else if (current_samples <= 64)
         return 7;  // Building detail: depth 4
      else if (current_samples <= 128)
         return 8;  // Good quality: depth 6
      else if (current_samples <= 256)
         return 16;  // High quality: depth 8
      else if (current_samples <= 512)
         return 16; // Very high quality: depth 12
      else if (current_samples <= 1024)
         return 24; // Excellent quality: depth 16      
      else
         return min(512, max_depth); // Final quality: depth up to 256
   }

   /**
    * @brief Render a batch of samples using CUDA
    */
   void renderBatch(vector<unsigned char> &display_image, vector<float> &accum_buffer, int &current_samples,
                    int max_samples, int samples_per_batch, float gamma, void *&d_rand_states, void *&d_accum_buffer,
                    CudaScene::Scene *gpu_scene, bool is_moving, bool adaptive_depth = false)
   {
      current_samples += samples_per_batch;
      if (current_samples > max_samples)
         current_samples = max_samples;

      int actual_samples_to_add = samples_per_batch;
      if (current_samples > max_samples)
         actual_samples_to_add = max_samples - (current_samples - samples_per_batch);

      // Calculate progressive max depth based on current sample count and motion state
      // If adaptive_depth is disabled, use the full max_depth
      int progressive_depth = adaptive_depth ? calculateProgressiveMaxDepth(current_samples, max_samples, is_moving) : max_depth;

      // Call CUDA to render and accumulate samples with progressive depth
      unsigned long long cuda_ray_count = ::renderPixelsCUDAAccumulative(
          display_image.data(), accum_buffer.data(), gpu_scene, image_width, image_height, camera_center.x(),
          camera_center.y(), camera_center.z(), pixel00_loc.x(), pixel00_loc.y(), pixel00_loc.z(), pixel_delta_u.x(),
          pixel_delta_u.y(), pixel_delta_u.z(), pixel_delta_v.x(), pixel_delta_v.y(), pixel_delta_v.z(),
          actual_samples_to_add, current_samples, progressive_depth, &d_rand_states, &d_accum_buffer,
          u.x(), u.y(), u.z(), v.x(), v.y(), v.z());

      n_rays.fetch_add(cuda_ray_count, std::memory_order_relaxed);

      // Apply gamma correction to display image
      applyGammaCorrection(display_image, accum_buffer, current_samples, gamma);
   }

   /**
    * @brief Apply gamma correction to accumulated samples
    */
   void applyGammaCorrection(vector<unsigned char> &display_image, const vector<float> &accum_buffer,
                             int current_samples, float gamma)
   {
      static const Interval intensity_range(0.0, 0.999);

      for (int j = 0; j < image_height; j++)
      {
         for (int i = 0; i < image_width; i++)
         {
            int pixel_idx = j * image_width + i;
            int display_idx = pixel_idx * image_channels;
            int accum_idx = pixel_idx * 3;

            float r = accum_buffer[accum_idx + 0] / current_samples;
            float g = accum_buffer[accum_idx + 1] / current_samples;
            float b = accum_buffer[accum_idx + 2] / current_samples;

            r = pow(r, 1.0f / gamma);
            g = pow(g, 1.0f / gamma);
            b = pow(b, 1.0f / gamma);

            display_image[display_idx + 0] = static_cast<unsigned char>(256 * intensity_range.clamp(r));
            display_image[display_idx + 1] = static_cast<unsigned char>(256 * intensity_range.clamp(g));
            display_image[display_idx + 2] = static_cast<unsigned char>(256 * intensity_range.clamp(b));
            if (image_channels == 4)
               display_image[display_idx + 3] = 255;
         }
      }
   }

   /**
    * @brief Update the display with current frame and UI
    */
   void displayFrame(SDLGuiHandler &gui, const vector<unsigned char> &display_image, int current_samples,
                     int samples_per_batch, float light_intensity, float background_intensity, float metal_fuzziness,
                     float glass_refraction_index,
                     bool accumulation_enabled, bool auto_orbit_enabled, bool dof_enabled,
                     float dof_aperture, float dof_focus_distance,
                     SliderBounds &samples_slider_bounds, SliderBounds &intensity_slider_bounds,
                     SliderBounds &background_slider_bounds, SliderBounds &fuzziness_slider_bounds,
                     SliderBounds &glass_ior_slider_bounds,
                     SliderBounds &dof_aperture_slider_bounds, SliderBounds &dof_focus_slider_bounds,
                     SDL_Rect &toggle_button_rect, SDL_Rect &orbit_button_rect, SDL_Rect &dof_button_rect)
   {
      gui.updateDisplay(display_image, image_channels);
      gui.drawLogo();
      gui.drawSampleCountText(current_samples);
      gui.drawUIControls(samples_per_batch, light_intensity, background_intensity, metal_fuzziness,
                         glass_refraction_index, accumulation_enabled, auto_orbit_enabled, samples_slider_bounds,
                         intensity_slider_bounds, background_slider_bounds, fuzziness_slider_bounds,
                         glass_ior_slider_bounds, toggle_button_rect, orbit_button_rect);
      gui.drawEffectsPanel(dof_enabled, dof_aperture, dof_focus_distance,
                          dof_aperture_slider_bounds, dof_focus_slider_bounds, dof_button_rect);
      gui.present();
   }

   // Forward declaration - scene is created in main.cc
   Scene::SceneDescription createDefaultScene(); // Implemented in main.cc
};

#endif // SDL2_FOUND
