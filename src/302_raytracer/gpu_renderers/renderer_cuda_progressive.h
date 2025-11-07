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

#include <SDL.h>
#include <algorithm>
#include <chrono>
#include "../scene_description.h"
#include "../scene_builder.h"

// Forward declarations of CUDA functions
extern "C"
{
   void setLightIntensity(float intensity);
   void setBackgroundIntensity(float intensity);
   void setMetalFuzziness(float fuzziness);
   unsigned long long renderPixelsCUDAAccumulative(unsigned char *image, float *accum_buffer, CudaScene::Scene* scene, int width, int height,
                                                  double cam_center_x, double cam_center_y, double cam_center_z,
                                                  double pixel00_x, double pixel00_y, double pixel00_z,
                                                  double delta_u_x, double delta_u_y, double delta_u_z,
                                                  double delta_v_x, double delta_v_y, double delta_v_z,
                                                  int samples_to_add, int total_samples_so_far, int max_depth,
                                                  void **d_rand_states_ptr, void **d_accum_buffer_ptr);
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
    */
   void renderPixelsSDLContinuous(vector<unsigned char> &image, int samples_per_batch = 8, bool auto_accumulate = true)
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
      bool use_stratified_sampling = false; // Start with uniform sampling
      bool needs_rerender = false;
      bool force_immediate_render = false; // Flag to force rendering immediately after state change
      float samples_per_batch_float = static_cast<float>(samples_per_batch); // Float version for slider

      // Set initial rendering parameters
      ::setLightIntensity(light_intensity);
      ::setBackgroundIntensity(background_intensity);
      ::setMetalFuzziness(metal_fuzziness);
      ::setStratifiedSampling(use_stratified_sampling ? 1 : 0);

      // UI state
      SliderBounds samples_slider_bounds = {0, 0, 0, 0, 1.0f, 256.0f, &samples_per_batch_float};
      SliderBounds intensity_slider_bounds = {0, 0, 0, 0, 0.1f, 3.0f, &light_intensity};
      SliderBounds background_slider_bounds = {0, 0, 0, 0, 0.0f, 3.0f, &background_intensity};
      SliderBounds fuzziness_slider_bounds = {0, 0, 0, 0, 0.0f, 5.0f, &metal_fuzziness};
      SDL_Rect toggle_button_rect = {0, 0, 0, 0};
      SDL_Rect orbit_button_rect = {0, 0, 0, 0};
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
      CudaScene::Scene* gpu_scene = Scene::CudaSceneBuilder::buildGPUScene(scene_desc);

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
                                                     light_intensity, background_intensity, use_stratified_sampling,
                                                     needs_rerender, camera_changed))
               {
                  // Sync samples_per_batch from float slider value
                  samples_per_batch = static_cast<int>(samples_per_batch_float);

                  if (camera_changed)
                  {
                     ::setLightIntensity(light_intensity);
                     ::setBackgroundIntensity(background_intensity);
                     int sampling_mode = use_stratified_sampling ? 1 : 0;
                     std::cout << "Calling setStratifiedSampling(" << sampling_mode << ")" << std::endl;
                     ::setStratifiedSampling(sampling_mode);
                     std::cout << "Sampling strategy changed to: "
                               << (use_stratified_sampling ? "STRATIFIED" : "UNIFORM") << std::endl;
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
                       background_slider_bounds, fuzziness_slider_bounds, toggle_button_rect, orbit_button_rect,
                       accumulation_enabled, samples_per_batch_float, light_intensity, background_intensity,
                       metal_fuzziness, needs_rerender, camera_changed, gui.getShowControls()))
               {
                  // Sync samples_per_batch from float slider value after modification
                  samples_per_batch = static_cast<int>(samples_per_batch_float);

                  if (camera_changed)
                  {
                     ::setLightIntensity(light_intensity);
                     ::setBackgroundIntensity(background_intensity);
                     ::setMetalFuzziness(metal_fuzziness);
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
                                                    fuzziness_slider_bounds, samples_per_batch_float, light_intensity,
                                                    background_intensity, metal_fuzziness, needs_rerender,
                                                    camera_changed, look_from, look_at, vup, w, gui.getShowControls()))
               {
                  // Sync samples_per_batch from float slider value
                  samples_per_batch = static_cast<int>(samples_per_batch_float);

                  if (camera_changed)
                  {
                     ::setLightIntensity(light_intensity);
                     ::setBackgroundIntensity(background_intensity);
                     ::setMetalFuzziness(metal_fuzziness);
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

         // Handle camera changes - restart rendering
         if (camera_changed)
         {
            std::cout << "Camera changed detected, resetting buffers and forcing re-render" << std::endl;
            camera_changed = false;
            current_samples = 0;
            force_immediate_render = true; // Force rendering after camera/settings change
            std::fill(accum_buffer.begin(), accum_buffer.end(), 0.0f);

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
            displayFrame(gui, display_image, current_samples, samples_per_batch, light_intensity, background_intensity,
                         metal_fuzziness, accumulation_enabled, camera_control.isAutoOrbitEnabled(),
                         use_stratified_sampling, samples_slider_bounds, intensity_slider_bounds,
                         background_slider_bounds, fuzziness_slider_bounds, toggle_button_rect, orbit_button_rect);
            image = display_image;
            needs_rerender = false;
         }

         // Render logic: accumulate if enabled, or render once if auto-accumulation is off
         bool should_render = (current_samples < max_samples && !camera_changed && running) || force_immediate_render;
         bool needs_initial_render =
             current_samples == 0 && !accumulation_enabled; // Render at least once when auto-accumulation is off

         if (should_render && (accumulation_enabled || needs_initial_render || force_immediate_render))
         {
            if (force_immediate_render)
            {
               std::cout << "Force immediate render triggered (samples=" << current_samples << ")" << std::endl;
            }
            force_immediate_render = false; // Reset flag after rendering
            renderBatch(display_image, accum_buffer, current_samples, max_samples, samples_per_batch, gamma,
                        d_rand_states, d_accum_buffer, gpu_scene);

            displayFrame(gui, display_image, current_samples, samples_per_batch, light_intensity, background_intensity,
                         metal_fuzziness, accumulation_enabled, camera_control.isAutoOrbitEnabled(),
                         use_stratified_sampling, samples_slider_bounds, intensity_slider_bounds,
                         background_slider_bounds, fuzziness_slider_bounds, toggle_button_rect, orbit_button_rect);

            image = display_image;
         }
         else if (current_samples >= max_samples && !camera_changed)
         {
            // Refresh display even when idle to show logo and UI
            displayFrame(gui, display_image, current_samples, samples_per_batch, light_intensity, background_intensity,
                         metal_fuzziness, accumulation_enabled, camera_control.isAutoOrbitEnabled(),
                         use_stratified_sampling, samples_slider_bounds, intensity_slider_bounds,
                         background_slider_bounds, fuzziness_slider_bounds, toggle_button_rect, orbit_button_rect);
            SDL_Delay(8); // ~60 FPS event polling
         }
         else if (!accumulation_enabled && current_samples > 0 && !camera_changed)
         {
            // Refresh display even when idle to show logo and UI
            displayFrame(gui, display_image, current_samples, samples_per_batch, light_intensity, background_intensity,
                         metal_fuzziness, accumulation_enabled, camera_control.isAutoOrbitEnabled(),
                         use_stratified_sampling, samples_slider_bounds, intensity_slider_bounds,
                         background_slider_bounds, fuzziness_slider_bounds, toggle_button_rect, orbit_button_rect);
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
    * @brief Render a batch of samples using CUDA
    */
   void renderBatch(vector<unsigned char> &display_image, vector<float> &accum_buffer, int &current_samples,
                    int max_samples, int samples_per_batch, float gamma, void *&d_rand_states, void *&d_accum_buffer,
                    CudaScene::Scene* gpu_scene)
   {
      current_samples += samples_per_batch;
      if (current_samples > max_samples)
         current_samples = max_samples;

      int actual_samples_to_add = samples_per_batch;
      if (current_samples > max_samples)
         actual_samples_to_add = max_samples - (current_samples - samples_per_batch);

      // Call CUDA to render and accumulate samples
      unsigned long long cuda_ray_count = ::renderPixelsCUDAAccumulative(
          display_image.data(), accum_buffer.data(), gpu_scene, image_width, image_height, camera_center.x(), camera_center.y(),
          camera_center.z(), pixel00_loc.x(), pixel00_loc.y(), pixel00_loc.z(), pixel_delta_u.x(), pixel_delta_u.y(),
          pixel_delta_u.z(), pixel_delta_v.x(), pixel_delta_v.y(), pixel_delta_v.z(), actual_samples_to_add,
          current_samples, max_depth, &d_rand_states, &d_accum_buffer);

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
                     bool accumulation_enabled, bool auto_orbit_enabled, bool use_stratified_sampling,
                     SliderBounds &samples_slider_bounds, SliderBounds &intensity_slider_bounds,
                     SliderBounds &background_slider_bounds, SliderBounds &fuzziness_slider_bounds,
                     SDL_Rect &toggle_button_rect, SDL_Rect &orbit_button_rect)
   {
      gui.updateDisplay(display_image, image_channels);
      gui.drawLogo();
      gui.drawSampleCountText(current_samples);
      gui.drawUIControls(samples_per_batch, light_intensity, background_intensity, metal_fuzziness,
                         accumulation_enabled, auto_orbit_enabled, use_stratified_sampling, samples_slider_bounds,
                         intensity_slider_bounds, background_slider_bounds, fuzziness_slider_bounds, toggle_button_rect,
                         orbit_button_rect);
      gui.present();
   }
   
   // Forward declaration - scene is created in main.cc
   Scene::SceneDescription createDefaultScene();  // Implemented in main.cc
};

#endif // SDL2_FOUND
