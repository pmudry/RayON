/**
 * @class RendererProgressiveSDL
 * @brief Interactive SDL renderer with progressive sample accumulation
 *
 * This renderer focuses on ray-tracing logic with progressive quality improvement.
 * GUI and camera control are delegated to separate handler classes.
 */
#pragma once

#ifdef SDL2_FOUND

#include "interval.h"
#include "camera_base.h"
#include "camera_cuda.h"
#include "sdl_gui_handler.h"
#include "sdl_gui_controls.h"

#include <SDL.h>
#include <algorithm>
#include <chrono>

class RendererProgressiveSDL : virtual public CameraBase
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
    * @param max_samples Maximum total samples to accumulate (default: 2048)
    * @param samples_per_batch Number of samples to add per batch (default: 8)
    * @param auto_accumulate Enable automatic sample accumulation (default: true)
    */
   void renderPixelsSDLContinuous(vector<unsigned char> &image, int max_samples = 2048, int samples_per_batch = 8,
                                  bool auto_accumulate = true)
   {
      // Initialize GUI
      SDLGuiHandler gui(image_width, image_height);
      if (!gui.initialize())
         return;

      gui.printControls(samples_per_batch, max_samples, auto_accumulate);

      // Initialize camera controls
      CameraControlHandler camera_control;
      camera_control.initializeCameraControls(lookfrom, lookat);

      // Ray-tracing state
      bool running = true;
      bool camera_changed = true;
      bool accumulation_enabled = auto_accumulate;
      int current_samples = 0;
      float gamma = 2.0f;
      float light_intensity = 1.0f;
      float background_intensity = 1.0f;
      bool needs_rerender = false;

      // Set initial rendering parameters
      ::setLightIntensity(light_intensity);
      ::setBackgroundIntensity(background_intensity);

      // UI state
      SliderBounds gamma_slider_bounds = {0, 0, 0, 0, 0.5f, 3.0f, &gamma};
      SliderBounds intensity_slider_bounds = {0, 0, 0, 0, 0.1f, 3.0f, &light_intensity};
      SliderBounds background_slider_bounds = {0, 0, 0, 0, 0.0f, 3.0f, &background_intensity};
      SDL_Rect toggle_button_rect = {0, 0, 0, 0};
      bool dragging_slider = false;
      SliderBounds *active_slider = nullptr;

      // Rendering buffers
      SDL_Event event;
      vector<unsigned char> display_image(image_width * image_height * image_channels);
      vector<float> accum_buffer(image_width * image_height * image_channels, 0.0f);
      void *d_rand_states = nullptr;

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
               else if (camera_control.handleKeyDown(event, accumulation_enabled, gamma, light_intensity,
                                                      background_intensity, needs_rerender, camera_changed))
               {
                  if (camera_changed)
                  {
                     ::setLightIntensity(light_intensity);
                     ::setBackgroundIntensity(background_intensity);
                  }
                  if (accumulation_enabled != auto_accumulate)
                  {
                     cout << "\n[Auto-accumulation " << (accumulation_enabled ? "ENABLED" : "DISABLED") << "]" << endl;
                     auto_accumulate = accumulation_enabled;
                  }
               }
            }
            else if (event.type == SDL_MOUSEBUTTONDOWN)
            {
               if (camera_control.handleMouseButtonDown(
                       event, dragging_slider, active_slider, gamma_slider_bounds, intensity_slider_bounds,
                       background_slider_bounds, toggle_button_rect, accumulation_enabled, gamma, light_intensity,
                       background_intensity, needs_rerender, camera_changed, gui.getShowControls()))
               {
                  if (camera_changed)
                  {
                     ::setLightIntensity(light_intensity);
                     ::setBackgroundIntensity(background_intensity);
                  }
                  if (accumulation_enabled != auto_accumulate)
                  {
                     cout << "\n[Auto-accumulation " << (accumulation_enabled ? "ENABLED" : "DISABLED") << "]" << endl;
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
               if (camera_control.handleMouseMotion(event, dragging_slider, active_slider, gamma_slider_bounds,
                                                    intensity_slider_bounds, background_slider_bounds, gamma,
                                                    light_intensity, background_intensity, needs_rerender,
                                                    camera_changed, lookfrom, lookat, vup, w, gui.getShowControls()))
               {
                  if (camera_changed)
                  {
                     ::setLightIntensity(light_intensity);
                     ::setBackgroundIntensity(background_intensity);
                  }
                  camera_changed = true;
               }
            }
            else if (event.type == SDL_MOUSEWHEEL)
            {
               if (camera_control.handleMouseWheel(event, lookfrom, lookat))
               {
                  camera_changed = true;
               }
            }
         }

         // Handle camera changes - restart rendering
         if (camera_changed)
         {
            camera_changed = false;
            current_samples = 0;
            std::fill(accum_buffer.begin(), accum_buffer.end(), 0.0f);

            if (d_rand_states != nullptr)
            {
               freeDeviceRandomStates(d_rand_states);
               d_rand_states = nullptr;
            }

            initialize(); // Recalculate camera parameters
         }

         // Reprocess with new gamma if needed (without re-rendering)
         if (needs_rerender && current_samples > 0)
         {
            applyGammaCorrection(display_image, accum_buffer, current_samples, gamma);
            displayFrame(gui, display_image, current_samples, gamma, light_intensity, background_intensity,
                         accumulation_enabled, gamma_slider_bounds, intensity_slider_bounds, background_slider_bounds,
                         toggle_button_rect);
            image = display_image;
            needs_rerender = false;
         }

         // Accumulate more samples if enabled
         if (current_samples < max_samples && !camera_changed && running && accumulation_enabled)
         {
            renderBatch(display_image, accum_buffer, current_samples, max_samples, samples_per_batch, gamma,
                        d_rand_states);

            displayFrame(gui, display_image, current_samples, gamma, light_intensity, background_intensity,
                         accumulation_enabled, gamma_slider_bounds, intensity_slider_bounds, background_slider_bounds,
                         toggle_button_rect);

            image = display_image;
         }
         else if (current_samples >= max_samples && !camera_changed)
         {
            SDL_Delay(16); // ~60 FPS event polling
         }
      }

      auto total_end = std::chrono::high_resolution_clock::now();
      cout << "\nTotal session time: " << timeStr(total_end - total_start) << endl;

      // Cleanup device resources
      if (d_rand_states != nullptr)
      {
         freeDeviceRandomStates(d_rand_states);
      }

      gui.cleanup();
   }

 private:
   /**
    * @brief Render a batch of samples using CUDA
    */
   void renderBatch(vector<unsigned char> &display_image, vector<float> &accum_buffer, int &current_samples,
                    int max_samples, int samples_per_batch, float gamma, void *&d_rand_states)
   {
      current_samples += samples_per_batch;
      if (current_samples > max_samples)
         current_samples = max_samples;

      int actual_samples_to_add = samples_per_batch;
      if (current_samples > max_samples)
         actual_samples_to_add = max_samples - (current_samples - samples_per_batch);

      // Call CUDA to render and accumulate samples
      unsigned long long cuda_ray_count = ::renderPixelsCUDAAccumulative(
          display_image.data(), accum_buffer.data(), image_width, image_height, camera_center.x(), camera_center.y(),
          camera_center.z(), pixel00_loc.x(), pixel00_loc.y(), pixel00_loc.z(), pixel_delta_u.x(), pixel_delta_u.y(),
          pixel_delta_u.z(), pixel_delta_v.x(), pixel_delta_v.y(), pixel_delta_v.z(), actual_samples_to_add,
          current_samples, max_depth, &d_rand_states);

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
   void displayFrame(SDLGuiHandler &gui, const vector<unsigned char> &display_image, int current_samples, float gamma,
                     float light_intensity, float background_intensity, bool accumulation_enabled,
                     SliderBounds &gamma_slider_bounds, SliderBounds &intensity_slider_bounds,
                     SliderBounds &background_slider_bounds, SDL_Rect &toggle_button_rect)
   {
      gui.updateDisplay(display_image, image_channels);
      gui.drawLogo();
      gui.drawSampleCountText(current_samples);
      gui.drawUIControls(gamma, light_intensity, background_intensity, accumulation_enabled, gamma_slider_bounds,
                         intensity_slider_bounds, background_slider_bounds, toggle_button_rect);
      gui.present();
   }
};

#endif // SDL2_FOUND
