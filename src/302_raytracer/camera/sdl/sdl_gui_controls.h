/**
 * @class CameraControlHandler
 * @brief Handles camera movement and interaction for the interactive renderer
 *
 * This class manages:
 * - Mouse and keyboard input for camera control
 * - Camera orbit, pan, and zoom operations
 * - Camera state (position, angles, etc.)
 */
#pragma once

#ifdef SDL2_FOUND

#include "vec3.h"
#include "sdl_gui_handler.h"

#include <SDL.h>
#include <SDL_keycode.h>
#include <algorithm>
#include <cmath>

class CameraControlHandler
{
 public:
   CameraControlHandler() = default;

   void initializeCameraControls(const Point3 &lookfrom, const Point3 &lookat)
   {
      left_button_down = false;
      right_button_down = false;
      last_mouse_x = 0;
      last_mouse_y = 0;

      Vec3 to_camera = lookfrom - lookat;
      camera_distance = to_camera.length();
      camera_azimuth = atan2(to_camera.x(), to_camera.z());
      camera_elevation = asin(to_camera.y() / camera_distance);
   }

   bool handleKeyDown(SDL_Event &event, bool &accumulation_enabled, float &samples_per_batch, float &light_intensity,
                      float &background_intensity, bool &needs_rerender, bool &camera_changed)
   {
      if (event.key.keysym.sym == SDLK_SPACE)
      {
         accumulation_enabled = !accumulation_enabled;
         return true;
      }
      else if (event.key.keysym.sym == SDLK_h)
      {
         // This will be handled separately to toggle GUI controls visibility
         return false; // Return false so it can be handled by the caller
      }
      else if (event.key.keysym.sym == SDLK_UP)
      {
         // Increase samples per batch (capped at 256)
         samples_per_batch = std::min(256.0f, samples_per_batch + 1.0f);
         return true;
      }
      else if (event.key.keysym.sym == SDLK_DOWN)
      {
         // Decrease samples per batch (minimum 1)
         samples_per_batch = std::max(1.0f, samples_per_batch - 1.0f);
         return true;
      }
      else if (event.key.keysym.sym == SDLK_RIGHT)
      {
         light_intensity = std::min(3.0f, light_intensity + 0.1f);
         camera_changed = true;
         return true;
      }
      else if (event.key.keysym.sym == SDLK_LEFT)
      {
         light_intensity = std::max(0.1f, light_intensity - 0.1f);
         camera_changed = true;
         return true;
      }
      return false;
   }

   bool handleMouseButtonDown(SDL_Event &event, bool &dragging_slider, SliderBounds *&active_slider,
                              SliderBounds &samples_slider_bounds, SliderBounds &intensity_slider_bounds,
                              SliderBounds &background_slider_bounds, SDL_Rect &toggle_button_rect,
                              bool &accumulation_enabled, float &samples_per_batch, float &light_intensity,
                              float &background_intensity, bool &needs_rerender, bool &camera_changed,
                              bool show_controls)
   {
      if (event.button.button == SDL_BUTTON_LEFT)
      {
         int mx = event.button.x;
         int my = event.button.y;

         // Only handle slider/button interactions if controls are visible
         if (show_controls)
         {
            // Check if clicking on toggle button
            if (mx >= toggle_button_rect.x && mx <= toggle_button_rect.x + toggle_button_rect.w &&
                my >= toggle_button_rect.y && my <= toggle_button_rect.y + toggle_button_rect.h)
            {
               accumulation_enabled = !accumulation_enabled;
               return true;
            }
            // Check sliders
            else if (checkSliderClick(mx, my, samples_slider_bounds, dragging_slider, active_slider, samples_per_batch,
                                      needs_rerender, camera_changed))
            {
               return true;
            }
            else if (checkSliderClick(mx, my, intensity_slider_bounds, dragging_slider, active_slider, light_intensity,
                                      needs_rerender, camera_changed))
            {
               camera_changed = true;
               return true;
            }
            else if (checkSliderClick(mx, my, background_slider_bounds, dragging_slider, active_slider,
                                      background_intensity, needs_rerender, camera_changed))
            {
               camera_changed = true;
               return true;
            }
         }

         // If controls are hidden or click was outside controls, handle as camera rotation
         left_button_down = true;
         last_mouse_x = event.button.x;
         last_mouse_y = event.button.y;
         return false;
      }
      else if (event.button.button == SDL_BUTTON_RIGHT)
      {
         right_button_down = true;
         last_mouse_x = event.button.x;
         last_mouse_y = event.button.y;
         return false;
      }
      return false;
   }

   void handleMouseButtonUp(SDL_Event &event, bool &dragging_slider, SliderBounds *&active_slider)
   {
      if (event.button.button == SDL_BUTTON_LEFT)
      {
         left_button_down = false;
         dragging_slider = false;
         active_slider = nullptr;
      }
      else if (event.button.button == SDL_BUTTON_RIGHT)
      {
         right_button_down = false;
      }
   }

   bool handleMouseMotion(SDL_Event &event, bool &dragging_slider, SliderBounds *&active_slider,
                          SliderBounds &samples_slider_bounds, SliderBounds &intensity_slider_bounds,
                          SliderBounds &background_slider_bounds, float &samples_per_batch, float &light_intensity,
                          float &background_intensity, bool &needs_rerender, bool &camera_changed, Point3 &lookfrom,
                          Point3 &lookat, const Vec3 &vup, const Vec3 &w, bool show_controls)
   {
      int mouse_x = event.motion.x;
      int mouse_y = event.motion.y;
      int dx = mouse_x - last_mouse_x;
      int dy = mouse_y - last_mouse_y;

      // Only handle slider dragging if controls are visible
      if (show_controls && dragging_slider && active_slider)
      {
         float ratio = (float)(mouse_x - active_slider->x) / active_slider->width;
         ratio = std::max(0.0f, std::min(1.0f, ratio));
         float new_value = active_slider->min_val + ratio * (active_slider->max_val - active_slider->min_val);

         if (active_slider == &samples_slider_bounds)
         {
            samples_per_batch = new_value;
            // No need to re-render, just update for next batch
         }
         else if (active_slider == &intensity_slider_bounds)
         {
            light_intensity = new_value;
            camera_changed = true;
         }
         else if (active_slider == &background_slider_bounds)
         {
            background_intensity = new_value;
            camera_changed = true;
         }
         return true; // Return true so the caller can apply the changes
      }

      if (left_button_down)
      {
         camera_azimuth += dx * 0.01;
         camera_elevation -= dy * 0.01;

         const double max_elevation = 1.5;
         if (camera_elevation > max_elevation)
            camera_elevation = max_elevation;
         if (camera_elevation < -max_elevation)
            camera_elevation = -max_elevation;

         lookfrom.e[0] = lookat.x() + camera_distance * cos(camera_elevation) * sin(camera_azimuth);
         lookfrom.e[1] = lookat.y() + camera_distance * sin(camera_elevation);
         lookfrom.e[2] = lookat.z() + camera_distance * cos(camera_elevation) * cos(camera_azimuth);

         last_mouse_x = mouse_x;
         last_mouse_y = mouse_y;
         return true;
      }
      else if (right_button_down)
      {
         double pan_speed = camera_distance * 0.001;
         Vec3 right = unit_vector(cross(vup, w)) * (-dx * pan_speed);
         Vec3 up = unit_vector(vup) * (dy * pan_speed);

         lookat += right + up;
         lookfrom += right + up;

         last_mouse_x = mouse_x;
         last_mouse_y = mouse_y;
         return true;
      }

      last_mouse_x = mouse_x;
      last_mouse_y = mouse_y;
      return false;
   }

   bool handleMouseWheel(SDL_Event &event, Point3 &lookfrom, Point3 &lookat)
   {
      double zoom_factor = (event.wheel.y > 0) ? 0.9 : 1.1;
      camera_distance *= zoom_factor;

      lookfrom.e[0] = lookat.x() + camera_distance * cos(camera_elevation) * sin(camera_azimuth);
      lookfrom.e[1] = lookat.y() + camera_distance * sin(camera_elevation);
      lookfrom.e[2] = lookat.z() + camera_distance * cos(camera_elevation) * cos(camera_azimuth);

      return true;
   }

 private:
   bool left_button_down = false;
   bool right_button_down = false;
   int last_mouse_x = 0;
   int last_mouse_y = 0;
   double camera_azimuth = 0.0;
   double camera_elevation = 0.0;
   double camera_distance = 0.0;

   bool checkSliderClick(int mx, int my, SliderBounds &slider, bool &dragging_slider, SliderBounds *&active_slider,
                         float &value, bool &needs_rerender, bool &camera_changed)
   {
      if (mx >= slider.x && mx <= slider.x + slider.width && my >= slider.y - 5 && my <= slider.y + slider.height + 5)
      {
         dragging_slider = true;
         active_slider = &slider;

         float ratio = (float)(mx - slider.x) / slider.width;
         value = slider.min_val + ratio * (slider.max_val - slider.min_val);
         value = std::max(slider.min_val, std::min(slider.max_val, value));
         needs_rerender = true;
         return true;
      }
      return false;
   }
};

#endif // SDL2_FOUND
