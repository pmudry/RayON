/**
 * @class CameraControlHandler
 * @brief Handles camera movement logic for the interactive renderer
 *
 * This class manages:
 * - interpreting SDL mouse and keyboard events for camera manipulation
 * - Camera orbit, pan, and zoom calculations
 * - Maintaining camera spherical coordinates (azimuth, elevation, distance)
 */
#pragma once

#ifdef SDL2_FOUND

#include "vec3.hpp"

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
      auto_orbit_enabled = false;
      orbit_speed = 0.3;  // Radians per second (slow rotation)

      Vec3 to_camera = lookfrom - lookat;
      camera_distance = to_camera.length();
      camera_azimuth = atan2(to_camera.x(), to_camera.z());
      camera_elevation = asin(to_camera.y() / camera_distance);
   }

   bool handleKeyDown(SDL_Event &event, bool &accumulation_enabled, float &samples_per_batch, float &light_intensity,
                      float &background_intensity, bool &needs_rerender, 
                      bool &camera_changed)
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
      else if (event.key.keysym.sym == SDLK_o)
      {
         // Toggle auto-orbit
         toggleAutoOrbit();
         return true;
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

   bool handleMouseButtonDown(SDL_Event &event)
   {
      if (event.button.button == SDL_BUTTON_LEFT)
      {
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

   void handleMouseButtonUp(SDL_Event &event)
   {
      if (event.button.button == SDL_BUTTON_LEFT)
      {
         left_button_down = false;
      }
      else if (event.button.button == SDL_BUTTON_RIGHT)
      {
         right_button_down = false;
      }
   }

   bool handleMouseMotion(SDL_Event &event, Point3 &lookfrom,
                          Point3 &lookat, const Vec3 &vup, const Vec3 &w)
   {
      int mouse_x = event.motion.x;
      int mouse_y = event.motion.y;
      int dx = mouse_x - last_mouse_x;
      int dy = mouse_y - last_mouse_y;

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

   bool updateAutoOrbit(Point3 &lookfrom, Point3 &lookat, float delta_time)
   {
      if (!auto_orbit_enabled)
         return false;

      // Update azimuth angle for smooth rotation
      camera_azimuth += orbit_speed * delta_time;
      
      // Keep angle in range [0, 2*PI]
      if (camera_azimuth > 2.0 * M_PI)
         camera_azimuth -= 2.0 * M_PI;

      // Update camera position
      lookfrom.e[0] = lookat.x() + camera_distance * cos(camera_elevation) * sin(camera_azimuth);
      lookfrom.e[1] = lookat.y() + camera_distance * sin(camera_elevation);
      lookfrom.e[2] = lookat.z() + camera_distance * cos(camera_elevation) * cos(camera_azimuth);

      return true;
   }

   void toggleAutoOrbit()
   {
      auto_orbit_enabled = !auto_orbit_enabled;
   }

   void setAutoOrbit(bool enabled)
   {
      auto_orbit_enabled = enabled;
   }

   bool isAutoOrbitEnabled() const
   {
      return auto_orbit_enabled;
   }

 private:
   bool left_button_down = false;
   bool right_button_down = false;
   int last_mouse_x = 0;
   int last_mouse_y = 0;
   double camera_azimuth = 0.0;
   double camera_elevation = 0.0;
   double camera_distance = 0.0;
   bool auto_orbit_enabled = false;
   double orbit_speed = 0.3;  // Radians per second

};

#endif // SDL2_FOUND