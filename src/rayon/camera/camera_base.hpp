/**
 * @class CameraBase
 * @brief Base class for camera with common ray tracing functionality
 *
 * This class contains the core camera parameters, initialization logic, and ray tracing
 * functions that are shared across all rendering implementations (CPU, CUDA, SDL, etc.)
 */
#pragma once

#include "constants.hpp"
#include "camera_frame.hpp"

#include <atomic>
#include "data_structures/vec3.hpp"

class CameraBase
{
 public:
   // Image parameters
   int image_width;
   int image_height;
   int image_channels; // Number of color channels per pixel (e.g., 3 for RGB)

   // Camera parameters
   double vfov = 35.0;                   // Vertical field of view in degrees
   Point3 look_from = Point3(-2, 2, 5);   // Point camera is looking from
   Point3 look_at = Point3(-2, -0.5, -1); // Point camera is looking at
   Vec3 vup = Vec3(0, 1, 0);             // Camera-relative "up" direction

   // Ray tracing
   std::atomic<long long> n_rays{0};           // Number of rays traced so far with this cam (thread-safe)
   int samples_per_pixel;                      // Number of samples per pixel for anti-aliasing
   const int max_depth = constants::MAX_DEPTH; // Maximum ray bounce depth

   CameraBase(const Point3 &center, const int image_width, const int image_height, const int image_channels,
              int samples_per_pixel = 1, const char *scene_file = nullptr)
       : image_width(image_width), image_height(image_height), image_channels(image_channels),
         samples_per_pixel(samples_per_pixel), camera_center(center)
   {
      updateFrame();
   }

   CameraBase() : CameraBase(Vec3(0, 0, 0), 720, 720, 3, 1, nullptr) {}

   virtual ~CameraBase() = default;

   static inline double degrees_to_radians(double degrees) { return degrees * M_PI / 180.0; }

   CameraFrame buildFrame() const
   {
      CameraFrame frame;
      frame.image_width = image_width;
      frame.image_height = image_height;
      frame.image_channels = image_channels;
      frame.samples_per_pixel = samples_per_pixel;
      frame.max_depth = max_depth;
      frame.camera_center = camera_center;
      frame.pixel00_loc = pixel00_loc;
      frame.pixel_delta_u = pixel_delta_u;
      frame.pixel_delta_v = pixel_delta_v;
      frame.u = u;
      frame.v = v;
      frame.w = w;
      return frame;
   }

   void updateFrame() { initialize(); }

 protected:
   Point3 camera_center; // Camera center
   Point3 pixel00_loc;   // Location of pixel 0, 0
   Vec3 pixel_delta_u;   // Offset to pixel to the right
   Vec3 pixel_delta_v;   // Offset to pixel below
   Vec3 u, v, w;         // Camera frame basis vectors

   void initialize()
   {
      camera_center = look_from;

      const auto focal_length = (look_from - look_at).length();
      const auto theta = degrees_to_radians(vfov);
      const auto h = tan(theta / 2);

      const auto viewport_height = 2 * h * focal_length;
      const auto viewport_width = viewport_height * (double(image_width) / image_height);

      w = unit_vector(look_from - look_at);
      u = unit_vector(cross(vup, w));
      v = cross(w, u);

      const auto viewport_u = viewport_width * u;
      const auto viewport_v = viewport_height * -v;

      pixel_delta_u = viewport_u / image_width;
      pixel_delta_v = viewport_v / image_height;

      const auto viewport_upper_left = camera_center - (focal_length * w) - viewport_u / 2 - viewport_v / 2;
      pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
   }
};
