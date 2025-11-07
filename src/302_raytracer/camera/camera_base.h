/**
 * @class CameraBase
 * @brief Base class for camera with common ray tracing functionality
 *
 * This class contains the core camera parameters, initialization logic, and ray tracing
 * functions that are shared across all rendering implementations (CPU, CUDA, SDL, etc.)
 */
#pragma once

#include "vec3.h"
#include "color.h"
#include "constants.h"
#include "interval.h"
#include "utils.h"

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

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
              int samples_per_pixel = 1)
       : image_width(image_width), image_height(image_height), image_channels(image_channels),
         samples_per_pixel(samples_per_pixel), camera_center(center)
   {
      initialize();
   }

   CameraBase() : CameraBase(Vec3(0, 0, 0), 720, 3, 1) {}

   virtual ~CameraBase() = default;

 protected:
   Point3 camera_center; // Camera center
   Point3 pixel00_loc;   // Location of pixel 0, 0
   Vec3 pixel_delta_u;   // Offset to pixel to the right
   Vec3 pixel_delta_v;   // Offset to pixel below
   Vec3 u, v, w;         // Camera frame basis vectors

   void initialize()
   {
      camera_center = look_from;

      // Determine viewport dimensions
      auto focal_length = (look_from - look_at).length();
      auto theta = utils::degrees_to_radians(vfov);
      auto h = tan(theta / 2);

      auto viewport_height = 2 * h * focal_length;
      auto viewport_width = viewport_height * (double(image_width) / image_height);

      // Compute camera basis vectors
      w = unit_vector(look_from - look_at);
      u = unit_vector(cross(vup, w));
      v = cross(w, u);

      // Calculate the vectors across the horizontal and down the vertical viewport edges
      auto viewport_u = viewport_width * u;
      auto viewport_v = viewport_height * -v;

      // Calculate the horizontal and vertical delta vectors from pixel to pixel
      pixel_delta_u = viewport_u / image_width;
      pixel_delta_v = viewport_v / image_height;

      // Calculate the location of the upper left pixel
      auto viewport_upper_left = camera_center - (focal_length * w) - viewport_u / 2 - viewport_v / 2;
      pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
   }

   /**
    * @brief Sets the pixel color in the image buffer
    */
   inline void setPixel(vector<unsigned char> &image, int x, int y, Color &c)
   {
      int index = (y * image_width + x) * image_channels;

      static const Interval intensity(0.0, 0.999);

      // Apply gamma correction (linear to sRGB) for proper display
      auto linear_to_gamma = [](double linear) { return sqrt(linear); };

      image[index + 0] = static_cast<int>(intensity.clamp(linear_to_gamma(c.x())) * 256);
      image[index + 1] = static_cast<int>(intensity.clamp(linear_to_gamma(c.y())) * 256);
      image[index + 2] = static_cast<int>(intensity.clamp(linear_to_gamma(c.z())) * 256);
   }

   /**
    * @brief Display progress bar for rendering
    */
   void showProgress(int current, int total)
   {
      const int barWidth = 70;
      static int frame = 0;
      const char *spinner = "|/-\\";
      float progress = (float)(current + 1) / total;
      int pos = barWidth * progress;

      cout << "Rendering: " << spinner[frame++ % 4] << " [";
      for (int i = 0; i < barWidth; ++i)
      {
         if (i < pos)
            cout << "█";
         else
            cout << "░";
      }
      cout << "] " << int(progress * 100.0) << " %\r";
      cout.flush();
   }

   /**
    * @brief Convert duration to human-readable time string
    */
   string timeStr(std::chrono::nanoseconds duration)
   {
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
      auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
      auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);

      std::ostringstream s;

      if (minutes.count() > 0)
      {
         s << minutes.count() << " minutes and " << (seconds.count() % 60) << " seconds";
      }
      else if (seconds.count() >= 10)
      {
         s << seconds.count() << " seconds";
      }
      else if (seconds.count() >= 1)
      {
         double sec_with_decimal = ms.count() / 1000.0;
         s << std::fixed << std::setprecision(2) << sec_with_decimal << " seconds";
      }
      else
      {
         s << ms.count() << " milliseconds";
      }

      return s.str();
   }
};
