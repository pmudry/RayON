#pragma once

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "data_structures/color.hpp"
#include "data_structures/interval.hpp"
#include "render_target.hpp"

namespace render
{
inline float applyGammaCorrection(float linear_value, float gamma)
{
   return powf(std::max(linear_value, 0.0f), 1.0f / gamma);
}

inline void writePixel(const RenderTargetView &target, int x, int y, const Color &c, float gamma = 2.0f)
{
   if (!target.isValid())
      return;

   const int index = (y * target.width + x) * target.channels;
   static const Interval intensity(0.0, 0.999);

   const float r = intensity.clamp(applyGammaCorrection(static_cast<float>(c.x()), gamma));
   const float g = intensity.clamp(applyGammaCorrection(static_cast<float>(c.y()), gamma));
   const float b = intensity.clamp(applyGammaCorrection(static_cast<float>(c.z()), gamma));

   auto &buffer = *target.pixels;
   buffer[index + 0] = static_cast<unsigned char>(256 * r);
   if (target.channels > 1)
      buffer[index + 1] = static_cast<unsigned char>(256 * g);
   if (target.channels > 2)
      buffer[index + 2] = static_cast<unsigned char>(256 * b);
}

inline void convertAccumBufferToImage(const RenderTargetView &target, const std::vector<float> &accum_buffer,
                                      int num_samples, float gamma = 2.0f)
{
   if (!target.isValid() || accum_buffer.empty() || num_samples <= 0)
      return;

   static const Interval intensity_range(0.0, 0.999);

   auto &image = *target.pixels;
   for (int j = 0; j < target.height; ++j)
   {
      for (int i = 0; i < target.width; ++i)
      {
         const int pixel_idx = j * target.width + i;
         const int image_idx = pixel_idx * target.channels;
         const int accum_idx = pixel_idx * 3;

         float r = accum_buffer[accum_idx + 0] / num_samples;
         float g = accum_buffer[accum_idx + 1] / num_samples;
         float b = accum_buffer[accum_idx + 2] / num_samples;

         r = applyGammaCorrection(r, gamma);
         g = applyGammaCorrection(g, gamma);
         b = applyGammaCorrection(b, gamma);

         image[image_idx + 0] = static_cast<unsigned char>(256 * intensity_range.clamp(r));
         image[image_idx + 1] = static_cast<unsigned char>(256 * intensity_range.clamp(g));
         image[image_idx + 2] = static_cast<unsigned char>(256 * intensity_range.clamp(b));

         if (target.channels == 4)
            image[image_idx + 3] = 255;
      }
   }
}

inline void showProgress(int current, int total)
{
   using namespace std;
   const int barWidth = 70;
   static int frame = 0;
   const char *spinner = "|/-\\";
   float progress = (float)(current + 1) / total;
   int pos = barWidth * progress;

   cout << "Rendering: " << spinner[frame++ % 4] << " [";
   for (int i = 0; i < barWidth; ++i)
   {
      if (i < pos)
         std::cout << "█";
      else
         cout << "░";
   }
   cout << "] " << int(progress * 100.0) << " %\r";
   cout.flush();
}

inline std::string timeStr(std::chrono::nanoseconds duration)
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
} // namespace render
