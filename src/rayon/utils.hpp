#pragma once

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "camera.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

namespace utils
{

class FileUtils
{
 public:
   // Function to ensure directory exists
   static void ensureDirectoryExists(const string &filepath)
   {
      size_t pos = filepath.find_last_of("/\\");
      if (pos != string::npos)
      {
         string dir = filepath.substr(0, pos);
         filesystem::create_directories(dir);
      }
   }

   // Function to write image buffer to PNG file
   static void writeImage(const vector<unsigned char> &image, int image_width, int image_height, const string &filename)
   {
      ensureDirectoryExists(filename);
      const int channels = 3; // RGB

      if (stbi_write_png(filename.c_str(), image_width, image_height, channels, image.data(), image_width * channels))
      {
         cout << "Image saved successfully to " << filename << "\n";
      }
      else
      {
         cerr << "Failed to save image to " << filename << "\n";
      }
   }

   /**
    * Just for the sake of putting a gradient in a file
    */
   static void fillGradientImage(vector<unsigned char> &image, int IMAGE_WIDTH, int IMAGE_HEIGHT, int CHANNELS)
   {
      // Generate a simple gradient
      for (int y = 0; y < IMAGE_HEIGHT; ++y)
      {
         for (int x = 0; x < IMAGE_WIDTH; ++x)
         {
            int index = (y * IMAGE_WIDTH + x) * CHANNELS;
            image[index] = static_cast<unsigned char>(255.0 * y / IMAGE_WIDTH);      // Red
            image[index + 1] = static_cast<unsigned char>(255.0 * x / IMAGE_HEIGHT); // Green
            image[index + 2] = 100;                                                  // Blue
         }
      }
   }

   static void dumpImageToFile(vector<unsigned char> &image, int image_width, int image_height, string name)
   {
      // Write image to file
      writeImage(image, image_width, image_height, name);
   }

   static string buildTimestampedOutputPath()
   {
      auto now = chrono::system_clock::now();
      time_t raw_time = chrono::system_clock::to_time_t(now);
      std::tm local_tm;
#ifdef _WIN32
      localtime_s(&local_tm, &raw_time);
#else
      localtime_r(&raw_time, &local_tm);
#endif

      stringstream ss;
      ss << "rendered_images/output_" << put_time(&local_tm, "%Y-%m-%d_%H-%M-%S") << ".png";
      return ss.str();
   }

   static string formatDuration(std::chrono::nanoseconds duration)
   {
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
      auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
      auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);

      std::ostringstream s;

      if (minutes.count() > 0)
      {
         s << minutes.count() << "m " << (seconds.count() % 60) << "s";
      }
      else if (seconds.count() >= 10)
      {
         s << seconds.count() << "s";
      }
      else if (seconds.count() >= 1)
      {
         double sec_with_decimal = ms.count() / 1000.0;
         s << std::fixed << std::setprecision(2) << sec_with_decimal << "s";
      }
      else
      {
         s << ms.count() << "ms";
      }

      return s.str();
   }

   static void writeRenderStats(const Camera &camera, const string &image_path, uintmax_t image_size_bytes,
                                std::chrono::nanoseconds render_duration)
   {
      filesystem::path stats_path(image_path);
      stats_path.replace_extension(".txt");

      ofstream stats_file(stats_path);
      if (!stats_file)
      {
         cerr << "Failed to write stats to " << stats_path << "\n";
         return;
      }

      auto render_ms = std::chrono::duration_cast<std::chrono::milliseconds>(render_duration).count();
      double render_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(render_duration).count();
      double rays_per_second = 0.0;
      if (render_seconds > 0.0)
      {
         rays_per_second = static_cast<double>(camera.n_rays.load()) / render_seconds;
      }
      long long rays_per_second_int = static_cast<long long>(std::llround(rays_per_second));

      stats_file << "image: " << filesystem::path(image_path).filename().string() << '\n';
      stats_file << "samples_per_pixel: " << camera.samples_per_pixel << '\n';
      stats_file << "resolution: " << camera.image_width << " x " << camera.image_height << '\n';
      stats_file << "max_depth: " << camera.max_depth << '\n';
      stats_file << "rays_traced: " << camera.n_rays.load() << '\n';
      stats_file << "image_size_bytes: " << image_size_bytes << '\n';
      stats_file << "rays_per_second: " << rays_per_second_int << '\n';
      stats_file << "render_time_ms: " << render_ms << '\n';
      stats_file << "render_time_pretty: " << formatDuration(render_duration) << '\n';

      filesystem::path stats_json_path(image_path);
      stats_json_path.replace_extension(".json");
      ofstream stats_json(stats_json_path);
      if (!stats_json)
      {
         cerr << "Failed to write stats JSON to " << stats_json_path << "\n";
         return;
      }

      stats_json << "{\n";
      stats_json << "  \"image\": \"" << filesystem::path(image_path).filename().string() << "\",\n";
      stats_json << "  \"samples_per_pixel\": " << camera.samples_per_pixel << ",\n";
      stats_json << "  \"resolution\": { \"width\": " << camera.image_width << ", \"height\": " << camera.image_height
                 << " },\n";
      stats_json << "  \"max_depth\": " << camera.max_depth << ",\n";
      stats_json << "  \"rays_traced\": " << camera.n_rays.load() << ",\n";
      stats_json << "  \"image_size_bytes\": " << image_size_bytes << ",\n";
      stats_json << "  \"rays_per_second\": " << rays_per_second_int << ",\n";
      stats_json << "  \"render_time_ms\": " << render_ms << ",\n";
      stats_json << "  \"render_time_pretty\": \"" << formatDuration(render_duration) << "\n";
      stats_json << "}\n";
   }
}; // class FileUtils

// ANSI color codes for terminal output
namespace ansi_colors
{
const char *const RESET = "\033[0m";
const char *const RED = "\033[31m";
const char *const GREEN = "\033[32m";
const char *const YELLOW = "\033[33m";
const char *const BLUE = "\033[34m";
const char *const MAGENTA = "\033[35m";
const char *const CYAN = "\033[36m";
const char *const WHITE = "\033[37m";
const char *const BOLD_RED = "\033[1;31m";
} // namespace ansi_colors

// Custom streambuf that adds color prefix
class ColoredStreamBuf : public std::streambuf
{
 private:
   std::streambuf *original_buf;
   const char *color_code;
   bool at_line_start;

 public:
   // Global streambuf instance for cerr coloring
   ColoredStreamBuf *colored_cerr_buf = nullptr;

   ColoredStreamBuf(std::streambuf *buf, const char *color) : original_buf(buf), color_code(color), at_line_start(true)
   {
   }

   ~ColoredStreamBuf() override = default;

   // Function to enable colored cerr output
   void enable_colored_cerr()
   {
      if (!colored_cerr_buf)
      {
         colored_cerr_buf = new ColoredStreamBuf(std::cerr.rdbuf(), ansi_colors::BOLD_RED);
         std::cerr.rdbuf(colored_cerr_buf);
      }
   }

   // Function to disable colored cerr output (cleanup)
   void disable_colored_cerr()
   {
      if (colored_cerr_buf)
      {
         delete colored_cerr_buf;
         colored_cerr_buf = nullptr;
      }
   }

 protected:
   int overflow(int c) override
   {
      if (at_line_start && c != EOF)
      {
         // Write color code at start of line
         for (const char *p = color_code; *p; ++p)
         {
            original_buf->sputc(*p);
         }
         at_line_start = false;
      }

      if (c == '\n')
      {
         // Write reset code before newline
         for (const char *p = ansi_colors::RESET; *p; ++p)
         {
            original_buf->sputc(*p);
         }
         at_line_start = true;
      }

      return original_buf->sputc(c);
   }

   int sync() override { return original_buf->pubsync(); }   
};

}; // namespace utils
