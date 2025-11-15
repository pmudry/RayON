#include "camera/camera.hpp"
#include "constants.hpp"
#include "scene_description.hpp"
#include "scene_factory.hpp"
#include "utils.hpp"

#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <system_error>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace constants;
using namespace utils;

#ifndef RT_BUILD_TYPE_STRING
#define RT_BUILD_TYPE_STRING "Unknown"
#endif

static constexpr const char *current_build_configuration() { return RT_BUILD_TYPE_STRING; }

// Function to ensure directory exists
void ensureDirectoryExists(const string &filepath)
{
   size_t pos = filepath.find_last_of("/\\");
   if (pos != string::npos)
   {
      string dir = filepath.substr(0, pos);
      filesystem::create_directories(dir);
   }
}

// Function to write image buffer to PNG file
void writeImage(const vector<unsigned char> &image, int image_width, int image_height, const string &filename)
{
   ensureDirectoryExists(filename);
   const int channels = 3; // RGB

   if (stbi_write_png(filename.c_str(), image_width, image_height, channels, image.data(), image_width * channels))
   {
      cout << "Image saved successfully to " << filename << endl;
   }
   else
   {
      cerr << "Failed to save image to " << filename << endl;
   }
}

/**
 * Just for the sake of putting a gradient in a file
 */
void fillGradientImage(vector<unsigned char> &image)
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

void dumpImageToFile(vector<unsigned char> &image, int image_width, int image_height, string name)
{
   // Write image to file
   writeImage(image, image_width, image_height, name);
}

string buildTimestampedOutputPath()
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

string formatDuration(std::chrono::nanoseconds duration)
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

void writeRenderStats(const Camera &camera, const string &image_path, uintmax_t image_size_bytes,
                      std::chrono::nanoseconds render_duration)
{
   filesystem::path stats_path(image_path);
   stats_path.replace_extension(".txt");

   ofstream stats_file(stats_path);
   if (!stats_file)
   {
      cerr << "Failed to write stats to " << stats_path << endl;
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
      cerr << "Failed to write stats JSON to " << stats_json_path << endl;
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
   stats_json << "  \"render_time_pretty\": \"" << formatDuration(render_duration) << "\"\n";
   stats_json << "}\n";
}

struct ProgramArgs
{
   int rendering_method = -1; // -1 means not specified, will ask user
   int samples = SAMPLES_PER_PIXEL;
   int height = IMAGE_HEIGHT;
   int start_samples = 32;           // Number of samples to render initially when moving camera
   bool auto_accumulate = true;      // Enable auto-accumulation by default
   int target_fps = 60;              // Target FPS for interactive rendering (default: 60)
   bool adaptive_depth = false;      // Enable adaptive depth (default: off)
   const char *scene_file = nullptr; // Optional scene file to load
};

void dumpHelp()
{
   cout << "Options:\n";
   cout << "  -h, --help, /?         Show this help message\n";
   cout << "  -m <rendering method>  Set the rendering method (0: CPU sequential, 1: CPU parallel, 2: CUDA GPU, 3: "
           "CUDA GPU with SDL interactive display)\n";
   cout << "  -s <samples>           Set the number of samples per pixel (default: " << SAMPLES_PER_PIXEL << ")\n";
   cout << "  -r <height>            Set vertical resolution (allowed: 2160, 1080, 720, 360, 180, default: "
        << IMAGE_HEIGHT << ")\n";
   cout << "  --scene <file>         Load scene from YAML file (default: built-in scene)\n";
   cout << "  --start-samples <n>    Set initial samples when moving camera in interactive mode (default: 32)\n";
   cout << "  --target-fps <fps>     Set target frame rate for interactive rendering (default: 60)\n";
   cout << "                         Higher values = smoother motion but lower quality preview\n";
   cout << "                         Lower values = better quality preview but less smooth motion\n";
   cout << "  --adaptive-depth       Enable adaptive depth in interactive mode (progressively increases max depth)\n";
   cout << "  --no-auto-accumulate   Disable automatic sample accumulation in interactive mode\n";
}

ProgramArgs parseInput(int argc, char *argv[])
{
   ProgramArgs args;
   const vector<int> allowed_heights = {2160, 1080, 720, 360, 180};

   // Parse command-line arguments
   for (int i = 1; i < argc; ++i)
   {
      if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "/?") == 0)
      {
         cout << "Usage: " << argv[0] << " [options]\n";
         dumpHelp();
         args.samples = -1; // Indicate error
         return args;
      }
      else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)
      {
         // Validate rendering method
         if (strcmp(argv[i + 1], "0") == 0 || strcmp(argv[i + 1], "1") == 0 || strcmp(argv[i + 1], "2") == 0
#ifdef SDL2_FOUND
             || strcmp(argv[i + 1], "3") == 0
#endif
         )
         {
            args.rendering_method = atoi(argv[++i]);
         }
         else
         {
            cout << "Invalid rendering method specified after -m. Allowed values are 0, 1, 2, or 3.\n";
            args.samples = -1; // Indicate error
            return args;
         }
      }
      else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc)
      {
         args.samples = atoi(argv[++i]);
      }
      else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc)
      {
         int height = atoi(argv[++i]);
         if (find(allowed_heights.begin(), allowed_heights.end(), height) != allowed_heights.end())
         {
            args.height = height;
         }
         else
         {
            cerr << "Invalid resolution height: " << height << "\n";
            cerr << "Allowed values: 2160, 1080, 720, 360, 180\n";
            args.samples = -1; // Indicate error
            return args;
         }
      }
      else if (strcmp(argv[i], "--no-auto-accumulate") == 0)
      {
         args.auto_accumulate = false;
      }
      else if (strcmp(argv[i], "--adaptive-depth") == 0)
      {
         args.adaptive_depth = true;
      }
      else if (strcmp(argv[i], "--scene") == 0 && i + 1 < argc)
      {
         args.scene_file = argv[++i];
      }
      else if (strcmp(argv[i], "--start-samples") == 0 && i + 1 < argc)
      {
         args.start_samples = atoi(argv[++i]);
         if (args.start_samples < 1)
         {
            cerr << "Invalid start-samples value: " << args.start_samples << " (must be >= 1)\n";
            args.samples = -1; // Indicate error
            return args;
         }
      }
      else if (strcmp(argv[i], "--target-fps") == 0 && i + 1 < argc)
      {
         args.target_fps = atoi(argv[++i]);
         if (args.target_fps < 1 || args.target_fps > 1000)
         {
            cerr << "Invalid target-fps value: " << args.target_fps << " (must be 1-1000)\n";
            args.samples = -1; // Indicate error
            return args;
         }
      }
      else if (argv[i][0] == '-')
      {
         cerr << "Unknown argument: " << argv[i] << "\n";
         dumpHelp();
         args.samples = -1; // Indicate error
         return args;
      }
      else
      {
         cerr << "Unexpected argument: " << argv[i] << "\n";
         args.samples = -1; // Indicate error
         return args;
      }
   }

   return args;
}

int main(int argc, char *argv[])
{
   // Enable colored error output (all cerr messages will be displayed in red)
   enable_colored_cerr();

   ProgramArgs args = parseInput(argc, argv);

   int renderType = 2; // Default to CUDA

   if (args.samples < 0)
      return 1;

   if (args.rendering_method != -1)
   {
      cout << "Using rendering method from command line: " << args.rendering_method << endl;
      renderType = args.rendering_method;
   }
   else
   {
      // Choose rendering method
      cout << "Choose rendering method:" << endl;
      cout << "\t0. CPU sequential" << endl;
      cout << "\t1. CPU parallel" << endl;
      cout << "\t2. CUDA GPU (default)" << endl;
#ifdef SDL2_FOUND
      cout << "\t3. CUDA GPU with interactive SDL display" << endl;
#endif
      cout << "Enter choice (0, 1, 2"
#ifdef SDL2_FOUND
           << ", 3"
#endif
           << "): ";
      string input;
      getline(cin, input);

      cout << endl;

      if (!input.empty())
         renderType = stoi(input);
   }

   // Calculate width maintaining aspect ratio (16:9)
   int image_height = args.height;
   int image_width = (image_height * 16) / 9;

   string compiled_config = current_build_configuration();

   cout << endl;
   cout << "==============================================" << endl;
   cout << " 302 Ray tracer project v" << ver_major << " - " << compiled_config << endl;
   cout << " P.-A. Mudry, ISC 2026" << endl;
   cout << "==============================================" << endl;
   cout << "Using features : yaml_scene_loader, unified_scene_descriptions, cuda_optimization_1, BVH" << endl;
   cout << "fast_rnd, thread_block_optimal, inlining, atomic_reduction, russian_roulette" << endl;
   cout << "lambertian_cosine_weigthed_hemisphere_sampling, lambertian_owen_hash_distribution" << endl;
   cout << "inter_adaptive_depth, inter_target_fps" << endl << endl;
   cout << "Rendering at resolution: " << image_width << " x " << image_height << " pixels - ";
   cout << "Samples per pixel: " << args.samples << endl << endl;

   RndGen::set_seed(1984);

   Scene::SceneDescription scene_desc;

   if (args.scene_file == nullptr)
   {
      cout << "No scene file provided. Using default scene." << endl;
      // scene_desc = Scene::SceneFactory::singleObjectScene();
      scene_desc = Scene::SceneFactory::createDefaultScene();
   }
   else
   {
      scene_desc = Scene::SceneFactory::fromYAML(args.scene_file);
   }

   vector<unsigned char> localImage(image_width * image_height * CHANNELS);

   Camera c(Vec3(0, 0, 0), image_width, image_height, CHANNELS, args.samples);

   // c.look_at = Vec3(0, 0, -1);

   auto render_start = chrono::high_resolution_clock::now();

   switch (renderType)
   {
   case 0:
      cout << "Using CPU single threaded..." << endl;
      c.renderPixels(scene_desc, localImage);
      break;
   case 1:
      cout << "Using CPU parallel rendering..." << endl;
      c.renderPixelsParallel(scene_desc, localImage);
      break;

#ifdef SDL2_FOUND
   case 3:
      cout << "Using CUDA GPU with interactive SDL display..." << endl;
      c.samples_per_pixel = 10000; // Set high SPP for interactive mode
      c.renderPixelsSDLContinuous(scene_desc, localImage, args.start_samples, args.auto_accumulate, args.target_fps,
                                  args.adaptive_depth);
      break;
#endif
   default:
      cout << "Using CUDA GPU rendering..." << endl;
      c.renderPixelsCUDA(scene_desc, localImage);
      break;
   }

   auto render_end = chrono::high_resolution_clock::now();
   auto render_duration = render_end - render_start;

   const string output_path = buildTimestampedOutputPath();
   
   dumpImageToFile(localImage, c.image_width, c.image_height, "rendered_images/latest.png");
   dumpImageToFile(localImage, c.image_width, c.image_height, output_path);

   std::error_code file_size_ec;
   uintmax_t image_size_bytes = filesystem::file_size(output_path, file_size_ec);
   if (file_size_ec)
      image_size_bytes = 0;

   writeRenderStats(c, output_path, image_size_bytes, render_duration);

   cout.imbue(locale("en_US.UTF-8"));
   cout << "Rays traced: " << fixed << c.n_rays << endl;
   double render_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(render_duration).count();
   long long rays_per_second_int = 0;

   if (render_seconds > 0.0)
      rays_per_second_int = static_cast<long long>(std::llround(static_cast<double>(c.n_rays.load()) / render_seconds));

   cout << "Rays/sec: " << rays_per_second_int << endl;

   return 0;
}