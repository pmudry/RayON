#include "camera/camera.hpp"
#include "constants.hpp"
#include "cpu_renderers/renderer_cpu_parallel.hpp"
#include "cpu_renderers/renderer_cpu_single_thread.hpp"
#include "gpu_renderers/renderer_cuda_host.hpp"
#include "scene_description.hpp"
#include "render/benchmark_config.hpp"
#include "scenes/yaml_scene_loader.hpp" 

#define TINYOBJLOADER_IMPLEMENTATION 
#include "scene_factory.hpp" 
#include "utils.hpp"
#ifdef SDL2_FOUND
#include "gpu_renderers/renderer_cuda_progressive_host.hpp"
#endif

#include "render/render_coordinator.hpp"

#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <iostream>

#include <system_error>

using namespace constants;
using namespace utils;

#ifndef RT_BUILD_TYPE_STRING
#define RT_BUILD_TYPE_STRING "Unknown"
#endif

static constexpr const char *current_build_configuration() { return RT_BUILD_TYPE_STRING; }

struct ProgramArgs
{
   int rendering_method = -1; // -1 means not specified, will ask user
   int samples = -1;          // -1 means not specified, will use defaults based on mode
   int height = IMAGE_HEIGHT;
   int start_samples = 32;           // Number of samples to render initially when moving camera
   bool auto_accumulate = true;      // Enable auto-accumulation by default
   int target_fps = 60;              // Target FPS for interactive rendering (default: 60)
   bool adaptive_depth = false;      // Enable adaptive depth (default: off)
   bool debug_mode = false;          // Enable debug mode overlay
   const char *scene_file = nullptr; // Optional scene file to load
   bool is_benchmark_mode = false;   // Flag to enable benchmark mode
   const char *benchmark_config_file = nullptr; // Path to benchmark config YAML
   const char *benchmark_output_dir = nullptr; // Optional override for benchmark output directory
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
   cout << "  --debug                Enable debug overlay in interactive mode\n";
   cout << "  --benchmark <file>     Activate benchmark mode and load benchmark configuration from YAML file\n";
   cout << "  --benchmark-out <dir>  Specify output directory for benchmark results (default: benchmark_results)\n";
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
      else if (strcmp(argv[i], "--debug") == 0)
      {
         args.debug_mode = true;
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
      else if (strcmp(argv[i], "--benchmark") == 0 && i + 1 < argc)
      {
         args.is_benchmark_mode = true;
         args.benchmark_config_file = argv[++i];
      }
      else if (strcmp(argv[i], "--benchmark-out") == 0 && i + 1 < argc)
      {
         args.benchmark_output_dir = argv[++i];
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
   ColoredStreamBuf cs(cout.rdbuf(), ansi_colors::BOLD_RED);

   cs.enable_colored_cerr();

   ProgramArgs args = parseInput(argc, argv);

   if (args.samples < 0 && args.samples != -1)
      return 1;
   
   // Print application header always
   string compiled_config = current_build_configuration();
   cout << "\n";
   cout << "====================================" << "\n";
   cout << " RayON raytracer v" << version << " - " << compiled_config << "\n";
   cout << " Dr P.-A. Mudry, 2025" << "\n";
   cout << "====================================" << "\n";
#ifdef DIAGS
   cout << "Using features : yaml_scene_loader, unified_scene_descriptions, cuda_optimization_1, BVH" << "\n";
   cout << "fast_rnd, thread_block_optimal, inlining, atomic_reduction, russian_roulette" << "\n";
   cout << "lambertian_cosine_weighted_hemisphere_sampling, lambertian_owen_hash_distribution" << "\n";
   cout << "inter_adaptive_depth, inter_target_fps" << "\n\n";
#endif

   int renderType = 2;
   int image_height;
   int image_width;
   int effective_samples;

   Rayon::BenchmarkConfig benchmark_config; // Declare BenchmarkConfig object

   if (args.is_benchmark_mode)
   {
      cout << "Benchmark mode activated.\n";
      if (!args.benchmark_config_file)
      {
         cerr << "ERROR: --benchmark requires a configuration file path.\n";
         return 1;
      }
      if (!Scene::loadBenchmarkConfigFromYAML(args.benchmark_config_file, benchmark_config))
      {
         cerr << "ERROR: Failed to load benchmark configuration from " << args.benchmark_config_file << "\n";
         return 1;
      }

      // Override rendering parameters with benchmark config values
      image_width = benchmark_config.resolution_width;
      image_height = benchmark_config.resolution_height;
      effective_samples = benchmark_config.target_samples;
      
      renderType = 2; // Force CUDA GPU non-interactive for benchmarks
      args.scene_file = benchmark_config.scene_file.c_str(); // Use scene file from benchmark config

      cout << "Benchmark rendering at resolution: " << image_width << " x " << image_height << " pixels - ";
      cout << "Target samples per pixel: " << effective_samples << "\n";
      if (benchmark_config.max_time_seconds > 0) {
          cout << "Max render time: " << benchmark_config.max_time_seconds << " seconds.\n";
      }
      cout << "\n";
   } else {
        // Original logic for width/height and effective_samples if not in benchmark mode
        // Calculate width maintaining aspect ratio (16:9)
        image_height = args.height;
        image_width = (image_height * 16) / 9;

        // Determine effective samples
        effective_samples = args.samples;
        if (effective_samples == -1)
        {
            if (renderType == 3)
                effective_samples = 2000; // Default for interactive mode
            else
                effective_samples = SAMPLES_PER_PIXEL; // Default for others (64)
        }
        cout << "Rendering at resolution: " << image_width << " x " << image_height << " pixels - ";
        cout << "Samples per pixel: " << effective_samples << "\n\n";

        if (args.rendering_method != -1)
        {
            renderType = args.rendering_method;
        }
        else
        {
            // Choose rendering method
            cout << "Choose rendering method:" << "\n";
            cout << "\t0. CPU sequential" << "\n";
            cout << "\t1. CPU parallel" << "\n";
            cout << "\t2. CUDA GPU (default)" << "\n";
#ifdef SDL2_FOUND
            cout << "\t3. CUDA GPU with interactive SDL display" << "\n";
#endif
            cout << "Enter choice (0, 1, 2"
#ifdef SDL2_FOUND
                << ", 3"
#endif
                << "): ";
            string input;
            getline(cin, input);

            cout << "\n";

            if (!input.empty())
                renderType = stoi(input);
        }
   }

   RndGen::set_seed(1984);

   Scene::SceneDescription scene_desc;

   if (args.scene_file == nullptr)
   {
      cout << "No scene file provided, using default scene." << "\n";
      // scene_desc = Scene::SceneFactory::singleObjectScene();
      scene_desc = Scene::SceneFactory::createDefaultScene();
   }
   else
   {
      scene_desc = Scene::SceneFactory::load(args.scene_file);
   }

   vector<unsigned char> localImage(image_width * image_height * CHANNELS);

   Camera camera(Vec3(0, 0, 0), image_width, image_height, CHANNELS, effective_samples);
   
   // Apply camera settings from scene description
   camera.look_from = scene_desc.camera_position;
   camera.look_at = scene_desc.camera_look_at;
   camera.vup = scene_desc.camera_up;
   camera.vfov = scene_desc.camera_fov;
   camera.updateFrame();

   RenderCoordinator coordinator(camera, scene_desc);

   auto render_start = chrono::high_resolution_clock::now();

   switch (renderType)
   {
   case 0:
   {
      cout << "Using CPU single threaded..." << "\n";
      RendererCPU renderer;
      coordinator.render(renderer, localImage);
      break;
   }
   case 1:
   {
      cout << "Using CPU parallel rendering..." << "\n";
      RendererCPUParallel renderer;
      coordinator.render(renderer, localImage);
      break;
   }

#ifdef SDL2_FOUND
   case 3:
   {
      cout << "Using CUDA GPU with interactive SDL display..." << "\n";
      // camera.samples_per_pixel = 2000;
      RendererCUDAProgressive renderer;
      RendererCUDAProgressive::Settings settings;
      settings.samples_per_batch = args.start_samples;
      settings.auto_accumulate = args.auto_accumulate;
      settings.target_fps = args.target_fps;
      settings.adaptive_depth = args.adaptive_depth;
      settings.debug_mode = args.debug_mode;
      renderer.setSettings(settings);
      coordinator.render(renderer, localImage);
      break;
   }
#endif
   default:
   {
      cout << "Using CUDA GPU rendering..." << "\n";
      RendererCUDA renderer;
      if (args.is_benchmark_mode) {
          renderer.setBenchmarkConfig(benchmark_config);
      }
      coordinator.render(renderer, localImage);
      break;
   }
   }

   cout << "\n";

   auto render_end = chrono::high_resolution_clock::now();
   auto render_duration = render_end - render_start;

   cout.imbue(locale("en_US.UTF-8"));
   cout << "Rays traced: " << fixed << camera.n_rays << "\n";
   double render_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(render_duration).count();
   long long rays_per_second_int = 0;

   if (render_seconds > 0.0)
      rays_per_second_int =
          static_cast<long long>(std::llround(static_cast<double>(camera.n_rays.load()) / render_seconds));

   cout << "Rays/sec: " << rays_per_second_int << "\n";

   string output_path;

   if (args.is_benchmark_mode) {
       // Create benchmark_results directory if it doesn't exist
       string benchmark_dir = args.benchmark_output_dir ? args.benchmark_output_dir : "benchmark_results";
       std::filesystem::create_directories(benchmark_dir);

       // Build timestamp
       auto now = chrono::system_clock::now();
       time_t raw_time = chrono::system_clock::to_time_t(now);
       std::tm local_tm;
#ifdef _WIN32
       localtime_s(&local_tm, &raw_time);
#else
       localtime_r(&raw_time, &local_tm);
#endif
       stringstream ss;
       ss << benchmark_dir << "/" << put_time(&local_tm, "%Y-%m-%d_%H-%M-%S") << "_" << benchmark_config.output_name << ".png";
       output_path = ss.str();
       
       cout << "Saving benchmark results to: " << output_path << "\n";
   } else {
       output_path = utils::FileUtils::buildTimestampedOutputPath();
   }

   utils::FileUtils::dumpImageToFile(localImage, camera.image_width, camera.image_height, "rendered_images/latest.png");
   utils::FileUtils::dumpImageToFile(localImage, camera.image_width, camera.image_height, output_path);

   std::error_code file_size_ec;
   uintmax_t image_size_bytes = filesystem::file_size(output_path, file_size_ec);
   if (file_size_ec)
      image_size_bytes = 0;

   utils::FileUtils::writeRenderStats(camera, scene_desc, output_path, image_size_bytes, render_duration, 
                                      coordinator.getDeviceName(), coordinator.getVramUsage());

   return 0;
}