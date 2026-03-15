#include "camera/camera.hpp"
#include "constants.hpp"
#include "cpu_renderers/renderer_cpu_parallel.hpp"
#include "cpu_renderers/renderer_cpu_single_thread.hpp"
#include "gpu_renderers/renderer_cuda_host.hpp"
#include "scene_description.hpp"
#include "scene_factory.hpp"
#include "utils.hpp"

#ifdef SDL2_FOUND
#include "gpu_renderers/renderer_cuda_progressive_host.hpp"
#endif

#ifdef OPTIX_FOUND
#include "gpu_renderers/renderer_optix_host.hpp"
#endif

#if defined(SDL2_FOUND) && defined(OPTIX_FOUND)
#include "gpu_renderers/renderer_optix_progressive_host.hpp"
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

#ifdef SDL2_FOUND
static GuiTheme parseThemeName(const char *name)
{
   if (!name)
      return GuiTheme::NORD;
   std::string s(name);
   // Convert to lowercase for case-insensitive matching
   for (auto &ch : s)
      ch = static_cast<char>(tolower(ch));
   if (s == "light")
      return GuiTheme::LIGHT;
   if (s == "classic")
      return GuiTheme::CLASSIC;
   if (s == "nord")
      return GuiTheme::NORD;
   if (s == "dracula")
      return GuiTheme::DRACULA;
   if (s == "gruvbox")
      return GuiTheme::GRUVBOX;
   if (s == "catppuccin" || s == "mocha")
      return GuiTheme::CATPPUCCIN;
   return GuiTheme::NORD;
}
#endif

struct ProgramArgs
{
   int rendering_method = -1; // -1 means not specified, will ask user
   int samples = SAMPLES_PER_PIXEL;
   int height = IMAGE_HEIGHT;
   int width = -1; // -1 means derive from height using 16:9
   int samples_per_batch = INTERACTIVE_SAMPLES_PER_BATCH;
   int motion_samples = INTERACTIVE_MOTION_SAMPLES;
   bool auto_accumulate = true;
   bool adaptive_depth = false;
   bool adaptive_sampling = true;
   bool show_menu = false;
   const char *scene_file = nullptr;
   const char *theme = nullptr;
};

void dumpHelp()
{
   cout << "Options:\n";
   cout << "  -h, --help, /?         Show this help message\n";
   cout << "  -m <method>            Rendering method: 0=CPU sequential, 1=CPU parallel,\n";
   cout << "                         2=CUDA offline, 3=CUDA interactive (default: 3)\n";
   cout << "  --menu                 Show interactive method selection menu\n";
   cout << "  -r <WxH>               Arbitrary resolution, e.g. 1920x1080 or 800x600\n";
   cout << "  -r <height>            Preset height (16:9): 2160, 1080, 720, 360, 180 (default: "
        << IMAGE_HEIGHT << ")\n";
   cout << "  --scene <file>         Load scene from YAML file (default: built-in scene)\n";
   cout << "\n";
   cout << "Offline rendering (modes 0, 1, 2):\n";
   cout << "  -s <samples>           Samples per pixel (default: " << SAMPLES_PER_PIXEL << ")\n";
   cout << "\n";
   cout << "Interactive rendering (mode 3):\n";
   cout << "  --samples-per-batch <n>   Samples per batch at rest (default: " << INTERACTIVE_SAMPLES_PER_BATCH << ")\n";
   cout << "  --motion-samples <n>      Samples per batch while camera moves (default: " << INTERACTIVE_MOTION_SAMPLES << ")\n";
   cout << "  --adaptive-depth          Progressively increase max bounce depth\n";
   cout << "  --no-adaptive-sampling    Disable converged-pixel skipping\n";
   cout << "  --no-auto-accumulate      Disable automatic sample accumulation\n";
   cout << "  --theme <name>            GUI theme: light, classic, nord, dracula, gruvbox, catppuccin\n";
}

ProgramArgs parseInput(int argc, char *argv[])
{
   ProgramArgs args;

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
#ifdef OPTIX_FOUND
             || strcmp(argv[i + 1], "4") == 0
#endif
#if defined(SDL2_FOUND) && defined(OPTIX_FOUND)
             || strcmp(argv[i + 1], "5") == 0
#endif
         )
         {
            args.rendering_method = atoi(argv[++i]);
         }
         else
         {
            cout << "Invalid rendering method specified after -m. Allowed values are 0, 1, 2"
#ifdef SDL2_FOUND
                 ", 3"
#endif
#ifdef OPTIX_FOUND
                 ", 4"
#endif
#if defined(SDL2_FOUND) && defined(OPTIX_FOUND)
                 ", 5"
#endif
                 ".\n";
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
         const char *res = argv[++i];
         const char *x_ptr = strchr(res, 'x');
         if (x_ptr != nullptr)
         {
            // WxH format
            int w = atoi(res);
            int h = atoi(x_ptr + 1);
            if (w < 1 || h < 1)
            {
               cerr << "Invalid resolution: " << res << ". Both width and height must be >= 1.\n";
               args.samples = -1;
               return args;
            }
            args.width = w;
            args.height = h;
         }
         else
         {
            // Plain height — accept any positive value (not restricted to presets)
            int height = atoi(res);
            if (height < 1)
            {
               cerr << "Invalid resolution height: " << res << "\n";
               args.samples = -1;
               return args;
            }
            args.height = height;
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
      else if (strcmp(argv[i], "--no-adaptive-sampling") == 0)
      {
         args.adaptive_sampling = false;
      }
      else if (strcmp(argv[i], "--scene") == 0 && i + 1 < argc)
      {
         args.scene_file = argv[++i];
      }
      else if (strcmp(argv[i], "--samples-per-batch") == 0 && i + 1 < argc)
      {
         args.samples_per_batch = atoi(argv[++i]);
         if (args.samples_per_batch < 1)
         {
            cerr << "Invalid samples-per-batch value: " << args.samples_per_batch << " (must be >= 1)\n";
            args.samples = -1; // Indicate error
            return args;
         }
      }
      else if (strcmp(argv[i], "--motion-samples") == 0 && i + 1 < argc)
      {
         args.motion_samples = atoi(argv[++i]);
         if (args.motion_samples < 1)
         {
            cerr << "Invalid motion-samples value: " << args.motion_samples << " (must be >= 1)\n";
            args.samples = -1;
            return args;
         }
      }
      else if (strcmp(argv[i], "--theme") == 0 && i + 1 < argc)
      {
         args.theme = argv[++i];
      }
      else if (strcmp(argv[i], "--menu") == 0)
      {
         args.show_menu = true;
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

   int renderType = 2; // Default to CUDA

   if (args.samples < 0)
      return 1;

   // Calculate width maintaining aspect ratio (16:9)
   int image_height = args.height;
   int image_width = (args.width > 0) ? args.width : (image_height * 16) / 9;
   string compiled_config = current_build_configuration();

   cout << "\n";
   cout << "====================================" << "\n";
   cout << " RayON raytracer v" << version << " - " << compiled_config << "\n";
   cout << " An ISC demo by Dr P.-A. Mudry, 2025-2026" << "\n";
   cout << "====================================" << "\n";
#ifdef DIAGS
   cout << "Using features : yaml_scene_loader, unified_scene_descriptions, cuda_optimization_1, BVH" << "\n";
   cout << "fast_rnd, thread_block_optimal, inlining, atomic_reduction, russian_roulette" << "\n";
   cout << "lambertian_cosine_weighted_hemisphere_sampling, lambertian_owen_hash_distribution" << "\n";
   cout << "inter_adaptive_depth" << "\n\n";
#endif
   cout << "Rendering at resolution: " << image_width << " x " << image_height << " pixels - ";
   cout << "Samples per pixel: " << args.samples << "\n\n";

   if (args.rendering_method != -1)
   {
      renderType = args.rendering_method;
   }
   else if (args.show_menu)
   {
      // Choose rendering method
      cout << "Choose rendering method:" << "\n";
      cout << "\t0. CPU sequential" << "\n";
      cout << "\t1. CPU parallel" << "\n";
      cout << "\t2. CUDA GPU (default)" << "\n";
#ifdef SDL2_FOUND
      cout << "\t3. CUDA GPU with interactive SDL display" << "\n";
#endif
#ifdef OPTIX_FOUND
      cout << "\t4. OptiX GPU (hardware RT cores)" << "\n";
#endif
#if defined(SDL2_FOUND) && defined(OPTIX_FOUND)
      cout << "\t5. OptiX GPU with interactive SDL display" << "\n";
#endif
      cout << "Enter choice (0, 1, 2"
#ifdef SDL2_FOUND
           << ", 3"
#endif
#ifdef OPTIX_FOUND
           << ", 4"
#endif
#if defined(SDL2_FOUND) && defined(OPTIX_FOUND)
           << ", 5"
#endif
           << "): ";
      string input;
      getline(cin, input);

      cout << "\n";

      if (!input.empty())
         renderType = stoi(input);
   }
   else
   {
      // Default: interactive when SDL2 is available, offline CUDA otherwise
#ifdef SDL2_FOUND
      renderType = 3;
#else
      renderType = 2;
#endif
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
      scene_desc = Scene::SceneFactory::fromYAML(args.scene_file);
   }

   vector<unsigned char> localImage(image_width * image_height * CHANNELS);

   Camera camera(Vec3(0, 0, 0), image_width, image_height, CHANNELS, args.samples);

   // Apply camera settings from scene description (YAML or factory)
   camera.look_from = scene_desc.camera_position;
   camera.look_at = scene_desc.camera_look_at;
   camera.vup = scene_desc.camera_up;
   camera.vfov = scene_desc.camera_fov;

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
      camera.samples_per_pixel = INTERACTIVE_MAX_SPP;
      RendererCUDAProgressive renderer;
      RendererCUDAProgressive::Settings settings;
      settings.samples_per_batch = args.samples_per_batch;
      settings.motion_samples = args.motion_samples;
      settings.auto_accumulate = args.auto_accumulate;
      settings.adaptive_depth = args.adaptive_depth;
      settings.adaptive_sampling = args.adaptive_sampling;
      settings.theme = parseThemeName(args.theme);
      renderer.setSettings(settings);
      coordinator.render(renderer, localImage);
      break;
   }
#endif
#ifdef OPTIX_FOUND
   case 4:
   {
      cout << "Using OptiX GPU rendering (hardware RT cores)..." << "\n";
      RendererOptiX renderer;
      coordinator.render(renderer, localImage);
      break;
   }
#endif
#if defined(SDL2_FOUND) && defined(OPTIX_FOUND)
   case 5:
   {
      cout << "Using OptiX GPU with interactive SDL display..." << "\n";
      camera.samples_per_pixel = 2000;
      RendererOptiXProgressive renderer;
      RendererOptiXProgressive::Settings settings;
      settings.samples_per_batch = args.samples_per_batch;
      settings.auto_accumulate = args.auto_accumulate;
      settings.target_fps = 60.0f;
      settings.adaptive_depth = args.adaptive_depth;
      renderer.setSettings(settings);
      coordinator.render(renderer, localImage);
      break;
   }
#endif
   default:
   {
      cout << "Using CUDA GPU rendering..." << "\n";
      RendererCUDA renderer;
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

   const string output_path = utils::FileUtils::buildTimestampedOutputPath();

   utils::FileUtils::dumpImageToFile(localImage, camera.image_width, camera.image_height, "rendered_images/latest.png");
   utils::FileUtils::dumpImageToFile(localImage, camera.image_width, camera.image_height, output_path);

   std::error_code file_size_ec;
   uintmax_t image_size_bytes = filesystem::file_size(output_path, file_size_ec);
   if (file_size_ec)
      image_size_bytes = 0;

   utils::FileUtils::writeRenderStats(camera, output_path, image_size_bytes, render_duration);

   return 0;
}