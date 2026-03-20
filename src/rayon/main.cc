#include "camera/camera.hpp"
#include "constants.hpp"
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
   bool auto_accumulate = true;
   bool adaptive_sampling = true;
   bool hdr_cache = true;
   bool show_menu = false;
   bool use_sobol = true; // Sobol' quasi-random sampler (false = classic PCG)
   // MIS / NEE options
   bool mis_enabled           = true;
   bool motion_gate_mis       = true;
   bool nee_first_bounce_only = false;
   int  nee_stride            = 1;
   const char *scene_file = nullptr;
   const char *theme = nullptr;
};

void dumpHelp()
{
   cout << "Options:\n";
   cout << "  -h, --help, /?         Show this help message\n";
   cout << "  -m <method>            Rendering method: 2=CUDA offline, 3=CUDA interactive (default: 3)\n";
   cout << "                         4=OptiX offline, 5=OptiX interactive (if built with OptiX)\n";
   cout << "  --menu                 Show interactive method selection menu\n";
   cout << "  -r <WxH>               Arbitrary resolution, e.g. 1920x1080 or 800x600\n";
   cout << "  -r <height>            Preset height (16:9): 2160, 1080, 720, 360, 180 (default: "
        << IMAGE_HEIGHT << ")\n";
   cout << "  --scene <file>         Load scene from YAML file (default: built-in scene)\n";
   cout << "\n";
   cout << "Offline rendering (modes 2, 4):\n";
   cout << "  -s <samples>           Samples per pixel (default: " << SAMPLES_PER_PIXEL << ")\n";
   cout << "\n";
   cout << "Interactive rendering (mode 3):\n";
      cout << "  --samples-per-batch <n>   Fixed samples per batch for interactive rendering (default: "
         << INTERACTIVE_SAMPLES_PER_BATCH << ")\n";
   cout << "  --no-adaptive-sampling    Disable converged-pixel skipping\n";
   cout << "  --no-auto-accumulate      Disable automatic sample accumulation\n";
   cout << "  --no-hdr-cache            Disable disk cache for HDR sky textures (always re-decode .hdr)\n";
   cout << "  --sampler <sobol|pcg>     GPU sampler type: sobol = low-discrepancy (default), pcg = classic PRNG\n";
   cout << "  --theme <name>            GUI theme: light, classic, nord, dracula, gruvbox, catppuccin\n";
   cout << "MIS / NEE options (GPU modes 2, 3, 5):\n";
   cout << "  --no-mis                  Disable Multiple-Importance Sampling (max throughput, noisier)\n";
   cout << "  --no-motion-gate-mis      Keep MIS on during camera motion (disable Option A auto-gate; interactive only)\n";
   cout << "  --nee-first-bounce        Restrict NEE shadow rays to the first path bounce (Option B)\n";
   cout << "  --nee-stride <N>          Do NEE on 1 of every N samples, contribution \xc3\x97N (Option C; 1=always)\n";
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
         if (strcmp(argv[i + 1], "2") == 0
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
            cout << "Invalid rendering method specified after -m. Allowed values are 2"
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
      else if (strcmp(argv[i], "--no-adaptive-sampling") == 0)
      {
         args.adaptive_sampling = false;
      }
      else if (strcmp(argv[i], "--no-hdr-cache") == 0)
      {
         args.hdr_cache = false;
      }
      else if (strcmp(argv[i], "--no-mis") == 0)
      {
         args.mis_enabled = false;
      }
      else if (strcmp(argv[i], "--no-motion-gate-mis") == 0)
      {
         args.motion_gate_mis = false;
      }
      else if (strcmp(argv[i], "--nee-first-bounce") == 0)
      {
         args.nee_first_bounce_only = true;
      }
      else if (strcmp(argv[i], "--nee-stride") == 0 && i + 1 < argc)
      {
         args.nee_stride = atoi(argv[++i]);
         if (args.nee_stride < 1)
         {
            cerr << "Invalid nee-stride value: must be >= 1\n";
            args.samples = -1;
            return args;
         }
      }
      else if (strcmp(argv[i], "--sampler") == 0 && i + 1 < argc)
      {
         const char *sampler = argv[++i];
         if (strcmp(sampler, "sobol") == 0)
            args.use_sobol = true;
         else if (strcmp(sampler, "pcg") == 0)
            args.use_sobol = false;
         else
         {
            cerr << "Unknown sampler '" << sampler << "'. Valid options: sobol, pcg\n";
            args.samples = -1;
            return args;
         }
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
      cout << "Enter choice (2"
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

   // Configure GPU sampler (Sobol' by default; ignored by CPU renderers)
   setSobolSampler(args.use_sobol);
   cout << "GPU sampler: " << (args.use_sobol ? "Sobol' (low-discrepancy)" : "PCG (classic PRNG)") << "\n\n";
#ifdef OPTIX_FOUND
   setOptiXSobolSampler(args.use_sobol);
#endif

   Scene::SceneDescription scene_desc;

   if (args.scene_file == nullptr)
   {
      cout << "No scene file provided, using default scene." << "\n";
      // scene_desc = Scene::SceneFactory::singleObjectScene();
      scene_desc = Scene::SceneFactory::createDefaultScene();
   }
   else
   {
      // OptiX renderers (mode 4/5) don't use the CPU BVH — skip it to save time on large scenes.
      const bool is_optix_mode = (renderType == 4 || renderType == 5);
      scene_desc = Scene::SceneFactory::fromYAML(args.scene_file, /*skip_cpu_bvh=*/is_optix_mode);
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
#ifdef SDL2_FOUND
   case 3:
   {
      cout << "Using CUDA GPU with interactive SDL display..." << "\n";
      camera.samples_per_pixel = INTERACTIVE_MAX_SPP;
      RendererCUDAProgressive renderer;
      RendererCUDAProgressive::Settings settings;
      settings.samples_per_batch = args.samples_per_batch;
      settings.auto_accumulate = args.auto_accumulate;
      settings.adaptive_sampling = args.adaptive_sampling;
      settings.hdr_cache = args.hdr_cache;
      settings.theme = parseThemeName(args.theme);
      settings.mis_enabled           = args.mis_enabled;
      settings.motion_gate_mis       = args.motion_gate_mis;
      settings.nee_first_bounce_only = args.nee_first_bounce_only;
      settings.nee_stride            = args.nee_stride;
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
      camera.samples_per_pixel = INTERACTIVE_MAX_SPP;
      RendererOptiXProgressive renderer;
      RendererOptiXProgressive::Settings settings;
      settings.samples_per_batch = args.samples_per_batch;
      settings.auto_accumulate = args.auto_accumulate;
      settings.adaptive_sampling = args.adaptive_sampling;
      settings.hdr_cache = args.hdr_cache;
      settings.theme = parseThemeName(args.theme);
      settings.mis_enabled           = args.mis_enabled;
      settings.motion_gate_mis       = args.motion_gate_mis;
      settings.nee_first_bounce_only = args.nee_first_bounce_only;
      settings.nee_stride            = args.nee_stride;
      renderer.setSettings(settings);
      coordinator.render(renderer, localImage);
      break;
   }
#endif
   default:
   {
      cout << "Using CUDA GPU rendering..." << "\n";
      // Apply MIS/NEE settings to the CUDA global constants before rendering.
      ::setMISEnabled(args.mis_enabled);
      ::setNEEFirstBounceOnly(args.nee_first_bounce_only);
      ::setNEEStride(args.nee_stride);
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