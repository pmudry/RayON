#include "camera/camera.h"
#include "constants.h"
#include "gpu_renderers/renderer_cuda.h"
#include "hittable_list.h"
#include "scene_builder.h"
#include "scene_description.h"
#include "scene_factory.h"
#include "sphere.h"
#include "utils.h"
#include "yaml_scene_loader.h"

#include <filesystem>
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stb_image_write.h"

using namespace constants;
using namespace utils;

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

struct ProgramArgs
{
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

   if (args.samples < 0)
      return 1;

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

   int choice = 2; // Default to CUDA
   string input;
   getline(cin, input);

   cout << endl;

   if (!input.empty())
      choice = stoi(input);

   // Calculate width maintaining aspect ratio (16:9)
   int image_height = args.height;
   int image_width = (image_height * 16) / 9;

   cout << endl;
   cout << "======================================================" << endl;
   cout << " 302 Ray tracer project v" << ver_major << " -- P.-A. Mudry, ISC 2026" << endl;
   cout << "======================================================" << endl << endl;
   cout << "Using features : yaml_scene_loader, unified_scene_descriptions, cuda_optimization_1, BVH" << endl;
   cout << "fast random (no curand_uniform), thread_block_optimal, inlining, atomic_reduction, russian_roulette,"
        << endl;
   cout << "inter_adaptive_depth, inter_target_fps" << endl << endl;
   cout << "Rendering at resolution: " << image_width << " x " << image_height<< " pixels - ";
   cout << "Samples per pixel: " << args.samples << endl << endl;

   RndGen::set_seed(123);

   Scene::SceneDescription scene_desc;

   if(args.scene_file == nullptr)
   {
      cout << "No scene file provided. Using default scene." << endl;
      scene_desc = Scene::SceneFactory::createDefaultScene();
   }
   else{
      scene_desc = Scene::SceneFactory::fromYAML(args.scene_file);
   }
     
   vector<unsigned char> localImage(image_width * image_height * CHANNELS);
   
   Camera c(Vec3(0, 0, 0), image_width, image_height, CHANNELS, args.samples);   
   
   switch (choice)
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

   // Save output image to resources directory
   dumpImageToFile(localImage, c.image_width, c.image_height, "res/output.png");

   cout.imbue(locale("en_US.UTF-8"));
   cout << "Rays traced: " << fixed << c.n_rays << endl;

   return 0;
}