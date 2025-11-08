#include "camera/camera.h"
#include "constants.h"
#include "gpu_renderers/renderer_cuda.h"
#include "hittable_list.h"
#include "scene_builder.h"
#include "scene_description.h"
#include "sphere.h"
#include "utils.h"
#include "yaml_scene_loader.h"

#include <filesystem>
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stb_image_write.h"

using namespace constants;
using namespace utils;

// Global scene file path (set from command line)
static const char *g_scene_file = nullptr;

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

/**
 * @brief Create unified scene description used by all rendering options
 * This is the single source of truth for the scene
 * Uses global g_scene_file if set, otherwise creates default scene
 */
Scene::SceneDescription create_scene_description()
{
   using namespace Scene;
   SceneDescription scene_desc;

   // If scene file specified, try to load it
   if (g_scene_file != nullptr)
   {
      cout << "Attempting to load scene from: " << g_scene_file << endl;
      if (loadSceneFromYAML(g_scene_file, scene_desc))
      {
         // Build BVH if enabled in scene
         if (scene_desc.use_bvh)
         {
            cout << "Building BVH acceleration structure..." << endl;
            scene_desc.buildBVH();
            cout << "BVH built with " << scene_desc.top_level_bvh.nodes.size() << " nodes" << endl;
         }
         return scene_desc; // Successfully loaded
      }
      else
      {
         cerr << "WARNING: Failed to load scene file, using default scene instead" << endl;
      }
   }

   // === Default scene - Materials ===
   int mat_ground = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(0.44, 0.7, 0.95)));
   int mat_golden = scene_desc.addMaterial(MaterialDesc::roughMirror(Vec3(1.0, 0.85, 0.47), 0.03));
   int mat_blue_rough = scene_desc.addMaterial(MaterialDesc::roughMirror(Vec3(0.3, 0.3, 0.91), 0.3));
   int mat_red_dots =
       scene_desc.addMaterial(MaterialDesc::fibonacciDots(Vec3(0.9, 0.1, 0.1), Vec3(0.02, 0.02, 0.02), 12, 0.33f));
   int mat_glass = scene_desc.addMaterial(MaterialDesc::glass(1.5));
   int mat_yellow = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(247 / 255.0, 241 / 255.0, 159 / 255.0)));
   int mat_blue = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(140 / 255.0, 198 / 255.0, 230 / 255.0)));
   int mat_violet = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(168 / 255.0, 144 / 255.0, 192 / 255.0)));
   int mat_rose = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(226 / 255.0, 171 / 255.0, 186 / 255.0)));
   int mat_green = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(152 / 255.0, 199 / 255.0, 191 / 255.0)));
   int mat_light = scene_desc.addMaterial(MaterialDesc::light(Vec3(4.8, 4.1, 3.7)));
   int mat_torus_orange = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(1.0, 0.6, 0.2)));

   // === Default scene - Geometry ===
   scene_desc.addSphere(Vec3(0, -950.5, -1), 950.0, mat_ground); // Ground "plane"
   scene_desc.addSphere(Vec3(-3.5, 0.45, -1.8), 0.8, mat_golden);
   scene_desc.addDisplacedSphere(Vec3(1.2, 0, -2), 0.5, mat_blue_rough, 0.2f, 0);
   scene_desc.addSphere(Vec3(-1.3, 0.18, -5), 0.7, mat_red_dots);
   scene_desc.addSphere(Vec3(-0.7, 0.2, -0.3), 0.6, mat_glass);
   
   // ISC spheres
   scene_desc.addSphere(Vec3(-3.5, -0.3, 1.2), 0.2, mat_yellow);
   scene_desc.addSphere(Vec3(-3.0, -0.3, 1.2), 0.2, mat_blue);
   scene_desc.addSphere(Vec3(-2.5, -0.3, 1.2), 0.2, mat_violet);
   scene_desc.addSphere(Vec3(-2.0, -0.3, 1.2), 0.2, mat_rose);
   scene_desc.addSphere(Vec3(-1.5, -0.3, 1.2), 0.2, mat_green);

   scene_desc.addRectangle(Vec3(-1.0, 3.0, -2.0), Vec3(2.5, 0, 0), Vec3(0, 0, 1.5), mat_light);

   // === NEW SDF SHAPES from Íñigo Quilez's distance functions ===
   // All SDF shapes now support rotation! Last parameter is Vec3(rotX, rotY, rotZ) in radians
   // Rotation is applied in X, Y, Z order using Euler angles
   // Example: Vec3(M_PI * 0.25, 0, 0) rotates 45 degrees around X axis
   
   // Materials for new SDF shapes
   int mat_death_star = scene_desc.addMaterial(MaterialDesc::metal(Vec3(0.6, 0.6, 0.65), 0.1));
   int mat_hollow_sphere = scene_desc.addMaterial(MaterialDesc::roughMirror(Vec3(0.9, 0.7, 0.3), 0.2));
   int mat_octahedron = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(0.2, 0.8, 0.4)));
   int mat_pyramid = scene_desc.addMaterial(MaterialDesc::roughMirror(Vec3(0.8, 0.3, 0.3), 0.15));
   
   // Death Star - positioned at back left, rotated to show cutout
   scene_desc.addSDFDeathStar(Vec3(-3.5, 1.2, -4.5), 0.8, 0.5, 1.0, mat_death_star, Vec3(0, M_PI * 0.3, 0));
   
   // Cut Hollow Sphere - positioned center back, tilted for better view
   scene_desc.addSDFCutHollowSphere(Vec3(0.0, 0.8, -5.0), 0.7, 0.3, 0.1, mat_hollow_sphere, Vec3(M_PI * 0.15, 0, 0));
   
   // Octahedron - positioned at front right, rotated 45 degrees
   scene_desc.addSDFOctahedron(Vec3(2.5, 0.5, -2.0), 0.6, mat_octahedron, Vec3(0, M_PI * 0.25, M_PI * 0.25));
   
   // Pyramid - positioned at right back, rotated to face camera
   scene_desc.addSDFPyramid(Vec3(3.0, 0.0, -4.0), 0.8, mat_pyramid, Vec3(0, M_PI * 0.4, 0));
   
   // Original SDF Torus - rotated to show hole better
   scene_desc.addSDFTorus(Vec3(1.5, 0.7, -3.5), 0.6, 0.2, mat_torus_orange, Vec3(M_PI * 0.3, M_PI * 0.2, 0));

   // Add many more spheres to test BVH performance
   for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
         double x = -4.5 + i * 1.0;
         double z = -8.0 + j * 1.0;
         int mat = (i + j) % 4;
         int material = mat == 0 ? mat_yellow : (mat == 1 ? mat_blue : (mat == 2 ? mat_violet : mat_rose));
         scene_desc.addSphere(Vec3(x, -0.4, z), 0.15, material);
      }
   }

   // Build BVH for default scene (always enabled for default)
   scene_desc.use_bvh = true;
   scene_desc.buildBVH();
   cout << "Built BVH with " << scene_desc.top_level_bvh.nodes.size() << " nodes for " << scene_desc.geometries.size()
        << " geometries" << endl;

   return scene_desc;
}


// Implementation of RendererCUDA::createDefaultScene() - uses unified scene
Scene::SceneDescription RendererCUDA::createDefaultScene() { return create_scene_description(); }

Scene::SceneDescription RendererCUDAProgressive::createDefaultScene() { return create_scene_description(); }

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
         cout << "Options:\n";
         cout << "  -h, --help, /?         Show this help message\n";
         cout << "  -s <samples>           Set the number of samples per pixel (default: " << SAMPLES_PER_PIXEL
              << ")\n";
         cout << "  -r <height>            Set vertical resolution (allowed: 2160, 1080, 720, 360, 180, default: "
              << IMAGE_HEIGHT << ")\n";
         cout << "  --scene <file>         Load scene from YAML file (default: built-in scene)\n";
         cout << "  --start-samples <n>    Set initial samples when moving camera in interactive mode (default: 32)\n";
         cout << "  --target-fps <fps>     Set target frame rate for interactive rendering (default: 60)\n";
         cout << "                         Higher values = smoother motion but lower quality preview\n";
         cout << "                         Lower values = better quality preview but less smooth motion\n";
         cout << "  --adaptive-depth       Enable adaptive depth in interactive mode (progressively increases max depth)\n";
         cout << "  --no-auto-accumulate   Disable automatic sample accumulation in interactive mode\n";
         args.samples = -1;
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
            args.samples = -1;
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
            args.samples = -1;
            return args;
         }
      }
      else if (strcmp(argv[i], "--target-fps") == 0 && i + 1 < argc)
      {
         args.target_fps = atoi(argv[++i]);
         if (args.target_fps < 1 || args.target_fps > 1000)
         {
            cerr << "Invalid target-fps value: " << args.target_fps << " (must be 1-1000)\n";
            args.samples = -1;
            return args;
         }
      }
      else if (argv[i][0] == '-')
      {
         cerr << "Unknown argument: " << argv[i] << "\n";
         cout << "Usage: " << argv[0] << " [options]\n";
         cout << "Options:\n";
         cout << "  -h, --help, /?         Show this help message\n";
         cout << "  -s <samples>           Set the number of samples per pixel (default: " << SAMPLES_PER_PIXEL
              << ")\n";
         cout << "  -r <height>            Set vertical resolution (allowed: 2160, 1080, 720, 360, 180, default: "
              << IMAGE_HEIGHT << ")\n";
         cout << "  --scene <file>         Load scene from YAML file (default: built-in scene)\n";
         cout << "  --target-fps <fps>     Set target frame rate for interactive rendering (default: 60)\n";
         cout << "  --adaptive-depth       Enable adaptive depth in interactive mode (progressively increases max depth)\n";
         cout << "  --start-samples <n>    Set initial samples when moving camera in interactive mode (default: 32)\n";
         cout << "  --no-auto-accumulate   Disable automatic sample accumulation in interactive mode\n";
         args.samples = -1;
         return args;
      }
      else
      {
         cerr << "Unexpected argument: " << argv[i] << "\n";
         args.samples = -1;
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
   {
      return 1;
   }

   // Set global scene file for scene loading
   g_scene_file = args.scene_file;

   // Calculate width maintaining aspect ratio (16:9)
   int image_height = args.height;
   int image_width = (image_height * 16) / 9;

   Camera c(Vec3(0, 0, 0), image_width, image_height, CHANNELS, args.samples);

   vector<unsigned char> image(c.image_width * c.image_height * CHANNELS);

   cout << endl;
   cout << "======================================================" << endl;
   cout << " 302 Ray tracer project v" << ver_major << " -- P.-A. Mudry, ISC 2026" << endl;
   cout << "======================================================" << endl << endl;
   cout << "Using features : yaml_scene_loader, unified_scene_descriptions, cuda_optimization_1, BVH" << endl;
   cout << "fast random (no curand_uniform), thread_block_optimal, inlining, atomic_reduction, russian_roulette," << endl;
   cout << "inter_adaptive_depth, inter_target_fps" << endl << endl;
   cout << "Rendering at resolution: " << c.image_width << " x " << c.image_height << " pixels - ";
   cout << "Samples per pixel: " << args.samples << endl << endl;

   RndGen::set_seed(123);

   // Create unified scene description and convert to CPU scene
   Scene::SceneDescription scene_desc = create_scene_description();
   Hittable_list cpu_scene = Scene::CPUSceneBuilder::buildCPUScene(scene_desc);

   vector<unsigned char> localImage(image.size());

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
   {
      choice = stoi(input);
   }

   switch (choice)
   {
   case 0:
      cout << "Using CPU single threaded..." << endl;
      c.renderPixels(cpu_scene, localImage);
      break;
   case 1:
      cout << "Using CPU parallel rendering..." << endl;
      c.renderPixelsParallel(cpu_scene, localImage);
      break;

#ifdef SDL2_FOUND
   case 3:
      cout << "Using CUDA GPU with interactive SDL display..." << endl;
      c.renderPixelsSDLContinuous(localImage, args.start_samples, args.auto_accumulate, args.target_fps, args.adaptive_depth);
      break;
#endif
   default:
      cout << "Using CUDA GPU rendering..." << endl;
      c.renderPixelsCUDA(localImage);
      break;
   }

   // Save output image to resources directory
   dumpImageToFile(localImage, c.image_width, c.image_height, "res/output.png");

   cout.imbue(locale("en_US.UTF-8"));
   cout << "Rays traced: " << fixed << c.n_rays << endl;

   return 0;
}