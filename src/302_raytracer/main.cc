#include "camera/camera.h"
#include "constants.h"
#include "hittable_list.h"
#include "sphere.h"
#include "utils.h"
#include "scene_description.h"
#include "scene_builder.h"
#include "gpu_renderers/renderer_cuda.h"

#include <filesystem>
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stb_image_write.h"

using namespace constants;
using namespace utils;

// Function to write image buffer to PNG file
void writeImage(const vector<unsigned char> &image, int image_width, int image_height, const string &filename)
{
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

using scene = Hittable_list;

scene demo_scene()
{
   scene s;

   auto material_uniform_red = make_shared<Constant>(Color(1, 0.0, 0.0));
   auto material_uniform_blue = make_shared<Constant>(Color(0, 0.0, 1.0));
   auto material_normals = make_shared<ShowNormals>(Color(0, 0.0, 0.0));
   auto material_lambert = make_shared<Lambertian>(Color(0.7, 0.7, 0.7));
   auto material_metal = make_shared<Lambertian>(Color(0.7, 0.7, 0.7));

   s.add(make_shared<Sphere>(Point3(0, -950.5, -1), 950, material_uniform_red)); // Ground
   s.add(make_shared<Sphere>(Point3(-3.5, 0.45, -1.8), .8, material_uniform_blue));
   s.add(make_shared<Sphere>(Point3(-1.3, 0.18, -5), .7, material_uniform_blue));
   s.add(make_shared<Sphere>(Point3(-.7, .2, -.3), .6, material_uniform_blue));
   s.add(make_shared<Sphere>(Point3(1.2, 0, -2), 0.5, material_uniform_blue));

   // Small "ISC" spheres at the bottom
   for (int i = 0; i < 5; i++)
   {
      s.add(make_shared<Sphere>(Point3(-3.5 + i * 0.5, -0.3, 1.2), 0.2, material_lambert));
   }

   return s;
}

/**
 * @brief Create demo scene using new SceneDescription API
 * This demonstrates the unified scene format that works for both CPU and GPU
 */
Scene::SceneDescription create_scene_description()
{
   using namespace Scene;
   SceneDescription scene;
   
   // === Define Materials ===
   int mat_red = scene.addMaterial(MaterialDesc::constant(Vec3(1.0, 0.0, 0.0)));
   int mat_blue = scene.addMaterial(MaterialDesc::constant(Vec3(0.0, 0.0, 1.0)));
   int mat_lambert = scene.addMaterial(MaterialDesc::lambertian(Vec3(0.7, 0.7, 0.7)));
   
   // === Add Geometry ===
   scene.addSphere(Vec3(0, -950.5, -1), 950.0, mat_red);  // Ground
   scene.addSphere(Vec3(-3.5, 0.45, -1.8), 0.8, mat_blue);
   scene.addSphere(Vec3(-1.3, 0.18, -5), 0.7, mat_blue);
   scene.addSphere(Vec3(-0.7, 0.2, -0.3), 0.6, mat_blue);
   scene.addSphere(Vec3(1.2, 0, -2), 0.5, mat_blue);
   
   // Small "ISC" spheres
   for (int i = 0; i < 5; i++) {
      scene.addSphere(Vec3(-3.5 + i * 0.5, -0.3, 1.2), 0.2, mat_lambert);
   }
   
   return scene;
}

struct ProgramArgs
{
   int samples = SAMPLES_PER_PIXEL;
   int height = IMAGE_HEIGHT;
   int start_samples = 32;      // Number of samples to render initially when moving camera
   bool auto_accumulate = true; // Enable auto-accumulation by default
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
         cout << "  --start-samples <n>    Set initial samples when moving camera in interactive mode (default: 32)\n";
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

/**
 * @brief Create exact replica of original CUDA hardcoded scene for comparison
 * This scene matches the legacy hit_world() function in renderer_cuda.cu
 */
Scene::SceneDescription create_original_cuda_scene()
{
   using namespace Scene;
   SceneDescription scene_desc;
   
   // === Materials matching original scene ===
   
   // Ground: Lambertian with blue tint
   int mat_ground = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(0.44, 0.7, 0.95)));
   
   // Left golden sphere: Rough mirror with golden tint
   int mat_golden = scene_desc.addMaterial(MaterialDesc::roughMirror(Vec3(1.0, 0.85, 0.47), 0.03));
   
   // Blue rough mirror (golf ball - but without displacement for now)
   int mat_blue_rough = scene_desc.addMaterial(MaterialDesc::roughMirror(Vec3(0.3, 0.3, 0.91), 0.3));
   
   // Red-black dotted sphere using Fibonacci dots pattern
   int mat_red_dots = scene_desc.addMaterial(MaterialDesc::fibonacciDots(
       Vec3(0.9, 0.1, 0.1),        // Base red color
       Vec3(0.02, 0.02, 0.02),     // Near-black dot color
       12,                          // Number of dots (Fibonacci grid spacing)
       0.33f                        // Dot radius in radians (~13 degrees)
   ));
   
   // Glass sphere
   int mat_glass = scene_desc.addMaterial(MaterialDesc::glass(1.5));
   
   // ISC Logo spheres colors
   int mat_yellow = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(247/255.0, 241/255.0, 159/255.0)));
   int mat_blue = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(140/255.0, 198/255.0, 230/255.0)));
   int mat_violet = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(168/255.0, 144/255.0, 192/255.0)));
   int mat_rose = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(226/255.0, 171/255.0, 186/255.0)));
   int mat_green = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(152/255.0, 199/255.0, 191/255.0)));
   
   // Area light
   int mat_light = scene_desc.addMaterial(MaterialDesc::light(Vec3(4.8, 4.1, 3.7)));
   
   // === Geometry matching original scene ===
   
   // Ground sphere
   scene_desc.addSphere(Vec3(0, -950.5, -1), 950.0, mat_ground);
   
   // Left golden sphere
   scene_desc.addSphere(Vec3(-3.5, 0.45, -1.8), 0.8, mat_golden);
   
   // "Golf" ball with dimple displacement
   scene_desc.addDisplacedSphere(Vec3(1.2, 0, -2), 0.5, mat_blue_rough, 0.2f, 0);
   
   // Red-black dotted sphere with Fibonacci pattern
   scene_desc.addSphere(Vec3(-1.3, 0.18, -5), 0.7, mat_red_dots);
   
   // Glass sphere
   scene_desc.addSphere(Vec3(-0.7, 0.2, -0.3), 0.6, mat_glass);
   
   // ISC Logo spheres (5 small spheres)
   scene_desc.addSphere(Vec3(-3.5, -0.3, 1.2), 0.2, mat_yellow);
   scene_desc.addSphere(Vec3(-3.0, -0.3, 1.2), 0.2, mat_blue);
   scene_desc.addSphere(Vec3(-2.5, -0.3, 1.2), 0.2, mat_violet);
   scene_desc.addSphere(Vec3(-2.0, -0.3, 1.2), 0.2, mat_rose);
   scene_desc.addSphere(Vec3(-1.5, -0.3, 1.2), 0.2, mat_green);
   
   // Area light (rectangular)
   scene_desc.addRectangle(Vec3(-1.0, 3.0, -2.0), Vec3(2.5, 0, 0), Vec3(0, 0, 1.5), mat_light);
   
   return scene_desc;
}

/**
 * @brief Test the new scene-based CUDA rendering with exact replica of original scene
 * This allows direct comparison between old (hardcoded) and new (data-driven) architectures
 */
void test_new_scene_cuda_rendering(int width, int height, int samples)
{
   using namespace Scene;
   
   cout << "\n========================================" << endl;
   cout << "TESTING NEW SCENE-BASED CUDA RENDERING" << endl;
   cout << "Exact replica of original hardcoded scene" << endl;
   cout << "========================================\n" << endl;
   
   // Create scene matching original
   SceneDescription scene_desc = create_original_cuda_scene();
   
   cout << "Scene created: " << scene_desc.materials.size() << " materials, " 
        << scene_desc.geometries.size() << " geometries" << endl;
   
   // Build GPU scene
   cout << "Building GPU scene..." << endl;
   CudaScene::Scene* gpu_scene = CudaSceneBuilder::buildGPUScene(scene_desc);
   cout << "GPU scene built successfully!" << endl;
   
   // Setup camera - MATCHING Camera class parameters exactly
   int max_depth = 50;
   
   // Camera parameters from CameraBase
   Point3 look_from = Point3(-2, 2, 5);
   Point3 look_at = Point3(-2, -0.5, -1);
   Vec3 vup = Vec3(0, 1, 0);
   double vfov = 35.0;  // degrees
   
   // Calculate camera parameters
   auto focal_length = (look_from - look_at).length();
   auto theta = utils::degrees_to_radians(vfov);
   auto h = tan(theta / 2);
   
   auto viewport_height = 2 * h * focal_length;
   auto viewport_width = viewport_height * (double(width) / height);
   
   // Camera basis vectors
   Vec3 w = unit_vector(look_from - look_at);
   Vec3 u = unit_vector(cross(vup, w));
   Vec3 v = cross(w, u);
   
   // Viewport vectors
   Vec3 viewport_u = viewport_width * u;
   Vec3 viewport_v = viewport_height * -v;
   
   // Pixel deltas
   Vec3 pixel_delta_u = viewport_u / width;
   Vec3 pixel_delta_v = viewport_v / height;
   
   // First pixel location
   Vec3 viewport_upper_left = look_from - (focal_length * w) - viewport_u / 2 - viewport_v / 2;
   Vec3 pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
   
   // Allocate image buffer
   vector<unsigned char> image(width * height * 3);
   
   cout << "Rendering " << width << "x" << height << " with " << samples << " samples..." << endl;
   auto start_time = std::chrono::high_resolution_clock::now();
   
   // Render using new scene-based function
   unsigned long long ray_count = renderPixelsCUDAWithScene(
       image.data(), gpu_scene, width, height,
       look_from.x(), look_from.y(), look_from.z(),
       pixel00_loc.x(), pixel00_loc.y(), pixel00_loc.z(),
       pixel_delta_u.x(), pixel_delta_u.y(), pixel_delta_u.z(),
       pixel_delta_v.x(), pixel_delta_v.y(), pixel_delta_v.z(),
       samples, max_depth
   );
   
   auto end_time = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
   
   cout << "Rendering completed in " << duration.count() << " ms" << endl;
   cout.imbue(locale("en_US.UTF-8"));
   cout << "Rays traced: " << fixed << ray_count << endl;
   
   // Save image with descriptive filename
   string filename = "res/new_scene_" + to_string(width) + "x" + to_string(height) + "_s" + to_string(samples) + ".png";
   writeImage(image, width, height, filename);
   
   // Cleanup
   CudaSceneBuilder::freeGPUScene(gpu_scene);
   cout << "GPU scene freed" << endl;
   
   cout << "\nTest completed successfully!" << endl;
   cout << "Compare this image with the legacy CUDA renderer output.\n" << endl;
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

   // Calculate width maintaining aspect ratio (16:9)
   int image_height = args.height;
   int image_width = (image_height * 16) / 9;

   Camera c(Vec3(0, 0, 0), image_width, image_height, CHANNELS, args.samples);

   vector<unsigned char> image(c.image_width * c.image_height * CHANNELS);

   cout << endl;
   cout << "======================================================" << endl;
   cout << " 302 Ray tracer project v" << ver_major << " -- P.-A. Mudry, ISC 2026" << endl;
   cout << "======================================================" << endl << endl;
   cout << "Rendering at resolution: " << c.image_width << " x " << c.image_height << " pixels" << endl;
   cout << "Samples per pixel: " << args.samples << endl << endl;

   RndGen::set_seed(123);

   // === OLD SCENE API (will be deprecated) ===
   // scene s = many_spheres();
   scene scene = demo_scene();

   // === NEW SCENE API (unified CPU/GPU format) ===
   // Uncomment to test new scene description system:
   // Scene::SceneDescription scene_desc = create_scene_description();
   // scene scene = Scene::CPUSceneBuilder::buildCPUScene(scene_desc);

   vector<unsigned char> localImage(image.size());

   // Choose rendering method
   cout << "Choose rendering method:" << endl;
   cout << "\t0. CPU sequential" << endl;
   cout << "\t1. CPU parallel" << endl;
   cout << "\t2. CUDA GPU (default)" << endl;
#ifdef SDL2_FOUND
   cout << "\t3. CUDA GPU with interactive SDL display" << endl;
#endif
   cout << "\t4. TEST: New scene-based CUDA rendering (Phase 3)" << endl;
   cout << "Enter choice (0, 1, 2"
#ifdef SDL2_FOUND
        << ", 3"
#endif
        << ", or 4): ";

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
      c.renderPixels(scene, localImage);
      break;
   case 1:
      cout << "Using CPU parallel rendering..." << endl;
      c.renderPixelsParallel(scene, localImage);
      break;
#ifdef SDL2_FOUND
   case 3:
      cout << "Using CUDA GPU with interactive SDL display..." << endl;
      c.renderPixelsSDLContinuous(localImage, args.start_samples, args.auto_accumulate);
      break;
#endif
   case 4:
      test_new_scene_cuda_rendering(image_width, image_height, args.samples);
      return 0;  // Exit after test
   default:
      cout << "Using CUDA GPU rendering..." << endl;
      c.renderPixelsCUDA(localImage);
      break;
   }

   // Create res directory if it doesn't exist
   dumpImageToFile(localImage, c.image_width, c.image_height, "res/output.png");

   cout.imbue(locale("en_US.UTF-8"));
   cout << "Rays traced: " << fixed << c.n_rays << endl;

   return 0;
}