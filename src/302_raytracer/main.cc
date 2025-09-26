#include "camera.h"
#include "constants.h"
#include "cube.h"
#include "hittable_list.h"
#include "sphere.h"

#include <filesystem>
#include <future>
#include <iostream>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stb_image_write.h"

using namespace constants;

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

void dumpImageToFile(vector<unsigned char> &image, string name)
{
   // Write image to file
   writeImage(image, IMAGE_WIDTH, IMAGE_HEIGHT, name);
}

using scene = Hittable_list;

scene demo_scene()
{
   scene s;

   s.add(make_shared<Sphere>(Point3(0, -950.5, -1), 950)); // Ground
   s.add(make_shared<Sphere>(Point3(-3.5, 0.45, -1.8), .8));
   s.add(make_shared<Sphere>(Point3(-1.3, 0.18, -5), .7));
   s.add(make_shared<Sphere>(Point3(-.7, .2, -.3), .6));
   s.add(make_shared<Sphere>(Point3(1.2, 0, -2), 0.5));

   // Small "ISC" spheres at the bottom
   for (int i = 0; i < 5; i++)
   {
      s.add(make_shared<Sphere>(Point3(-3.5 + i * 0.5, -0.3, 1.2), 0.2));
   }

   return s;
}

scene single_cube()
{
   scene s;

   s.add(make_shared<Sphere>(Point3(0, -1000.5, -1), 1000));
   s.add(make_shared<Sphere>(Point3(-1, 1, -1), .5));
   auto rotatedCube = make_shared<Cube>(Point3(0, 0, -1), 1, Vec3(0, 45, 0));
   s.add(rotatedCube);
   return s;
}

int parseInput(int argc, char *argv[])
{
   // Parse command-line arguments
   for (int i = 1; i < argc; ++i)
   {
      if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "/?") == 0)
      {
         cout << "Usage: " << argv[0] << " [options]\n";
         cout << "Options:\n";
         cout << "  -h, --help, /?  Show this help message\n";
         cout << "  -s <samples>    Set the number of samples per pixel (default: " << SAMPLES_PER_PIXEL << ")\n";
         return -1;
      }
      else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc)
      {
         return atoi(argv[++i]);
      }
      else if (argv[i][0] == '-')
      {
         cerr << "Unknown argument: " << argv[i] << "\n";
         cout << "Usage: " << argv[0] << " [options]\n";
         cout << "Options:\n";
         cout << "  -h, --help, /?  Show this help message\n";
         cout << "  -s <samples>    Set the number of samples per pixel (default: " << SAMPLES_PER_PIXEL << ")\n";
         return -1;
      }
      else
      {
         cerr << "Unexpected argument: " << argv[i] << "\n";
         return -1;
      }
   }

   return SAMPLES_PER_PIXEL;
}

int main(int argc, char *argv[])
{
   int samples = parseInput(argc, argv);

   Camera c(Vec3(0, 0, 0), IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, samples);

   vector<unsigned char> image(c.image_width * c.image_height * CHANNELS);

   cout << endl;
   cout << "=====================================================" << endl;
   cout << " 302 Ray tracer project v" << ver_major << " -- P.-A. Mudry, ISC 2026" << endl;
   cout << "=====================================================" << endl << endl;
   cout << "Rendering at resolution: " << c.image_width << " x " << c.image_height << " pixels" << endl;
   cout << "Samples per pixel: " << samples << endl << endl;

   RndGen::set_seed(123);

   // scene s = many_spheres();
   scene scene = demo_scene();

   vector<unsigned char> localImage(image.size());

   // Choose rendering method
   cout << "Choose rendering method:" << endl;
   cout << "\t0. CPU sequential" << endl;
   cout << "\t1. CPU parallel" << endl;
   cout << "\t2. CUDA GPU (default)" << endl;
   cout << "\t3. CUDA GPU with Real-time Display" << endl;
   cout << "Enter choice (1, 2, or 3): ";

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
      c.renderPixelsParallelWithTiming(scene, localImage);
      break;
   case 3:
#ifdef SDL2_FOUND
      cout << "Using CUDA GPU rendering with real-time display..." << endl;
      c.renderPixelsCUDART(localImage);
#else
      cout << "SDL2 not found. Falling back to standard CUDA rendering..." << endl;
      c.renderPixelsCUDA(localImage);
#endif
      break;
   default:
      cout << "Using CUDA GPU rendering..." << endl;
      c.renderPixelsCUDA(localImage);
      break;
   }

   // Create res directory if it doesn't exist
   dumpImageToFile(localImage, "res/output.png");

   cout.imbue(locale("en_US.UTF-8"));
   cout << "Rays traced: " << fixed << c.n_rays << endl;

   return 0;
}