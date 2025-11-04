#include "constants.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"

#include <algorithm>
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
   
   s.add(make_shared<Sphere>(Point3(0, -950.5, -1), 950,    material_uniform_red)); // Ground   
   s.add(make_shared<Sphere>(Point3(-3.5, 0.45, -1.8), .8,  material_uniform_blue));
   s.add(make_shared<Sphere>(Point3(-1.3, 0.18, -5), .7,    material_uniform_blue));
   s.add(make_shared<Sphere>(Point3(-.7, .2, -.3), .6,      material_uniform_blue));
   s.add(make_shared<Sphere>(Point3(1.2, 0, -2), 0.5,       material_uniform_blue));

   // Small "ISC" spheres at the bottom
   for (int i = 0; i < 5; i++)
   {
      s.add(make_shared<Sphere>(Point3(-3.5 + i * 0.5, -0.3, 1.2), 0.2, material_lambert));
   }

   return s;
}

struct ProgramArgs {
   int samples = SAMPLES_PER_PIXEL;
   int height = IMAGE_HEIGHT;
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
         cout << "  -h, --help, /?  Show this help message\n";
         cout << "  -s <samples>    Set the number of samples per pixel (default: " << SAMPLES_PER_PIXEL << ")\n";
         cout << "  -r <height>     Set vertical resolution (allowed: 2160, 1080, 720, 360, 180, default: " << IMAGE_HEIGHT << ")\n";
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
      else if (argv[i][0] == '-')
      {
         cerr << "Unknown argument: " << argv[i] << "\n";
         cout << "Usage: " << argv[0] << " [options]\n";
         cout << "Options:\n";
         cout << "  -h, --help, /?  Show this help message\n";
         cout << "  -s <samples>    Set the number of samples per pixel (default: " << SAMPLES_PER_PIXEL << ")\n";
         cout << "  -r <height>     Set vertical resolution (allowed: 2160, 1080, 720, 360, 180, default: " << IMAGE_HEIGHT << ")\n";
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
   cout << "=====================================================" << endl;
   cout << " 302 Ray tracer project v" << ver_major << " -- P.-A. Mudry, ISC 2026" << endl;
   cout << "=====================================================" << endl << endl;
   cout << "Rendering at resolution: " << c.image_width << " x " << c.image_height << " pixels" << endl;
   cout << "Samples per pixel: " << args.samples << endl << endl;

   RndGen::set_seed(123);

   // scene s = many_spheres();
   scene scene = demo_scene();

   vector<unsigned char> localImage(image.size());

   // Choose rendering method
   cout << "Choose rendering method:" << endl;
   cout << "\t0. CPU sequential" << endl;
   cout << "\t1. CPU parallel" << endl;
   cout << "\t2. CUDA GPU (default)" << endl;
   cout << "Enter choice (0, 1, or 2): ";

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