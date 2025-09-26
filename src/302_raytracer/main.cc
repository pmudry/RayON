#include "camera.h"
#include "constants.h"
#include "cube.h"
#include "hittable_list.h"
#include "sphere.h"

#include <iostream>

#include <memory>

#include <filesystem>
#include <future>
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

// [[deprecated("Use the other setPixel function instead")]]
// inline void setPixel(vector<unsigned char> &viewPort, int x, int y, unsigned
// char r, unsigned char g, unsigned char b)
// {
//     int index = (y * image_width + x) * channels;
//     viewPort[index] = r;
//     viewPort[index + 1] = g;
//     viewPort[index + 2] = b;
// }

/**
 * @brief Calculates the intersection point of a ray with a sphere
 *
 * This function determines if and where a ray intersects with a sphere by
 * solving the quadratic equation formed by substituting the ray equation into
 * the sphere equation.
 *
 * **Mathematical Background:**
 * - Sphere equation: `(C−P)⋅(C−P) = r²` where P is a point on the sphere
 * - Ray equation: `P = O + t*d` where O is origin, d is direction, t is
 * distance
 * - Substitution yields quadratic: `at² + bt + c = 0` where:
 *   - `a = d⋅d` (direction vector dot product)
 *   - `b = −2*d⋅(C−O)` (relates direction to center-origin vector)
 *   - `c = (C−O)⋅(C−O) − r²` (distance from origin to center minus radius
 * squared)
 *
 * @param center The center point of the sphere in 3D space
 * @param radius The radius of the sphere (must be positive)
 * @param r The ray to test for intersection
 *
 * @return The parameter t for the intersection point along the ray:
 *         - Returns `-1.0` if no intersection occurs (discriminant < 0)
 *         - Returns the farther intersection point when two intersections exist
 *         - The actual intersection point can be computed as `r.origin() + t *
 * r.direction()`
 *
 * @note When the discriminant is non-negative, this function returns the larger
 * t value, corresponding to the exit point of the ray from the sphere
 */
// [[deprecated("Use the hit_sphere_simplified function instead")]]
// double hit_sphere(const point3 &center, double radius, const ray &r)
// {
//     auto a = r.direction().length_squared(); // Which is like r.dir · r.dir =
//     ||r.dir||^2 auto b = -2.0 * dot(r.direction(), center - r.origin()); auto
//     c = (center - r.origin()).length_squared() - radius * radius; auto
//     discriminant = b * b - 4 * a * c;

//     if (discriminant < 0)
//     {
//         return -1.0;
//     }
//     else
//     {
//         return (-b - sqrt(discriminant)) / (2.0 * a); // We return the
//         closest intersection
//     }
// }

// [[deprecated("Use the hit function in sphere instead")]]
// double hit_sphere_simplified(const point3 &center, double radius, const ray
// &r)
// {
//     auto oc = center - r.origin();
//     auto a = r.direction().length_squared(); // Which is like r.dir · r.dir =
//     ||r.dir||^2 auto h = dot(r.direction(), oc); auto c =
//     (oc).length_squared() - radius * radius; auto discriminant = h * h - a *
//     c;

//     if (discriminant < 0)
//     {
//         return -1.0;
//     }
//     else
//     {
//         return (h - sqrt(discriminant)) / a; // We return the closest
//         intersection
//     }
// }

// [[deprecated("Use the ray_color function with hittable instead")]]
// inline color ray_color_v0(const ray &r)
// {
//     vec3 unit_direction = unit_vector(r.direction());
//     auto sphere_center = point3(0, 0, -1);

//     auto t = hit_sphere_simplified(sphere_center, 0.5, r);

//     if (t > 0.0)
//     {
//         // Normal is the vector from the sphere center to the hit point
//         vec3 normal = unit_vector(r.at(t) - sphere_center);
//         return 0.5 * color(normal.x() + 1, normal.y() + 1, normal.z() + 1);
//     }

//     // Le vecteur unit_direction variera entre -1 et +1 en x et y
//     // A blue to white gradient background
//     double q = 0.5 * (unit_direction.x() + 1.0);
//     return (1.0 - q) * color(1.0, 1.0, 1.0) + q * color(0.5, 0.7, 1.0);
// }

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

   cout << endl << "=====================================================" << endl;
   cout << " 302 Ray tracer project v" << ver_major << " -- P.-A. Mudry, ISC 2026" << endl;
   cout << "=====================================================" << endl << endl;
   cout << "Rendering at resolution: " << c.image_width << " x " << c.image_height << " pixels" << endl;
   cout << "Samples per pixel: " << samples << endl << endl;

   // Create a new scene for this frame
   int i = 0;
   RndGen::set_seed(123);

   // scene s = many_spheres();
   scene s = demo_scene();

   vector<unsigned char> localImage(image.size());

   // Choose rendering method
   cout << "Choose rendering method:" << endl;
   cout << "\t1. CPU Parallel" << endl;
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

   if (choice == 1)
   {
      cout << "Using CPU parallel rendering..." << endl;
      c.renderPixelsParallelWithTiming(s, localImage);
   }
   else if (choice == 3)
   {
#ifdef SDL2_FOUND
      cout << "Using CUDA GPU rendering with real-time display..." << endl;
      c.renderPixelsCUDART(localImage);
#else
      cout << "SDL2 not found. Falling back to standard CUDA rendering..." << endl;
      c.renderPixelsCUDA(localImage);
#endif
   }
   else
   {
      cout << "Using CUDA GPU rendering..." << endl;
      c.renderPixelsCUDA(localImage);
   }
   // Create res directory if it doesn't exist
   dumpImageToFile(localImage, "res/output" + to_string(i) + ".png");

   cout.imbue(locale("en_US.UTF-8"));
   cout << "Rays traced: " << fixed << c.n_rays << endl;

   return 0;
}
