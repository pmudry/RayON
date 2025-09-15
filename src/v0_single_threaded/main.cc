#include "utils.h"
#include "hittable_list.h"
#include "cube.h"
#include <iostream>

#include <thread>
#include <future>
#include <vector>
#include <filesystem>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"

// Function to write image buffer to PNG file
void writeImage(const vector<unsigned char> &image, int image_width, int image_height, const std::string &filename)
{
    const int channels = 3; // RGB
    if (stbi_write_png(filename.c_str(), image_width, image_height, channels, image.data(), image_width * channels))
    {
        std::cout << "Image saved successfully to " << filename << std::endl;
    }
    else
    {
        std::cerr << "Failed to save image to " << filename << std::endl;
    }
}

// [[deprecated("Use the other setPixel function instead")]]
// inline void setPixel(vector<unsigned char> &viewPort, int x, int y, unsigned char r, unsigned char g, unsigned char b)
// {
//     int index = (y * image_width + x) * channels;
//     viewPort[index] = r;
//     viewPort[index + 1] = g;
//     viewPort[index + 2] = b;
// }

/**
 * @brief Calculates the intersection point of a ray with a sphere
 *
 * This function determines if and where a ray intersects with a sphere by solving
 * the quadratic equation formed by substituting the ray equation into the sphere equation.
 *
 * **Mathematical Background:**
 * - Sphere equation: `(C−P)⋅(C−P) = r²` where P is a point on the sphere
 * - Ray equation: `P = O + t*d` where O is origin, d is direction, t is distance
 * - Substitution yields quadratic: `at² + bt + c = 0` where:
 *   - `a = d⋅d` (direction vector dot product)
 *   - `b = −2*d⋅(C−O)` (relates direction to center-origin vector)
 *   - `c = (C−O)⋅(C−O) − r²` (distance from origin to center minus radius squared)
 *
 * @param center The center point of the sphere in 3D space
 * @param radius The radius of the sphere (must be positive)
 * @param r The ray to test for intersection
 *
 * @return The parameter t for the intersection point along the ray:
 *         - Returns `-1.0` if no intersection occurs (discriminant < 0)
 *         - Returns the farther intersection point when two intersections exist
 *         - The actual intersection point can be computed as `r.origin() + t * r.direction()`
 *
 * @note When the discriminant is non-negative, this function returns the larger t value,
 *       corresponding to the exit point of the ray from the sphere
 */
// [[deprecated("Use the hit_sphere_simplified function instead")]]
// double hit_sphere(const point3 &center, double radius, const ray &r)
// {
//     auto a = r.direction().length_squared(); // Which is like r.dir · r.dir = ||r.dir||^2
//     auto b = -2.0 * dot(r.direction(), center - r.origin());
//     auto c = (center - r.origin()).length_squared() - radius * radius;
//     auto discriminant = b * b - 4 * a * c;

//     if (discriminant < 0)
//     {
//         return -1.0;
//     }
//     else
//     {
//         return (-b - sqrt(discriminant)) / (2.0 * a); // We return the closest intersection
//     }
// }

// [[deprecated("Use the hit function in sphere instead")]]
// double hit_sphere_simplified(const point3 &center, double radius, const ray &r)
// {
//     auto oc = center - r.origin();
//     auto a = r.direction().length_squared(); // Which is like r.dir · r.dir = ||r.dir||^2
//     auto h = dot(r.direction(), oc);
//     auto c = (oc).length_squared() - radius * radius;
//     auto discriminant = h * h - a * c;

//     if (discriminant < 0)
//     {
//         return -1.0;
//     }
//     else
//     {
//         return (h - sqrt(discriminant)) / a; // We return the closest intersection
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

int image_width = 0;
int image_height = 720;
const int channels = 3; // RGB

/**
 * Just for the sake of putting a gradient in a file
 */
void fillGradientImage(vector<unsigned char> &image)
{
    // Generate a simple gradient
    for (int y = 0; y < image_height; ++y)
    {
        for (int x = 0; x < image_width; ++x)
        {
            int index = (y * image_width + x) * channels;
            image[index] = static_cast<unsigned char>(255.0 * y / image_width);      // Red
            image[index + 1] = static_cast<unsigned char>(255.0 * x / image_height); // Green
            image[index + 2] = 100;                                                  // Blue
        }
    }
}

void dumpImageToFile(vector<unsigned char> &image, string name)
{
    // Write image to file
    writeImage(image, image_width, image_height, name);
}

using scene = hittable_list;

scene many_spheres()
{
    scene s;

    s.add(make_shared<sphere>(point3(0, -1000.5, -1), 1000));
    s.add(make_shared<sphere>(point3(0, 0, 0), .5));
    
    auto leftSphere = make_shared<sphere>(point3(-1, 1, -1), .8);
    leftSphere.get()->isMirror = true;
    s.add(leftSphere);
    
    for(int i = 0; i < 180; i++){
        auto obj = make_shared<sphere>(
            point3(
                RndGen::random_normal(-2, 2), 
                RndGen::random_normal(-0.5, 0.4),
                RndGen::random_normal(-0.5, -4)),
                .1 + RndGen::random_double(0, 0.2));
        
        if(RndGen::random_double() < 0.3){
            obj.get()-> isMirror = true;
        }

        s.add(obj);
    }
    
    return s;
}

scene single_cube()
{
    scene s;
    
    s.add(make_shared<sphere>(point3(0, -1000.5, -1), 1000));
    s.add(make_shared<sphere>(point3(-1, 1, -1), .5));
    auto rotatedCube = make_shared<cube>(point3(0, 0, -1), 1, vec3(0, 45, 0));
    s.add(rotatedCube);
    return s;
}

int main()
{
    const int samples_per_pixel = 1024;

    Camera c(vec3(0, 0, 0), 1080, channels, samples_per_pixel);

    image_width = c.image_width;
    image_height = c.image_height;

    vector<unsigned char> image(c.image_width * c.image_height * channels);

    std::cout << "Rendering at resolution: " << c.image_width << " x " << c.image_height << " pixels" << std::endl;

    // Create a new scene for this frame
    int i = 0;

    RndGen::set_seed(123);

    scene s = many_spheres();
    //scene s = single_cube();
    
    // frameScene.add(rotatedCube);
    vector<unsigned char> localImage(image.size());
    
    // Choose rendering method
    std::cout << "Choose rendering method:" << std::endl;
    std::cout << "1. CPU Parallel" << std::endl;
    std::cout << "2. CUDA GPU" << std::endl;
    std::cout << "Enter choice (1 or 2): ";
    
    // int choice;
    // std::cin >> choice;
    
        std::cout << "Using CUDA GPU rendering..." << std::endl;
        c.renderPixelsCUDA(localImage);

    // if (choice == 2) {
    //     std::cout << "Using CUDA GPU rendering..." << std::endl;
    //     c.renderPixelsCUDA(localImage);
    // } else {
    //     std::cout << "Using CPU parallel rendering..." << std::endl;
    //     c.renderPixelsParallelWithTiming(s, localImage);
    // }
    
    // Create res directory if it doesn't exist    
    dumpImageToFile(localImage, "res/output" + to_string(i) + ".png");

    // std::function<void(int)> renderFrame = [&](int i)
    // {
    //     // Create a new scene for this frame
    //     hittable_list frameScene;
    //     frameScene.add(make_shared<sphere>(point3(0, -100.5, -1), 100));
    //     auto rotatedCube = make_shared<cube>(point3(0, 0, -1), 1, vec3(0, 45 + i * 3, 0));
    //     frameScene.add(rotatedCube);
    //     vector<unsigned char> localImage(image.size());
    //     c.renderPixels(frameScene, localImage);
    //     dumpImageToFile(localImage, "res/output" + to_string(i) + ".png");
    // };

    // const int numFrames = 120;
    // const int numThreads = std::thread::hardware_concurrency();
    // std::vector<std::future<void>> futures;

    // for (int i = 0; i < numFrames; ++i)
    // {
    //     futures.push_back(std::async(std::launch::async, renderFrame, i));
    //     if (futures.size() >= numThreads)
    //     {
    //         for (auto &f : futures)
    //         {
    //             f.get(); // Wait for all threads to finish
    //         }
    //         futures.clear();
    //     }
    // }

    // // Wait for any remaining threads
    // for (auto &f : futures)
    // {
    //     f.get();
    // }

    //FIXME: does not work for very large numbers
    cout.imbue(std::locale("en_US.UTF-8"));
    cout << "Rays traced: " << std::fixed << c.n_rays << endl;

    return 0;
}