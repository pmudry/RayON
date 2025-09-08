#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"

#include "vec3.h"
#include "ray.h"
#include "color.h"

// Image dimensions
const auto aspect_ratio = 16.0 / 9.0;
const int image_height = 360;
const int image_width = static_cast<int>(image_height * aspect_ratio);
const int channels = 3; // RGB

// Camera setting, where the viewport is what the camera sees
auto focal_length = 1.0;
auto viewport_height = 2.0;
auto viewport_width = (static_cast<double>(image_width) / image_height) * viewport_height;
auto camera_center = point3(0, 0, 0);

// Calculate the vectors across the horizontal and down the vertical viewport edges.
auto viewport_u = vec3(viewport_width, 0, 0);
auto viewport_v = vec3(0, -viewport_height, 0);

// Calculate the horizontal and vertical delta vectors from pixel to pixel.
auto pixel_delta_u = viewport_u / image_width;
auto pixel_delta_v = viewport_v / image_height;

// Calculate the location of the upper left pixel.
auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

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

inline void setPixel(vector<unsigned char> &viewPort, int x, int y, color& c){    
    int index = (y * image_width + x) * channels;
    viewPort[index + 0] = static_cast<int>(c.x() * 255.999);
    viewPort[index + 1] = static_cast<int>(c.y() * 255.999);
    viewPort[index + 2] = static_cast<int>(c.z() * 255.999);
}

inline void setPixel(vector<unsigned char> &viewPort, int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
    int index = (y * image_width + x) * channels;
    viewPort[index] = r;
    viewPort[index + 1] = g;
    viewPort[index + 2] = b;
}

color ray_color(const ray& r)
{
    vec3 unit_direction = unit_vector(r.direction());
    // cout << "r: " << r.direction() << ", unit: " << unit_direction << endl;

    // Le vecteur unit_direction variera entre -1 et +1 en x et y

    // A blue to white gradient background
    double t = 0.5 * (unit_direction.y() + 1.0);    
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);    
}

void renderPixels(std::vector<unsigned char> &image)
{
    for (int y = 0; y < image_height; ++y)
    {
        for (int x = 0; x < image_width; ++x)
        {
            // Calculate the direction of the ray for the current pixel
            vec3 pixel_center = pixel00_loc + x * pixel_delta_u + y * pixel_delta_v;
            // cout << "Pixel[" << x << "," << y << "] at " << pixel_center << endl;
            vec3 ray_direction = pixel_center - camera_center;
            
            // Create a ray from the camera center through the pixel
            ray r(camera_center, ray_direction);

            color pixel_color(ray_color(r));
            setPixel(image, x, y, pixel_color);
        }
    }
}

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

int main()
{
    std::vector<unsigned char> image(image_width * image_height * channels);
    renderPixels(image);

    dumpImageToFile(image, "output.png");

    // Just some vec3 tests
    auto v1 = vec3(1, 2, 3);
    auto v2 = vec3(2, 2, 3);
    auto v3 = v1 + v2;
    vec3 v4 = cross(v1, v2);

    clog << v3 << endl;
    clog << v4 << endl;

    return 0;
}