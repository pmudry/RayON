#include "utils.h"
#include "hittable_list.h"
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"

// Image dimensions
const auto aspect_ratio = 16.0 / 9.0;
const int image_height = 360;
const int image_width = static_cast<int>(image_height * aspect_ratio);
const int channels = 3; // RGB

// Camera setting, where the viewport is what the camera sees
auto focal_length = 1.0;
auto viewport_height = 2;
auto viewport_width = (static_cast<double>(image_width) / image_height) * viewport_height;
auto camera_center = point3(0, 0, 0);

// Calculate the vectors across the horizontal and down the vertical viewport edges.
auto viewport_u = vec3(viewport_width, 0, 0);
auto viewport_v = vec3(0, -viewport_height, 0);

// Calculate the horizontal and vertical delta vectors from pixel to pixel.
auto pixel_delta_u = viewport_u / image_width;
auto pixel_delta_v = viewport_v / image_height;

// Calculate the location of the upper left pixel.
auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
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



[[deprecated("Use the other setPixel function instead")]]
inline void setPixel(vector<unsigned char> &viewPort, int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
    int index = (y * image_width + x) * channels;
    viewPort[index] = r;
    viewPort[index + 1] = g;
    viewPort[index + 2] = b;
}

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
[[deprecated("Use the hit_sphere_simplified function instead")]]
double hit_sphere(const point3 &center, double radius, const ray &r)
{
    auto a = r.direction().length_squared(); // Which is like r.dir · r.dir = ||r.dir||^2
    auto b = -2.0 * dot(r.direction(), center - r.origin());
    auto c = (center - r.origin()).length_squared() - radius * radius;
    auto discriminant = b * b - 4 * a * c;

    if (discriminant < 0)
    {
        return -1.0;
    }
    else
    {
        return (-b - sqrt(discriminant)) / (2.0 * a); // We return the closest intersection
    }
}

[[deprecated("Use the hit function in sphere instead")]]
double hit_sphere_simplified(const point3 &center, double radius, const ray &r)
{
    auto oc = center - r.origin();
    auto a = r.direction().length_squared(); // Which is like r.dir · r.dir = ||r.dir||^2
    auto h = dot(r.direction(), oc);
    auto c = (oc).length_squared() - radius * radius;
    auto discriminant = h * h - a * c;

    if (discriminant < 0)
    {
        return -1.0;
    }
    else
    {
        return (h - sqrt(discriminant)) / a; // We return the closest intersection
    }
}



[[deprecated("Use the ray_color function with hittable instead")]]
inline color ray_color_v0(const ray &r)
{
    vec3 unit_direction = unit_vector(r.direction());
    auto sphere_center = point3(0, 0, -1);

    auto t = hit_sphere_simplified(sphere_center, 0.5, r);

    if (t > 0.0)
    {
        // Normal is the vector from the sphere center to the hit point
        vec3 normal = unit_vector(r.at(t) - sphere_center);
        return 0.5 * color(normal.x() + 1, normal.y() + 1, normal.z() + 1);
    }

    // Le vecteur unit_direction variera entre -1 et +1 en x et y
    // A blue to white gradient background
    double q = 0.5 * (unit_direction.x() + 1.0);
    return (1.0 - q) * color(1.0, 1.0, 1.0) + q * color(0.5, 0.7, 1.0);
}

// void renderPixels_v0(std::vector<unsigned char> &image)
// {
//     // All our hittable objects
//     hittable_list scene;

//     scene.add(make_shared<sphere>(point3(0, 0, -10), .9));
//     scene.add(make_shared<sphere>(point3(0, -100.5, -0), 100));

//     for (int y = 0; y < image_height; ++y)
//     {
//         for (int x = 0; x < image_width; ++x)
//         {
//             // Progress indicator
//             if (x == 0)
//             {
//                 showProgress(y, image_height);
//             }
//             // Calculate the direction of the ray for the current pixel
//             vec3 pixel_center = pixel00_loc + x * pixel_delta_u + y * pixel_delta_v;
//             // cout << "Pixel[" << x << "," << y << "] at " << pixel_center << endl;
//             vec3 ray_direction = pixel_center - camera_center;

//             // Create a ray from the camera center through the pixel
//             ray r(camera_center, unit_vector(ray_direction));

//             color pixel_color(ray_color(r, scene));
//             setPixel(image, x, y, pixel_color);
//         }
//     }
//     showProgress(image_height - 1, image_height);
//     cout << endl;
// }

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
    vector<unsigned char> image(image_width * image_height * channels);
    Camera c(vec3(0,0,0), channels);
    hittable_list scene;
    scene.add(make_shared<sphere>(point3(0, 0, -1), .5));
    scene.add(make_shared<sphere>(point3(0, -100.5, -1), 100));
    c.renderPixels(scene, image);

    std::cout << "🖼️ resolution: " << image_width << " x " << image_height << " pixels" << std::endl;

    dumpImageToFile(image, "output.png");

    cout.imbue(std::locale("en_US.UTF-8"));
    cout << "Rays traced: " << std::fixed << c.n_rays << endl;

    return 0;
}