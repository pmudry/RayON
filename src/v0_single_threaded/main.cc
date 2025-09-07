#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"

#include "vec3.h"

// Image dimensions
const int image_width = 512;
const int image_height = 512;
const int channels = 3; // RGB

// Function to write image buffer to PNG file
void writeImage(const std::vector<unsigned char> &image, int image_width, int image_height, const std::string &filename)
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

/**
 * Just for the sake of putting a gradient in a file
 */
void fillGradientImage(vector<unsigned char> &image){
    // Generate a simple gradient
    for (int y = 0; y < image_height; ++y)
    {
        for (int x = 0; x < image_width; ++x)
        {
            int index = (y * image_width + x) * channels;
            image[index] = static_cast<unsigned char>(255.0 * y / image_width);      // Red
            image[index + 1] = static_cast<unsigned char>(255.0 * x / image_height); // Green
            image[index + 2] = 100;                                            // Blue
        }
    } 
}

void image2Disk(vector<unsigned char> &image, string name){
    // Generate a simple gradient
    for (int y = 0; y < image_height; ++y)
    {
        for (int x = 0; x < image_width; ++x)
        {
            int index = (y * image_width + x) * channels;
            image[index] = static_cast<unsigned char>(255.0 * y / image_width);      // Red
            image[index + 1] = static_cast<unsigned char>(255.0 * x / image_height); // Green
            image[index + 2] = 100;                                            // Blue
        }
    }

    // Write image to file
    writeImage(image, image_width, image_height, name);
}   

int main(){ 
    std::vector<unsigned char> image(image_width * image_height * channels);
    fillGradientImage(image);
    
    image2Disk(image, "output.png");

    // Just some vec3 tests 
    auto v1 = vec3(1,2,3);
    auto v2 = vec3(2,2,3);
    auto v3 = v1 + v2;
    vec3 v4 = cross(v1, v2);

    clog << v3 << endl;
    clog << v4 << endl;

    return 0;
}