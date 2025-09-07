#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"

// Image dimensions
const int width = 512;
const int height = 512;
const int channels = 3; // RGB

// Function to write image buffer to PNG file
void writeImage(const std::vector<unsigned char> &image, int width, int height, const std::string &filename)
{
    const int channels = 3; // RGB
    if (stbi_write_png(filename.c_str(), width, height, channels, image.data(), width * channels))
    {
        std::cout << "Image saved successfully to " << filename << std::endl;
    }
    else
    {
        std::cerr << "Failed to save image to " << filename << std::endl;
    }
}

void writeGradientImage(vector<unsigned char> &image)
{
    // Generate a simple gradient
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int index = (y * width + x) * channels;
            image[index] = static_cast<unsigned char>(255.0 * y / width);      // Red
            image[index + 1] = static_cast<unsigned char>(255.0 * x / height); // Green
            image[index + 2] = 100;                                            // Blue
        }
    }

    // Write image to file
    writeImage(image, width, height, "output.png");
}

int main()
{ 
    // Testing image output
    std::vector<unsigned char> image(width * height * channels);
    writeGradientImage(image);

    return 0;
}