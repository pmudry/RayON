#pragma once

#include <string>

namespace constants
{
    const std::string ver_major = "1";
    const std::string ver_minor = "5";
    const std::string ver_patch = "0";
    const std::string version = ver_major + "." + ver_minor + "." + ver_patch;

    // Image specifics settings
    const double ASPECT_RATIO = 16.0 / 9.0;
    const int SET_IMAGE_HEIGHT = 720;
    const int IMAGE_HEIGHT = SET_IMAGE_HEIGHT;
    const int IMAGE_WIDTH = (int)(ASPECT_RATIO * SET_IMAGE_HEIGHT);
    const int CHANNELS = 3; // RGB

    // Renderer specific settings
    const int SAMPLES_PER_PIXEL = 64; // Number of samples per pixel for anti-aliasing
    const int MAX_DEPTH = 16;         // Maximum recursion depth for ray tracing
}; 