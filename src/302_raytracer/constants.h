#pragma once

#include <string>

namespace constants
{
    const std::string ver_major = "1.0";

    const double ASPECT_RATIO = 16.0 / 9.0;

    const int SET_IMAGE_HEIGHT = 1080;

    const int IMAGE_HEIGHT = SET_IMAGE_HEIGHT;
    const int IMAGE_WIDTH = (int)(ASPECT_RATIO * SET_IMAGE_HEIGHT);
    const int CHANNELS = 3; // RGB

    // Renderer specific settings
    const int SAMPLES_PER_PIXEL = 16; // Number of samples per pixel for anti-aliasing
    int MAX_DEPTH = 8;               // Maximum recursion depth for ray tracing
}; 