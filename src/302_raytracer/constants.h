#pragma once

#include <string>

namespace constants
{
    const char* const ver_major = "1.1";

    const double ASPECT_RATIO = 16.0 / 9.0;

    const int SET_IMAGE_HEIGHT = 720;

    const int IMAGE_HEIGHT = SET_IMAGE_HEIGHT;
    const int IMAGE_WIDTH = (int)(ASPECT_RATIO * SET_IMAGE_HEIGHT);
    const int CHANNELS = 3; // RGB

    // Renderer specific settings
    const int SAMPLES_PER_PIXEL = 32; // Number of samples per pixel for anti-aliasing
    int MAX_DEPTH = 16;               // Maximum recursion depth for ray tracing
}; 