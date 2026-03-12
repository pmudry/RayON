#pragma once

#include <string>

namespace constants
{
    const std::string ver_major = "1";
    const std::string ver_minor = "5";
    const std::string ver_patch = "2";
    const std::string version = ver_major + "." + ver_minor + "." + ver_patch;

    // Image specifics settings
    const double ASPECT_RATIO = 16.0 / 9.0;
    const int SET_IMAGE_HEIGHT = 720;
    const int IMAGE_HEIGHT = SET_IMAGE_HEIGHT;
    const int IMAGE_WIDTH = (int)(ASPECT_RATIO * SET_IMAGE_HEIGHT);
    const int CHANNELS = 3; // RGB

    // Default renderer settings
    const int SAMPLES_PER_PIXEL = 64; // Default samples per pixel for path tracing
    const int MAX_DEPTH = 16;         // Maximum ray bounce depth

    // Interactive renderer defaults (mode 3)
    const int INTERACTIVE_SAMPLES_PER_BATCH = 50;  // Samples per batch when camera is still
    const int INTERACTIVE_MOTION_SAMPLES = 10;     // Samples per batch while camera is moving
    const int INTERACTIVE_MAX_SPP = 50000;         // Max SPP budget for interactive accumulation
};