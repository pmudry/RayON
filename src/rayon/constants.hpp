#pragma once

#include <string>

namespace constants
{
    inline const std::string ver_major = "1";
    inline const std::string ver_minor = "6";
    inline const std::string ver_patch = "1";
    inline const std::string version = ver_major + "." + ver_minor + "." + ver_patch;

    // Image specifics settings
    inline constexpr double ASPECT_RATIO = 16.0 / 9.0;
    inline constexpr int IMAGE_HEIGHT = 720;
    inline constexpr int IMAGE_WIDTH = static_cast<int>(ASPECT_RATIO * IMAGE_HEIGHT);
    inline constexpr int CHANNELS = 3; // RGB

    // Default renderer settings
    inline constexpr int SAMPLES_PER_PIXEL = 64; // Default samples per pixel for path tracing
    inline constexpr int MAX_DEPTH = 16;          // Maximum ray bounce depth

    // Interactive renderer defaults (mode 3)
    inline constexpr int INTERACTIVE_SAMPLES_PER_BATCH = 6;  // Fixed samples per batch in interactive mode
    // Per-sample luminance ceiling while the camera moves. Each motion frame is displayed on its
    // own with only a few samples per pixel, so any sample of a few units shows as a bright dot;
    // the still image uses a ceiling that grows with the sample count instead (unbiased in the limit).
    inline constexpr float MOTION_FIREFLY_CLAMP = 1.0f;
    inline constexpr int INTERACTIVE_MAX_SPP = 100000;       // Max SPP budget for interactive accumulation
};