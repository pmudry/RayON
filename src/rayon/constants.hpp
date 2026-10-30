#pragma once

#include <string>

namespace constants
{
    inline const std::string ver_major = "1";
    inline const std::string ver_minor = "8";
    inline const std::string ver_patch = "0";
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
    inline constexpr int INTERACTIVE_SAMPLES_PER_BATCH = 8;  // Fixed samples per batch in interactive mode
    // Per-sample luminance ceiling while the camera moves (the still image uses a ceiling that
    // grows with the sample count instead, unbiased in the limit). Kept at the historical level:
    // a low ceiling darkens bright regions noticeably between motion and rest. Isolated bright
    // dots of the motion preview are handled at display time instead, see MOTION_DESPECKLE_*.
    inline constexpr float MOTION_FIREFLY_CLAMP = 20.0f;
    // Display-time despeckle of motion frames: a pixel whose luminance exceeds
    // RATIO x (mean of its 8 neighbours) + FLOOR is scaled down to that level. Only isolated dots
    // are touched; uniformly bright regions keep their value, and the accumulation is untouched.
    inline constexpr float MOTION_DESPECKLE_RATIO = 2.0f;
    inline constexpr float MOTION_DESPECKLE_FLOOR = 0.02f;
    inline constexpr int INTERACTIVE_MAX_SPP = 100000;       // Max SPP budget for interactive accumulation
};