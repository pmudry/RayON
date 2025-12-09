#pragma once

#include <string>

namespace Rayon {

/**
 * @brief Configuration for a benchmark run.
 */
struct BenchmarkConfig {
    std::string scene_file;         // Path to the YAML scene file to load
    std::string output_name;        // Base name for output JSON and PNG files
    int target_samples;             // Number of samples per pixel to reach
    float max_time_seconds;         // Maximum time in seconds for the benchmark (0 for no limit)
    int resolution_width;           // Render resolution width
    int resolution_height;          // Render resolution height

    BenchmarkConfig()
        : scene_file("")
        , output_name("benchmark_run")
        , target_samples(1024)
        , max_time_seconds(0.0f) // No time limit by default
        , resolution_width(1280)
        , resolution_height(720)
    {}
};

} // namespace Rayon
