/**
 * @class Camera
 * @brief Unified camera class with all rendering methods
 *
 * This class provides a single interface to all available rendering methods:
 * - Single-threaded CPU rendering
 * - Multi-threaded parallel CPU rendering  
 * - CUDA GPU rendering
 * - Interactive SDL progressive rendering with camera controls
 *
 * The implementation is split across multiple files for better organization:
 * - camera_base.h: Common camera parameters and ray tracing logic
 * - renderer_cpu.h: Single-threaded CPU renderer
 * - renderer_cpu_parallel.h: Multi-threaded CPU renderer
 * - renderer_cuda_host.hpp: CUDA GPU renderer interface
 * - renderer_cuda_progressive_host.hpp: Interactive CUDA renderer with progressive sampling
 */
#pragma once

#include "camera_base.hpp"

/**
 * @class Camera
 * @brief Main camera class that combines all rendering capabilities
 *
 * This class inherits from all renderer classes to provide a unified interface.
 * Virtual inheritance is used to avoid the diamond problem with CameraBase.
 */
class Camera : public CameraBase
{
 public:
   Camera(const Point3 &center, const int image_width, const int image_height, const int image_channels,
          int samples_per_pixel = 1)
       : CameraBase(center, image_width, image_height, image_channels, samples_per_pixel)
   {
   }

   Camera() : CameraBase(Vec3(0, 0, 0), 720, 720, 3, 1, nullptr) {}
};
