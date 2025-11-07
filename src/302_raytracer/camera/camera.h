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
 * - renderer_cuda.h: CUDA GPU renderer
 * - renderer_cuda_progressive.h: Interactive CUDA renderer with progressive sampling
 */
#pragma once

#include "camera_base.h"
#include "cpu_renderers/renderer_cpu.h"
#include "cpu_renderers/renderer_cpu_parallel.h"
#include "gpu_renderers/renderer_cuda.h"

#ifdef SDL2_FOUND
#include "gpu_renderers/renderer_cuda_progressive.h"
#endif

/**
 * @class Camera
 * @brief Main camera class that combines all rendering capabilities
 *
 * This class inherits from all renderer classes to provide a unified interface.
 * Virtual inheritance is used to avoid the diamond problem with CameraBase.
 */
class Camera : public RendererCPU, public RendererCPUParallel, public RendererCUDA
#ifdef SDL2_FOUND
               ,
               public RendererCUDAProgressive
#endif
{
 public:
   // Constructor that initializes the virtual base class CameraBase
   Camera(const Point3 &center, const int image_width, const int image_height, const int image_channels,
          int samples_per_pixel = 1)
       : CameraBase(center, image_width, image_height, image_channels, samples_per_pixel)
   {
   }

   Camera() : CameraBase(Vec3(0, 0, 0), 720, 3, 1) {}

   // All rendering methods are inherited:
   // - renderPixels() from RendererCPU
   // - renderPixelsParallel() from RendererCPUParallel
   // - renderPixelsCUDA() from RendererCUDA
   // - renderPixelsSDLContinuous() from RendererCUDAProgressive (if SDL2_FOUND)
};
