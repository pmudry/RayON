#pragma once

#include <atomic>
#include <string>

#include "camera/camera_base.hpp"
#include "render_target.hpp"

namespace Scene
{
class SceneDescription;
}

namespace Rayon {
    struct BenchmarkConfig;
}

/// Shared render state that renderers can update or consult while producing an image.
struct RenderContext
{
   /// Global ray counter that accumulates rays shot across renderer invocations.
   std::atomic<long long> &ray_counter;
   /// Gamma value used when writing pixels back to the render target.
   float gamma = 2.0f;
};

/// Bundles the immutable inputs required by a renderer to produce an image.
struct RenderRequest
{
   /// Camera that provides the frame configuration for ray generation.
   CameraBase &camera;
   /// Scene description shared across CPU and GPU builders.
   const Scene::SceneDescription &scene;
   /// Output buffer the renderer writes into.
   RenderTargetView target;
};

/// Minimal interface implemented by every rendering backend (CPU, CUDA, SDL, ...).
class IRenderer
{
 public:
   virtual ~IRenderer() = default;
   /// Executes a render request and writes into the provided target.
   virtual void render(const RenderRequest &request, RenderContext &context) = 0;

   /// Optional: Set benchmark configuration for renderers that support it
   virtual void setBenchmarkConfig(const Rayon::BenchmarkConfig& config) {}
};
