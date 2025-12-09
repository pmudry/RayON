#pragma once

#include <vector>
#include <string>

#include "renderer_interface.hpp"

/// Helper that wires a camera and scene together and dispatches render requests.
class RenderCoordinator
{
 public:
   RenderCoordinator(CameraBase &camera, const Scene::SceneDescription &scene) : camera_(camera), scene_(scene) {}

   CameraBase &camera() { return camera_; }
   const Scene::SceneDescription &scene() const { return scene_; }

   /// Builds a render request for the provided renderer and writes into the CPU-side image buffer.
   void render(IRenderer &renderer, std::vector<unsigned char> &image, float gamma = 2.0f);

   const std::string& getDeviceName() const { return last_device_name; }
   size_t getVramUsage() const { return last_vram_usage; }

 private:
   CameraBase &camera_;
   const Scene::SceneDescription &scene_;
   std::string last_device_name = "Unknown";
   size_t last_vram_usage = 0;
};
