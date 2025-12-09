#include "render_coordinator.hpp"

#include "render_target.hpp"

void RenderCoordinator::render(IRenderer &renderer, std::vector<unsigned char> &image, float gamma)
{
   const int required_size = camera_.image_width * camera_.image_height * camera_.image_channels;
   if (static_cast<int>(image.size()) != required_size)
   {
      image.assign(required_size, 0);
   }

   camera_.updateFrame();

   RenderTargetView target{&image, camera_.image_width, camera_.image_height, camera_.image_channels};
   RenderRequest request{camera_, scene_, target};
   RenderContext context{camera_.n_rays, gamma};

   renderer.render(request, context);
   
   last_device_name = context.device_name;
   last_vram_usage = context.vram_usage_bytes;
}
