/**
 * @file texture_loader.cc
 * @brief Implements SceneDescription::addTexture() using stb_image
 *
 * Compiled separately so that STB_IMAGE_IMPLEMENTATION is defined exactly
 * once. The sdl_gui_handler.hpp uses STB_IMAGE_STATIC for its own copy, so
 * there is no duplicate-symbol conflict.
 */

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "scene_description.hpp"
#include <iostream>

namespace Scene
{

int SceneDescription::addTexture(const std::string &path)
{
   // Deduplicate by path — return existing id if already loaded
   for (int i = 0; i < static_cast<int>(textures.size()); ++i)
   {
      if (textures[i].path == path)
         return i;
   }

   TextureDesc tex;
   tex.path = path;

   int channels = 0;
   unsigned char *raw = stbi_load(path.c_str(), &tex.width, &tex.height, &channels, 4 /*force RGBA*/);
   if (!raw)
   {
      std::cerr << "Texture: Cannot load '" << path << "': " << stbi_failure_reason() << "\n";
      return -1;
   }

   tex.channels = 4;
   const size_t sz = static_cast<size_t>(tex.width) * static_cast<size_t>(tex.height) * 4u;
   tex.data.assign(raw, raw + sz);
   stbi_image_free(raw);

   std::cout << "Texture: Loaded '" << path << "' (" << tex.width << "x" << tex.height << " RGBA)\n";
   textures.push_back(std::move(tex));
   return static_cast<int>(textures.size() - 1);
}

} // namespace Scene
