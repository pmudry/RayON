#pragma once

#include <vector>

/// Lightweight view that wraps a raw RGBA/RGB pixel buffer and its dimensions.
struct RenderTargetView
{
   std::vector<unsigned char> *pixels = nullptr;
   int width = 0;
   int height = 0;
   int channels = 0;

   /// Returns true if the view points to a valid buffer with non-zero dimensions.
   bool isValid() const { return pixels != nullptr && width > 0 && height > 0 && channels > 0; }
};
