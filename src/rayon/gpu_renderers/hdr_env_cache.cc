/**
 * @file hdr_env_cache.cc
 * @brief Implementation of loadHdrEnvHalf() — fast HDR loader with float16 disk cache.
 *
 * This file owns the one and only STB_IMAGE_IMPLEMENTATION for HDR decoding.
 * Keep it isolated here to avoid polluting other translation units.
 */

// STB_IMAGE_STATIC gives all stb functions internal linkage so there is no
// conflict with the copies in sdl_gui_handler.hpp / texture_loader.cc.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "external/stb_image.h"

#include "hdr_env_cache.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Portable float32 → IEEE 754 float16 bit-cast.
/// Values exceeding float16 max (65504) are clamped to that max rather than
/// being converted to ±inf, which would later propagate NaN through arithmetic
/// and produce black firefly pixels in the rendered image.
static uint16_t f32_to_f16(float f)
{
   // Clamp to float16 representable range before conversion to prevent
   // overflow to ±infinity (which causes NaN in downstream arithmetic).
   constexpr float kHalfMax = 65504.0f;
   if (f > kHalfMax)  f = kHalfMax;
   if (f < -kHalfMax) f = -kHalfMax;

   uint32_t x;
   std::memcpy(&x, &f, 4);
   const uint32_t s = (x >> 16) & 0x8000u;
   const int32_t  e = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
   const uint32_t m = x & 0x7FFFFFu;
   if (e <= 0)  return static_cast<uint16_t>(s);             // ±0 / underflow → zero
   if (e >= 31) return static_cast<uint16_t>(s | 0x7BFFu);  // clamped max normal
   return static_cast<uint16_t>(s | (static_cast<uint32_t>(e) << 10) | (m >> 13));
}

// ---------------------------------------------------------------------------
// Cache format
// ---------------------------------------------------------------------------

static constexpr uint32_t HDR_CACHE_MAGIC   = 0x52484452u; // 'RHDR'
static constexpr uint32_t HDR_CACHE_VERSION = 2u; // v1 → v2: clamps fp16 overflow to ±65504

struct HdrCacheHeader
{
   uint32_t magic;
   uint32_t version;
   uint32_t width;
   uint32_t height;
   uint64_t source_size; ///< byte-size of the original .hdr file (staleness check)
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<uint16_t> loadHdrEnvHalf(const std::string &hdr_path, int &out_w, int &out_h,
                                      bool use_cache)
{
   const std::string cache_path = hdr_path + ".hdrcache";

   // Source file size for staleness detection (C++17 std::filesystem).
   std::error_code fsec;
   uint64_t        src_size = 0u;
   try { src_size = static_cast<uint64_t>(std::filesystem::file_size(hdr_path, fsec)); }
   catch (...) {}

   // --- Try reading the binary cache (fast path) ---
   if (use_cache)
   {
      std::ifstream cache(cache_path, std::ios::binary);
      if (cache)
      {
         HdrCacheHeader hdr{};
         cache.read(reinterpret_cast<char *>(&hdr), sizeof(hdr));
         const bool valid = cache.good()
                            && hdr.magic   == HDR_CACHE_MAGIC
                            && hdr.version == HDR_CACHE_VERSION
                            && hdr.width   >  0
                            && hdr.height  >  0
                            && (src_size == 0 || hdr.source_size == src_size);
         if (valid)
         {
            const size_t n = static_cast<size_t>(hdr.width) * hdr.height * 4u;
            std::vector<uint16_t> data(n);
            cache.read(reinterpret_cast<char *>(data.data()),
                       static_cast<std::streamsize>(n * sizeof(uint16_t)));
            if (cache.good())
            {
               out_w = static_cast<int>(hdr.width);
               out_h = static_cast<int>(hdr.height);
               std::cout << "HDR cache: hit  '" << cache_path
                         << "' (" << out_w << "x" << out_h << ")\n";
               return data;
            }
         }
         std::cerr << "HDR cache: stale or corrupt — re-decoding '" << hdr_path << "'\n";
      }
   }

   // --- Slow path: decode RGBE via stbi ---
   int    w = 0, h = 0, ch = 0;
   float *raw = stbi_loadf(hdr_path.c_str(), &w, &h, &ch, 3);
   if (!raw)
   {
      std::cerr << "HDR: stbi_loadf failed for '" << hdr_path << "'\n";
      return {};
   }

   // Convert float3 → clamped float16 RGBA (A = 1.0 → 0x3C00).
   // Values above 65504 are clamped to prevent ±inf in the GPU texture, which
   // would cause NaN propagation and black firefly pixels.
   const size_t         n = static_cast<size_t>(w) * h * 4u;
   std::vector<uint16_t> data(n);
   for (int i = 0; i < w * h; ++i)
   {
      data[i * 4 + 0] = f32_to_f16(raw[i * 3 + 0]);
      data[i * 4 + 1] = f32_to_f16(raw[i * 3 + 1]);
      data[i * 4 + 2] = f32_to_f16(raw[i * 3 + 2]);
      data[i * 4 + 3] = 0x3C00u; // half(1.0f)
   }
   stbi_image_free(raw);
   out_w = w;
   out_h = h;

   // --- Save cache sidecar (skip when caching is disabled) ---
   if (use_cache)
   {
      std::ofstream cache(cache_path, std::ios::binary);
      if (cache)
      {
         const HdrCacheHeader hdr{HDR_CACHE_MAGIC, HDR_CACHE_VERSION,
                                  static_cast<uint32_t>(w), static_cast<uint32_t>(h), src_size};
         cache.write(reinterpret_cast<const char *>(&hdr),      sizeof(hdr));
         cache.write(reinterpret_cast<const char *>(data.data()),
                     static_cast<std::streamsize>(n * sizeof(uint16_t)));
         if (cache)
            std::cout << "HDR cache: saved '" << cache_path << "' (" << w << "x" << h << ")\n";
         else
            std::cerr << "HDR cache: write failed for '" << cache_path << "'\n";
      }
      else
         std::cerr << "HDR cache: cannot create '" << cache_path << "' (read-only?)\n";
   }

   return data;
}
