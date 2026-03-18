/**
 * @file hdr_env_cache.hpp
 * @brief Fast HDR environment map loader with float16 binary disk cache.
 *
 * On the first load of a .hdr file:
 *   - decodes RGBE via stbi_loadf
 *   - converts float3 → float16 RGBA (half4), clamping to float16 max (65504) to
 *     prevent ±inf in the texture which would cause black fireflies
 *   - writes a .hdrcache binary sidecar alongside the source file
 *
 * On subsequent loads the pre-baked binary cache is used directly, giving
 * roughly 5–10× faster load times for 4K/8K images.
 *
 * Implementation lives in hdr_env_cache.cc (owns STB_IMAGE_IMPLEMENTATION).
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

/**
 * @brief Load (or build+save) float16 RGBA data for a .hdr environment map.
 *
 * Format: w×h × 4 × uint16_t, row-major, IEEE 754 float16.
 * Suitable for a CUDA texture created with {16,16,16,16,cudaChannelFormatKindFloat}.
 *
 * @param hdr_path   Path to the .hdr source file.
 * @param out_w      [out] Image width in pixels.
 * @param out_h      [out] Image height in pixels.
 * @param use_cache  When true (default) reads/writes .hdrcache sidecar.
 *                   Pass false to always decode fresh (useful for debugging).
 * @return Half4 pixel data, or empty vector on failure.
 */
std::vector<uint16_t> loadHdrEnvHalf(const std::string &hdr_path, int &out_w, int &out_h,
                                      bool use_cache = true);
