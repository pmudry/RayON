#pragma once
/**
 * @file tone_mapping.hpp
 * @brief Display tone mapping: exposure and a filmic curve applied to the pixel mean before gamma.
 *
 * Without it the display is gamma + clip: a face that reflects a light far above white stays pure
 * white even on a half-covered pixel, so its outline shows as a staircase whatever the sample
 * count. A curve with a shoulder keeps those pixels below white and the edge antialiased. Only the
 * display is affected; the accumulation buffer stays linear and unbiased.
 */

#include <string>

#ifdef __CUDACC__
#define TONEMAP_HD __host__ __device__
#else
#define TONEMAP_HD
#endif

enum class ToneMapOperator : int
{
   NONE     = 0, ///< gamma + clip (the historical look)
   ACES     = 1, ///< filmic: ACES fit (Narkowicz 2015), soft shoulder, slightly more contrast
   REINHARD = 2, ///< L / (1 + L) on luminance: gentle, keeps hues, flatter highlights
};

struct ToneMapSettings
{
   int   op       = static_cast<int>(ToneMapOperator::NONE);
   float exposure = 1.0f; ///< linear multiplier applied before the curve
};

/// Tone mapping used by the next display conversion (offline image or interactive frame). The
/// renderers set it from the scene, the interactive modes from the GUI before each conversion.
inline ToneMapSettings &displayToneMap()
{
   static ToneMapSettings settings;
   return settings;
}

/// GUI labels, indexed by ToneMapOperator
inline const char *const *toneMapNames()
{
   static const char *const names[] = {"None (clip)", "ACES filmic", "Reinhard"};
   return names;
}

/// YAML name -> operator ("none", "aces", "reinhard"); unknown names give NONE
inline int toneMapFromName(const std::string &name)
{
   if (name == "aces" || name == "filmic")
      return static_cast<int>(ToneMapOperator::ACES);
   if (name == "reinhard")
      return static_cast<int>(ToneMapOperator::REINHARD);
   return static_cast<int>(ToneMapOperator::NONE);
}

inline const char *toneMapYamlName(int op)
{
   return op == static_cast<int>(ToneMapOperator::ACES)       ? "aces"
          : op == static_cast<int>(ToneMapOperator::REINHARD) ? "reinhard"
                                                               : "none";
}

/// Exposure and curve on a linear colour, in place; the result is display-linear in [0, 1]
/// (NONE leaves values above 1 for the caller's clip)
TONEMAP_HD inline void applyToneMap(float &r, float &g, float &b, int op, float exposure)
{
   r = (r > 0.0f ? r : 0.0f) * exposure;
   g = (g > 0.0f ? g : 0.0f) * exposure;
   b = (b > 0.0f ? b : 0.0f) * exposure;
   if (op == static_cast<int>(ToneMapOperator::ACES))
   {
      auto aces = [](float x) { return (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f); };
      r = aces(r);
      g = aces(g);
      b = aces(b);
   }
   else if (op == static_cast<int>(ToneMapOperator::REINHARD))
   {
      const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
      if (lum > 0.0f)
      {
         const float s = 1.0f / (1.0f + lum);
         r *= s;
         g *= s;
         b *= s;
      }
   }
}
