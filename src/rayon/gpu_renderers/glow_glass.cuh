#pragma once
/**
 * @file glow_glass.cuh
 * @brief Interior of the GLOW_GLASS material, shared by CUDA and OptiX.
 *
 * A glass shell (refraction, Fresnel, total internal reflection: the GLASS code) filled with a
 * medium that glows and absorbs, like uranium glass or a lit resin block. Along a segment of
 * length d inside, with emission E per unit length and absorption sigma:
 *    gathered light = E (1 - exp(-sigma d)) / sigma      (E d when sigma = 0)
 *    transmittance  = exp(-sigma d)
 * Thick parts glow more than thin edges, and the object lights its surroundings a little through
 * the paths that cross it. It is not a NEE light: that light is found by the BSDF paths alone.
 */

#ifdef __CUDACC__
#define GLOW_HD __host__ __device__
#else
#define GLOW_HD
#endif

/// Glow gathered over d units of medium with absorption sigma (multiply by E); transmittance in T
GLOW_HD inline float glow_segment(float d, float sigma, float &T)
{
   if (sigma <= 1e-4f)
   {
      T = 1.0f;
      return d;
   }
   T = expf(-sigma * d);
   return (1.0f - T) / sigma;
}
