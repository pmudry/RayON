#pragma once
/**
 * @file chameleon_paint.cuh
 * @brief Colour law of the CHAMELEON material ("flip-flop" paint), shared by CUDA and OptiX.
 *
 * A clear coat whose base colour depends on the viewing angle: `albedo` seen face-on, `flip_color`
 * at grazing angles, the way pearlescent car paints and beetle shells change colour as the surface
 * turns away. The coat itself (Fresnel, roughness, sampling, NEE) is the CLEAR_COAT one; only the
 * base colour of a hit is picked from the cosine between the view ray and the normal.
 */

#ifdef __CUDACC__
#define CHAMELEON_HD __host__ __device__
#else
#define CHAMELEON_HD
#endif

/// Share of flip_color for a view at cos_view from the normal: 0 face-on, 1 from about 75 degrees
/// on, smooth in between (smoothstep over 1 - cos in [0, 0.75])
CHAMELEON_HD inline float chameleon_flip(float cos_view)
{
   float t = (1.0f - fminf(fmaxf(cos_view, 0.0f), 1.0f)) / 0.75f;
   t       = fminf(t, 1.0f);
   return t * t * (3.0f - 2.0f * t);
}
