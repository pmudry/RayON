#pragma once
/**
 * @file volume_medium.cuh
 * @brief Heterogeneous participating medium (smoke / cloud) helpers shared by the CUDA and OptiX kernels.
 *
 * A VOLUME material turns the inside of a closed mesh into a scattering medium. The extinction
 * coefficient is a procedural field built from three ingredients (see volume_density()):
 *  - a shape term from the mesh's signed distance grid: the density ramps up over `edge` units
 *    from the surface, and the shape noise (value fBm mixed with inverted Worley "billows")
 *    erodes that ramp so the silhouette frays into lumps and wisps;
 *  - an interior modulation by fractal value noise (`noise_strength`);
 *  - the base `density`.
 * The kernels sample free-flight distances with delta tracking (null collisions against the
 * majorant returned by volume_majorant()) and scatter with the Henyey-Greenstein phase function.
 *
 * The noise works on plain floats so both back ends can use it with their own vector types.
 */

#include "volume_params.h"
#include <cuda_runtime.h>

/// Scattering events a path may take inside a medium before it is cut (a dense white cloud
/// needs tens of events to brighten up; the kernels apply Russian roulette after a few).
constexpr int VOLUME_MAX_EVENTS = 96;
/// Null-collision steps per free flight before giving up (guards against a zero majorant)
constexpr int VOLUME_MAX_NULL_STEPS = 512;
/// Back faces a path may pass through (camera inside the medium, open meshes) before it is cut
constexpr int VOLUME_MAX_PASSES = 8;

__device__ __forceinline__ float vol_hash3(int x, int y, int z)
{
   unsigned int h = (unsigned int)x * 73856093u ^ (unsigned int)y * 19349663u ^ (unsigned int)z * 83492791u;
   h ^= h >> 13;
   h *= 0x5bd1e995u;
   h ^= h >> 15;
   return (float)(h & 0xffffffu) * (1.0f / 16777216.0f);
}

/// Smooth value noise in [0, 1]
__device__ __forceinline__ float vol_value_noise(float x, float y, float z)
{
   const float fx = floorf(x), fy = floorf(y), fz = floorf(z);
   const int ix = (int)fx, iy = (int)fy, iz = (int)fz;
   float tx = x - fx, ty = y - fy, tz = z - fz;
   tx = tx * tx * (3.0f - 2.0f * tx);
   ty = ty * ty * (3.0f - 2.0f * ty);
   tz = tz * tz * (3.0f - 2.0f * tz);
   const float c000 = vol_hash3(ix, iy, iz), c100 = vol_hash3(ix + 1, iy, iz);
   const float c010 = vol_hash3(ix, iy + 1, iz), c110 = vol_hash3(ix + 1, iy + 1, iz);
   const float c001 = vol_hash3(ix, iy, iz + 1), c101 = vol_hash3(ix + 1, iy, iz + 1);
   const float c011 = vol_hash3(ix, iy + 1, iz + 1), c111 = vol_hash3(ix + 1, iy + 1, iz + 1);
   const float x00 = c000 + (c100 - c000) * tx, x10 = c010 + (c110 - c010) * tx;
   const float x01 = c001 + (c101 - c001) * tx, x11 = c011 + (c111 - c011) * tx;
   const float y0 = x00 + (x10 - x00) * ty, y1 = x01 + (x11 - x01) * ty;
   return y0 + (y1 - y0) * tz;
}

/// Four-octave fractal value noise, roughly in [0, 1] with mean 0.5
__device__ __forceinline__ float vol_fbm(float x, float y, float z)
{
   float sum = 0.0f, amp = 0.5f, norm = 0.0f;
   for (int o = 0; o < 4; ++o)
   {
      sum += amp * vol_value_noise(x, y, z);
      norm += amp;
      x = x * 2.02f + 17.1f;
      y = y * 2.02f + 31.7f;
      z = z * 2.02f + 11.3f;
      amp *= 0.5f;
   }
   return sum / norm;
}

/// Inverted Worley (cellular) noise in [0, 1]: 1 at the feature points, 0 far from them, which
/// reads as rounded lumps ("cauliflower" billows) when used as a density
__device__ __forceinline__ float vol_worley(float x, float y, float z)
{
   const float fx = floorf(x), fy = floorf(y), fz = floorf(z);
   const int ix = (int)fx, iy = (int)fy, iz = (int)fz;
   float min_d2 = 8.0f;
   for (int dz = -1; dz <= 1; ++dz)
      for (int dy = -1; dy <= 1; ++dy)
         for (int dx = -1; dx <= 1; ++dx)
         {
            const int cx = ix + dx, cy = iy + dy, cz = iz + dz;
            const float px = (float)cx + vol_hash3(cx, cy, cz);
            const float py = (float)cy + vol_hash3(cx + 131, cy + 71, cz + 19);
            const float pz = (float)cz + vol_hash3(cx + 17, cy + 43, cz + 97);
            const float ddx = px - x, ddy = py - y, ddz = pz - z;
            min_d2 = fminf(min_d2, ddx * ddx + ddy * ddy + ddz * ddz);
         }
   return 1.0f - fminf(sqrtf(min_d2), 1.0f);
}

/// Five-octave inverted Worley in [0, 1]: big lumps carrying ever smaller ones, the way a cumulus
/// is built (the amplitude halves per octave, the frequency doubles)
__device__ __forceinline__ float vol_worley_fbm(float x, float y, float z)
{
   float sum = 0.0f, amp = 0.5f, norm = 0.0f;
   for (int o = 0; o < 5; ++o)
   {
      sum += amp * vol_worley(x, y, z);
      norm += amp;
      x = x * 2.03f + 5.3f;
      y = y * 2.03f + 9.1f;
      z = z * 2.03f + 1.7f;
      amp *= 0.5f;
   }
   return sum / norm;
}

__device__ __forceinline__ float vol_saturate(float v) { return fminf(fmaxf(v, 0.0f), 1.0f); }

/// Extinction multiplier at a point, evaluated procedurally (see the file header for the model).
/// volume_density() reads it from the baked field when one is bound.
__device__ __forceinline__ float volume_field_procedural(const VolumeParams &v, float x, float y, float z)
{
   // Interior modulation: density x (1 + noise_strength x (2 fbm - 1)), clamped at 0
   float interior = 1.0f;
   float fbm = 0.5f;
   const bool need_fbm = (v.noise_strength > 0.0f && v.noise_scale > 0.0f);
   if (need_fbm)
   {
      fbm      = vol_fbm(x * v.noise_scale, y * v.noise_scale, z * v.noise_scale);
      interior = fmaxf(0.0f, 1.0f + v.noise_strength * (2.0f * fbm - 1.0f));
   }

   // Shape: the surface is displaced by +-lumps along the shape noise (bulges where the noise is
   // high, crevices where it is low), then the density ramps up over `edge` units from there.
   float shape = 1.0f;
   if (v.sdf != 0 && v.edge > 0.0f)
   {
      const float u = (x - v.sdf_min.x) * v.sdf_inv_extent.x;
      const float w = (y - v.sdf_min.y) * v.sdf_inv_extent.y;
      const float q = (z - v.sdf_min.z) * v.sdf_inv_extent.z;
      float depth = -tex3D<float>(v.sdf, u, w, q); // > 0 inside the mesh
      const float lumps = fminf(fmaxf(v.lumps, 0.0f), VOLUME_BOX_PAD);
      if (depth >= v.edge + lumps)
         shape = 1.0f; // deep inside: no noise needed
      else if (depth <= -lumps)
         shape = 0.0f; // too far outside for any bulge to reach
      else
      {
         if (lumps > 0.0f && v.noise_scale > 0.0f)
         {
            const float sx = x * v.noise_scale, sy = y * v.noise_scale, sz = z * v.noise_scale;
            float shape_noise = need_fbm ? fbm : vol_fbm(sx, sy, sz);
            if (v.billow > 0.0f)
               shape_noise = (1.0f - v.billow) * shape_noise + v.billow * vol_worley_fbm(sx, sy, sz);
            // Stretch the contrast so the lumps reach the full +-lumps displacement
            shape_noise = vol_saturate((shape_noise - 0.2f) * 1.6f);
            depth += lumps * (2.0f * shape_noise - 1.0f);
         }
         shape = vol_saturate(depth / v.edge);
      }
   }
   return interior * shape;
}

/// Extinction coefficient at a point: the baked field when bound, the procedural model otherwise
__device__ __forceinline__ float volume_density(const VolumeParams &v, float x, float y, float z)
{
   if (v.field != 0)
   {
      const float u = (x - v.sdf_min.x) * v.sdf_inv_extent.x;
      const float w = (y - v.sdf_min.y) * v.sdf_inv_extent.y;
      const float q = (z - v.sdf_min.z) * v.sdf_inv_extent.z;
      return v.density * tex3D<float>(v.field, u, w, q);
   }
   return v.density * volume_field_procedural(v, x, y, z);
}

/// Empty-space skip for the tracking loops: outside the mesh, beyond the reach of the lumps, the
/// density is zero and the distance grid bounds how far that stays true along any direction
/// (|grad sdf| <= 1). Returns that distance, 0 when the point may hold density.
__device__ __forceinline__ float volume_empty_skip(const VolumeParams &v, float x, float y, float z)
{
   if (v.sdf == 0 || v.edge <= 0.0f)
      return 0.0f;
   const float u = (x - v.sdf_min.x) * v.sdf_inv_extent.x;
   const float w = (y - v.sdf_min.y) * v.sdf_inv_extent.y;
   const float q = (z - v.sdf_min.z) * v.sdf_inv_extent.z;
   const float outside = tex3D<float>(v.sdf, u, w, q) - fminf(fmaxf(v.lumps, 0.0f), VOLUME_BOX_PAD);
   // Tiny skips would creep towards the band without ever entering it (and eat the step budget):
   // below a few hundredths of a unit, take a regular stochastic step instead
   return outside > 0.02f ? outside : 0.0f;
}

/// Upper bound of volume_density() over space, for delta tracking
__device__ __forceinline__ float volume_majorant(const VolumeParams &v)
{
   return v.density * (1.0f + fmaxf(v.noise_strength, 0.0f));
}

/// Henyey-Greenstein phase function value (= its sampling pdf, per steradian) for cos(theta)
/// between the incoming propagation direction and the outgoing direction
__device__ __forceinline__ float hg_phase(float g, float cos_theta)
{
   const float denom = 1.0f + g * g - 2.0f * g * cos_theta;
   return (1.0f - g * g) / (12.566370614f * denom * sqrtf(fmaxf(denom, 1e-6f)));
}

/// Henyey-Greenstein: cosine of the scattering angle for asymmetry g in (-1, 1) and uniform u
__device__ __forceinline__ float hg_sample_cos(float g, float u)
{
   if (fabsf(g) < 1e-3f)
      return 1.0f - 2.0f * u;
   const float sq = (1.0f - g * g) / (1.0f - g + 2.0f * g * u);
   return (1.0f + g * g - sq * sq) / (2.0f * g);
}
