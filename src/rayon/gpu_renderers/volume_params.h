#pragma once
/**
 * @file volume_params.h
 * @brief Device-side parameters of a VOLUME (cloud / smoke) material, embedded in the CUDA and
 * OptiX material structs. Plain data only; the sampling code lives in volume_medium.cuh.
 */

#include <cuda_runtime.h>

/// Padding of the medium's bounding box around the mesh (and of its distance grid): the room the
/// lumps may grow into outside the mesh surface. Lump amplitudes above this are clipped.
constexpr float VOLUME_BOX_PAD = 0.75f;
/// Resolution of the baked extinction field (cells along the longest side of the box)
constexpr int VOLUME_FIELD_CELLS = 160;

struct VolumeParams
{
   float density;        ///< Extinction coefficient per scene unit
   float noise_scale;    ///< Frequency of the density / shape noise
   float noise_strength; ///< Interior density modulation: 0 homogeneous, 1 full contrast
   float edge;           ///< Depth from the surface over which the density ramps up (0 = hard edge, no SDF)
   float lumps;          ///< Amplitude (scene units) of the noise displacement of the surface, both ways
   float billow;         ///< 0..1: share of inverted Worley (cauliflower lumps) in the shape noise

   // Signed distance field of the bounding mesh (built at scene upload when edge > 0; 0 = none).
   // Stored as a 3D float texture over the box [sdf_min, sdf_min + 1 / sdf_inv_extent].
   cudaTextureObject_t sdf;
   cudaArray_t         sdf_array; ///< Backing array (host side bookkeeping only)
   float3              sdf_min;
   float3              sdf_inv_extent;
   // Baked extinction multiplier (shape x interior noise, in [0, 1 + noise_strength]) over the
   // same box, rebuilt whenever the parameters change: the tracking loops then pay one texture
   // fetch per step instead of the noise evaluation. 0 = evaluate the noise on the fly.
   cudaTextureObject_t field;
   cudaArray_t         field_array;
   int3                field_dims;
};
