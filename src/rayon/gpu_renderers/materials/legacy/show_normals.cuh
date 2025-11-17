/**
 * @file show_normals.cuh
 * @brief Material for visualizing surface normals
 *
 * Debug material that displays surface normals as colors.
 * Useful for verifying geometry and normal calculations.
 */

#pragma once
#include "cuda_raytracer.cuh"
#include "material_base.cuh"

namespace Materials
{

/**
 * @brief Material that visualizes surface normals as colors
 *
 * Maps normal vector components to RGB:
 * - X component → Red channel
 * - Y component → Green channel
 * - Z component → Blue channel
 * Normals are remapped from [-1,1] to [0,1] for display.
 */
struct ShowNormals : public MaterialBase<ShowNormals>
{
   ShowNormalsParams params;

   /**
    * @brief Construct show normals material
    * @param p Material parameters (albedo and normal)
    */
   __device__ __forceinline__ ShowNormals(const ShowNormalsParams &p) : params(p) {}

   /**
    * @brief Show normals material doesn't scatter - shows normal as color
    *
    * @param r_in Incident ray (unused)
    * @param rec Hit record containing surface normal
    * @param attenuation Output: not used
    * @param scattered Output: not set (ray absorbed)
    * @param state Random number generator state (unused)
    * @return false (ray absorbed, path terminates)
    */
   __device__ bool scatter(const ray_simple &r_in, const hit_record_simple &rec, f3 &attenuation, ray_simple &scattered,
                           curandState *state) const
   {
      return false; // Don't scatter - emit normal color instead
   }

   /**
    * @brief Get emitted color (displays the surface normal as color)
    * @return Normal vector mapped to RGB color space [0,1]³
    */
   __device__ __forceinline__ f3 emission() const { 
      // Map normal from [-1,1] to [0,1] for color display
      return 0.5f * (params.normal + f3(1.0f, 1.0f, 1.0f)); 
   }
};

} // namespace Materials
