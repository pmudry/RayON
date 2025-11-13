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
    * @param p Material parameters (albedo)
    */
   __device__ __forceinline__ ShowNormals(const ShowNormalsParams &p) : params(p) {}

   /**
    * @brief Show normals material doesn't scatter - shows normal as color
    *
    * @param r_in Incident ray (unused)
    * @param rec Hit record containing surface normal
    * @param attenuation Output: normal mapped to color space [0,1]³
    * @param scattered Output: not set (ray absorbed)
    * @param state Random number generator state (unused)
    * @return false (ray absorbed, path terminates)
    */
   __device__ bool scatter(const ray_simple &r_in, const hit_record_simple &rec, f3 &attenuation, ray_simple &scattered,
                           curandState *state) const
   {
      // Map normal from [-1,1]³ to [0,1]³ for color display
      attenuation = f3(0.5f * (rec.normal.x + 1.0f), 0.5f * (rec.normal.y + 1.0f), 0.5f * (rec.normal.z + 1.0f));
      return false; // Don't scatter - terminate path
   }

   /**
    * @brief Get emitted light (zero for non-emissive surfaces)
    * @return Black (no emission)
    */
   __device__ __forceinline__ f3 emission() const { return f3(0.0f, 0.0f, 0.0f); }
};

} // namespace Materials
