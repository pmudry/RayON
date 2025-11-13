/**
 * @file constant.cuh
 * @brief Constant material that absorbs rays and returns a solid color
 *
 * This material is useful for debugging and simple non-scattering surfaces.
 * It doesn't scatter rays - just absorbs them and returns its color.
 */

#pragma once

#include "../material_base.cuh"
#include "../../cuda_utils.cuh"

namespace Materials
{

/**
 * @brief Constant material - absorbs rays without scattering
 *
 * This material simply returns its color and terminates the ray path.
 * Useful for background elements or debugging.
 */
struct Constant : public MaterialBase<Constant>
{
   ConstantParams params;

   /**
    * @brief Construct constant material
    * @param p Material parameters (color)
    */
   __device__ __forceinline__ Constant(const ConstantParams &p) : params(p) {}

   /**
    * @brief Constant materials don't scatter - they absorb rays
    *
    * @param r_in Incident ray (unused)
    * @param rec Hit record (unused)
    * @param attenuation Output: set to material color
    * @param scattered Output: not set (ray absorbed)
    * @param state Random number generator state (unused)
    * @return false (ray absorbed, path terminates)
    */
   __device__ bool scatter(const ray_simple &r_in, const hit_record_simple &rec, f3 &attenuation, ray_simple &scattered,
                           curandState *state) const
   {
      // Absorb the ray and set color
      attenuation = params.color;
      return false; // Don't scatter - terminate path
   }

   /**
    * @brief Get emitted light (zero for non-emissive surfaces)
    * @return Black (no emission)
    */
   __device__ __forceinline__ f3 emission() const { return f3(0.0f, 0.0f, 0.0f); }
};

} // namespace Materials
