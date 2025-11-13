/**
 * @file mirror.cuh
 * @brief Perfect mirror (specular reflection) material
 *
 * Implements ideal specular reflection with optional tinting.
 * No roughness - produces perfectly sharp reflections.
 */

#pragma once
#include "cuda_raytracer.cuh"
#include "cuda_utils.cuh"
#include "material_base.cuh"

namespace Materials
{

/**
 * @brief Perfect mirror material with specular reflection
 *
 * Reflects incident rays according to the law of reflection (angle of incidence = angle of reflection).
 * The reflection direction is deterministic with no random perturbation.
 */
struct Mirror : public MaterialBase<Mirror>
{
   MirrorParams params;

   /**
    * @brief Construct mirror material
    * @param p Material parameters (albedo for tinting)
    */
   __device__ __forceinline__ Mirror(const MirrorParams &p) : params(p) {}

   /**
    * @brief Scatter incident ray via perfect specular reflection
    *
    * @param r_in Incident ray
    * @param rec Hit record containing surface information
    * @param attenuation Output: color attenuation (tint)
    * @param scattered Output: scattered ray direction
    * @param state Random number generator state (unused for perfect mirror)
    * @return true if ray reflects (false if absorbed below surface)
    */
   __device__ bool scatter(const ray_simple &r_in, const hit_record_simple &rec, f3 &attenuation, ray_simple &scattered,
                           curandState *state) const
   {
      f3 reflected = do_reflect(unit_vector(r_in.dir), rec.normal);
      scattered = ray_simple(rec.p, reflected);
      attenuation = params.albedo;

      // Absorb rays that reflect below the surface (shouldn't happen with proper normals)
      return dot(scattered.dir, rec.normal) > 0.0f;
   }

   /**
    * @brief Get emitted light (zero for non-emissive surfaces)
    * @return Black (no emission)
    */
   __device__ __forceinline__ f3 emission() const { return f3(0.0f, 0.0f, 0.0f); }
};

} // namespace Materials
