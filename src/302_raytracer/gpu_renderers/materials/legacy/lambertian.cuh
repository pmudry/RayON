/**
 * @file lambertian.cuh
 * @brief Lambertian (diffuse) material implementation
 *
 * Implements ideal diffuse reflection using cosine-weighted hemisphere sampling.
 * This is the physically-based model for rough, non-metallic surfaces.
 */

#pragma once
#include "cuda_raytracer.cuh"
#include "cuda_utils.cuh"
#include "material_base.cuh"

// Forward declaration from shader_common.cuh
__device__ f3 randUnitVector(curandState *state);

namespace Materials
{

/**
 * @brief Lambertian diffuse material
 *
 * Scatters rays uniformly over the hemisphere oriented along the surface normal,
 * with probability weighted by the cosine of the angle from the normal (Lambert's law).
 */
struct Lambertian : public MaterialBase<Lambertian>
{
   LambertianParams params;

   /**
    * @brief Construct Lambertian material
    * @param p Material parameters (albedo)
    */
   __device__ __forceinline__ Lambertian(const LambertianParams &p) : params(p) {}

   /**
    * @brief Scatter incident ray using cosine-weighted hemisphere sampling
    *
    * @param r_in Incident ray
    * @param rec Hit record containing surface information
    * @param attenuation Output: color attenuation from surface reflection
    * @param scattered Output: scattered ray direction
    * @param state Random number generator state
    * @return true (Lambertian surfaces always scatter)
    */
   __device__ bool scatter(const ray_simple &r_in, const hit_record_simple &rec, f3 &attenuation, ray_simple &scattered,
                           curandState *state) const
   {

      // Adding a random unit vector to the normal produces the correct distribution
      f3 scatter_direction = rec.normal + randUnitVector(state);

      // Catch degenerate case where random vector exactly cancels the normal
      // (happens with probability ~0, but prevents NaN propagation)
      if (scatter_direction.length_squared() < 1e-8f)
      {
         scatter_direction = rec.normal;
      }

      scattered = ray_simple(rec.p, scatter_direction);
      attenuation = params.albedo;
      return true;
   }

   /**
    * @brief Get emitted light (zero for non-emissive surfaces)
    * @return Black (no emission)
    */
   __device__ __forceinline__ f3 emission() const { return f3(0.0f, 0.0f, 0.0f); }
};

} // namespace Materials
