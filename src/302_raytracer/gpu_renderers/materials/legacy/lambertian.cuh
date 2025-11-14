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
#include <cassert>
#include <math_constants.h>
#include <stdio.h>

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

   __device__ __forceinline__ f3 sample_cosine_weighted_hemisphere(curandState *state) const
   {
      float u1 = rand_float(state);
      float u2 = rand_float(state);

      float r = sqrtf(u1);
      float theta = 2.0f * CUDART_PI_F * u2;

      float x = r * cosf(theta);
      float y = r * sinf(theta);
      float z = sqrtf(fmaxf(0.0f, 1.0f - u1));

      return f3(x, y, z);
   }

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
      f3 u, v;
      f3 w = normalize(rec.normal);
      build_orthonormal_basis(w, u, v);

      f3 local_dir = sample_cosine_weighted_hemisphere(state);

      f3 world_dir = local_dir.x * u + local_dir.y * v + local_dir.z * w;
      f3 scatter_direction = world_dir;

      //      scatter_direction = w + randInHemisphere(rec.normal, state);
      scattered = ray_simple(rec.p + 0.0001 * w,
                             scatter_direction); // The offset avoids self-intersection artifacts on large objects
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
