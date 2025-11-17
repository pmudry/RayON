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

#define COSINE_IMPLEMENTATION 1
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

   __device__ __forceinline__ unsigned int owen_hash(unsigned int x) const
   {
      x ^= x >> 17;
      x *= 0xed5ad4bbU;
      x ^= x >> 11;
      x *= 0xac4c1b51U;
      x ^= x >> 15;
      x *= 0x31848babU;
      x ^= x >> 14;
      return x;
   }

   __device__ inline float next_scrambled(curandState *rng, unsigned int salt) const
   {
      // true random → uint
      unsigned int r = __float_as_uint(rand_float(rng));

      // mix with pixel/salt
      r ^= salt;

      // scramble
      r = owen_hash(r);

      return (r & 0x00FFFFFF) * (1.0f / 16777216.0f);
   }

   /**
    * @brief Construct Lambertian material
    * @param p Material parameters (albedo)
    */
   __device__ __forceinline__ Lambertian(const LambertianParams &p) : params(p) {}

   __device__ __forceinline__ f3 sample_cosine_weighted_hemisphere(curandState *state) const
   {
      // This approach is simple
      // float u1 = rand_float(state);
      // float u2 = rand_float(state);

      // This approach uses Owen scrambling for better stratification
      unsigned int salt = __float_as_uint(rand_float(state)) * 92837111u;
      float u1 = next_scrambled(state, salt);
      float u2 = next_scrambled(state, salt ^ 0x6c8e9cf5u);

      // Generate cosine-weighted direction
      float r = sqrtf(u1);
      float theta = 2.0f * CUDART_PI_F * u2;

      float x = r * cosf(theta);
      float y = r * sinf(theta);
      float z = sqrtf(fmaxf(0.0f, 1.0f - u1));

      return f3(x, y, z);
   }

   __device__ __forceinline__ f3 random_in_hemisphere(const f3 &normal, curandState *rng) const
   {
      f3 in_unit_sphere = randOnUnitSphere(rng);
      if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
         return in_unit_sphere;
      else
         return -in_unit_sphere;
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
#if COSINE_IMPLEMENTATION
      f3 u, v;
      f3 w = normalize(rec.normal);
      build_orthonormal_basis(w, u, v);

      f3 local_dir = sample_cosine_weighted_hemisphere(state);

      f3 world_dir = local_dir.x * u + local_dir.y * v + local_dir.z * w;
      f3 scatter_direction = world_dir;
#else
      f3 scatter_direction = rec.normal + random_in_hemisphere(rec.normal, state);
#endif

      // The offset avoids self-intersection artifacts on large objects
      scattered = ray_simple(rec.p + 0.0001 * rec.normal, scatter_direction);
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
