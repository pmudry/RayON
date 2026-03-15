/**
 * @file clear_coat.cuh
 * @brief Clear-coat material: glossy dielectric coat over a diffuse base
 *
 * Two-lobe model used for car paint, lacquered wood, plastic toys, etc.
 * The coat is a smooth (or slightly rough) dielectric whose reflectance is
 * governed by Schlick's Fresnel approximation. The base layer is Lambertian.
 *
 * Algorithm (stochastic):
 *   1. Compute Fresnel reflectance F at the coat interface.
 *   2. With probability F  → reflect off the coat (glossy specular, achromatic).
 *   3. With probability (1-F) → scatter into the cosine-weighted base hemisphere
 *      and attenuate by the base albedo.
 *
 * This is an unbiased estimator of the two-lobe BRDF.
 */

#pragma once
#include "cuda_raytracer.cuh"
#include "cuda_utils.cuh"
#include "material_base.cuh"

__device__ f3 randOnUnitSphere(curandState *state);
__device__ float rand_float(curandState *state);

namespace Materials
{

struct ClearCoatParams
{
   f3 albedo;        // Base diffuse color
   float roughness;  // Coat roughness (0 = mirror coat, 0.2 = slightly rough)
   float coat_ior;   // Coat refractive index (1.5 = lacquer/plastic)
};

struct ClearCoat : public MaterialBase<ClearCoat>
{
   ClearCoatParams params;

   __device__ __forceinline__ ClearCoat(const ClearCoatParams &p) : params(p) {}

   __device__ __forceinline__ float schlick(float cos_theta) const
   {
      float r0 = (1.0f - params.coat_ior) / (1.0f + params.coat_ior);
      r0 = r0 * r0;
      float x = 1.0f - cos_theta;
      return r0 + (1.0f - r0) * x * x * x * x * x;
   }

   __device__ bool scatter(const ray_simple &r_in, const hit_record_simple &rec,
                           f3 &attenuation, ray_simple &scattered, curandState *state) const
   {
      f3 unit_dir = normalize(r_in.dir);
      float cos_theta = fminf(dot(-unit_dir, rec.normal), 1.0f);
      float F = schlick(cos_theta);

      if (rand_float(state) < F)
      {
         // Coat specular lobe
         f3 perturbed_normal = (params.roughness > 1e-3f)
            ? normalize(rec.normal + params.roughness * randOnUnitSphere(state))
            : rec.normal;
         f3 reflected = do_reflect(unit_dir, perturbed_normal);
         scattered = ray_simple(rec.p, reflected);
         attenuation = f3(1.0f, 1.0f, 1.0f);
         return dot(reflected, rec.normal) > 0.0f;
      }
      else
      {
         // Base Lambertian lobe — reuse rec fields populated by apply_material
         attenuation = params.albedo;
         // Simple cosine-weighted direction (fast version using rec.normal offset)
         f3 scatter_dir = rec.normal + randOnUnitSphere(state);
         if (scatter_dir.length_squared() < 1e-8f)
            scatter_dir = rec.normal;
         scattered = ray_simple(rec.p, normalize(scatter_dir));
         return true;
      }
   }

   __device__ __forceinline__ f3 emission() const { return f3(0.0f, 0.0f, 0.0f); }
};

} // namespace Materials
