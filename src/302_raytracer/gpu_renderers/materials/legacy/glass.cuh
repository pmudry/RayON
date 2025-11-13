/**
 * @file glass.cuh
 * @brief Glass/dielectric material with refraction and reflection
 *
 * Implements transparent materials using Snell's law and Schlick's approximation
 * for Fresnel reflectance. Supports total internal reflection.
 */

#pragma once
#include "cuda_raytracer.cuh"
#include "cuda_utils.cuh"
#include "material_base.cuh"

// Forward declarations
__device__ float rand_float(curandState *state);
extern __constant__ float g_glass_refraction_index;

namespace Materials
{

/**
 * @brief Glass/dielectric material with refraction
 *
 * Handles both reflection and refraction at dielectric interfaces.
 * Uses Fresnel equations (via Schlick's approximation) to determine
 * the reflection/refraction probability.
 */
struct Glass : public MaterialBase<Glass>
{
   GlassParams params;

   /**
    * @brief Construct glass material
    * @param p Material parameters (refractive index)
    */
   __device__ __forceinline__ Glass(const GlassParams &p) : params(p) {}

   /**
    * @brief Scatter incident ray via refraction or reflection
    *
    * Determines whether to reflect or refract based on:
    * 1. Total internal reflection condition (Snell's law)
    * 2. Fresnel reflectance (probabilistic using Schlick's approximation)
    *
    * @param r_in Incident ray
    * @param rec Hit record containing surface information
    * @param attenuation Output: color attenuation (white for clear glass)
    * @param scattered Output: scattered ray direction
    * @param state Random number generator state
    * @return true (glass always scatters)
    */
   __device__ bool scatter(const ray_simple &r_in, const hit_record_simple &rec, f3 &attenuation, ray_simple &scattered,
                           curandState *state) const
   {

      f3 unit_direction = unit_vector(r_in.dir);

      // Use global refraction index override if available
      float effective_refraction_index = g_glass_refraction_index;

      // Determine if we're entering or exiting the material
      // front_face = true means we're entering (air -> glass)
      // front_face = false means we're exiting (glass -> air)
      float refraction_ratio = rec.front_face ? (1.0f / effective_refraction_index) // air/glass
                                              : effective_refraction_index;         // glass/air

      // Calculate incident angle
      float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
      float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

      // Check for total internal reflection
      bool cannot_refract = refraction_ratio * sin_theta > 1.0f;

      f3 direction;

      if (cannot_refract || do_reflectance(cos_theta, refraction_ratio) > rand_float(state))
      {
         // Reflect (either forced by TIR or chosen by Fresnel probability)
         direction = do_reflect(unit_direction, rec.normal);
      }
      else
      {
         // Refract according to Snell's law
         direction = do_refract(unit_direction, rec.normal, refraction_ratio);
      }

      scattered = ray_simple(rec.p, direction);
      attenuation = f3(1.0f, 1.0f, 1.0f); // Clear glass doesn't absorb
      return true;
   }

   /**
    * @brief Get emitted light (zero for non-emissive surfaces)
    * @return Black (no emission)
    */
   __device__ __forceinline__ f3 emission() const { return f3(0.0f, 0.0f, 0.0f); }
};

} // namespace Materials
