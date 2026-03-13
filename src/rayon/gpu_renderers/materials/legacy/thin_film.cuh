/**
 * @file thin_film.cuh
 * @brief Thin-film interference material (soap bubbles, oil slicks)
 *
 * Implements an analytic RGB approximation of thin-film interference using
 * the Airy reflectance formula. For each RGB channel, a representative
 * wavelength is chosen and the total reflected intensity is computed from
 * the interference of light bouncing between the two surfaces of a thin
 * dielectric film.
 *
 * No spectral sampling is used — instead we evaluate the Airy formula at
 * three fixed wavelengths (R=650nm, G=550nm, B=450nm) and pack the result
 * into an RGB reflectance color.
 *
 * References:
 *   - "Thin-film interference", Wikipedia
 *   - Glassner, "Principles of Digital Image Synthesis", Ch. 25
 */

#pragma once
#include "cuda_raytracer.cuh"
#include "cuda_utils.cuh"
#include "material_base.cuh"

// Forward declarations
__device__ float rand_float(curandState *state);

namespace Materials
{

/**
 * @brief Parameters for thin-film interference material
 */
struct ThinFilmParams
{
   float film_thickness;   // Film thickness in nanometers
   float film_ior;         // Refractive index of the film layer
   float exterior_ior;     // Refractive index outside the film (usually 1.0 air)
};

/**
 * @brief Thin-film interference material
 *
 * Models a thin dielectric film (e.g. a soap bubble wall) where light
 * reflects from both the outer and inner surfaces. The two reflected
 * beams interfere constructively or destructively depending on the
 * optical path difference, which is a function of film thickness,
 * film IOR, and the angle of incidence.
 *
 * The bubble is treated as a transparent shell: some light is reflected
 * with the iridescent thin-film color, and the rest is transmitted
 * through. The reflection probability is used stochastically per-ray.
 */
struct ThinFilm : public MaterialBase<ThinFilm>
{
   ThinFilmParams params;

   __device__ __forceinline__ ThinFilm(const ThinFilmParams &p) : params(p) {}

   /**
    * @brief Compute Fresnel reflectance at a single interface (Schlick)
    */
   __device__ __forceinline__ float fresnel_r(float cos_i, float n1, float n2) const
   {
      float r0 = (n1 - n2) / (n1 + n2);
      r0 = r0 * r0;
      float x = 1.0f - cos_i;
      float x2 = x * x;
      return r0 + (1.0f - r0) * x2 * x2 * x; // Schlick's approximation
   }

   /**
    * @brief Compute thin-film reflectance for a single wavelength (Airy formula)
    *
    * Models the infinite sum of reflected beams from a thin parallel-sided
    * dielectric slab. The result oscillates between constructive and
    * destructive interference as a function of the optical path difference.
    *
    * @param wavelength_nm  Wavelength in nanometers
    * @param cos_theta_i    Cosine of incidence angle (in exterior medium)
    * @return Reflectance in [0, 1]
    */
   __device__ float airy_reflectance(float wavelength_nm, float cos_theta_i) const
   {
      float n0 = params.exterior_ior; // exterior (air)
      float n1 = params.film_ior;     // film
      float n2 = params.exterior_ior; // interior (air again for a bubble)

      // Snell's law: compute cos(theta) inside the film
      float sin_theta_i = sqrtf(fmaxf(0.0f, 1.0f - cos_theta_i * cos_theta_i));
      float sin_theta_t = (n0 / n1) * sin_theta_i;

      // Total internal reflection shouldn't happen for typical thin-film configs
      // but clamp for safety
      if (sin_theta_t >= 1.0f)
         return 1.0f;

      float cos_theta_t = sqrtf(1.0f - sin_theta_t * sin_theta_t);

      // Fresnel reflectance at each interface
      float R01 = fresnel_r(cos_theta_i, n0, n1); // exterior → film
      float R12 = fresnel_r(cos_theta_t, n1, n2); // film → interior

      // Phase difference from optical path length through the film (round trip)
      // delta = (4 * pi * n_film * d * cos(theta_t)) / lambda
      float delta = (4.0f * CUDART_PI_F * n1 * params.film_thickness * cos_theta_t) / wavelength_nm;

      // Airy formula for total reflectance of a thin film:
      //   R = (R01 + R12 + 2*sqrt(R01*R12)*cos(delta)) /
      //       (1 + R01*R12 + 2*sqrt(R01*R12)*cos(delta))
      float sqrt_R = sqrtf(R01 * R12);
      float cos_delta = cosf(delta);
      float numerator = R01 + R12 + 2.0f * sqrt_R * cos_delta;
      float denominator = 1.0f + R01 * R12 + 2.0f * sqrt_R * cos_delta;

      return numerator / fmaxf(denominator, 1e-8f);
   }

   /**
    * @brief Compute RGB thin-film reflectance color
    *
    * Evaluates the Airy formula at three representative wavelengths:
    *   R = 650 nm, G = 550 nm, B = 450 nm
    *
    * @param cos_theta_i Cosine of incidence angle
    * @return f3 with per-channel reflectance
    */
   __device__ __forceinline__ f3 thin_film_color(float cos_theta_i) const
   {
      float R_r = airy_reflectance(650.0f, cos_theta_i); // Red
      float R_g = airy_reflectance(550.0f, cos_theta_i); // Green
      float R_b = airy_reflectance(450.0f, cos_theta_i); // Blue
      return f3(R_r, R_g, R_b);
   }

   /**
    * @brief Scatter incident ray with thin-film interference
    *
    * Stochastically reflects or transmits the ray based on the average
    * thin-film reflectance. Reflected rays carry the iridescent color;
    * transmitted rays pass through with complementary attenuation.
    */
   __device__ bool scatter(const ray_simple &r_in, const hit_record_simple &rec, f3 &attenuation,
                           ray_simple &scattered, curandState *state) const
   {
      f3 unit_dir = normalize(r_in.dir);
      float cos_theta_i = fmaxf(fabsf(dot(-unit_dir, rec.normal)), 0.001f);

      // Compute per-channel thin-film reflectance
      f3 R = thin_film_color(cos_theta_i);

      // Average reflectance for Russian roulette between reflect/transmit
      float avg_R = (R.x + R.y + R.z) / 3.0f;

      if (rand_float(state) < avg_R)
      {
         // Reflect with iridescent color
         f3 reflected = do_reflect(unit_dir, rec.normal);
         scattered = ray_simple(rec.p + 0.0001f * rec.normal, reflected);
         // Energy conservation: divide by probability of choosing reflection
         attenuation = R * (1.0f / fmaxf(avg_R, 0.001f));
         return true;
      }
      else
      {
         // Transmit through the film (the bubble is mostly transparent)
         // Slight refraction through the thin film is negligible; pass straight through
         scattered = ray_simple(rec.p - 0.0001f * rec.normal, unit_dir);
         // Transmitted energy = (1 - R), divided by probability of choosing transmission
         f3 T = f3(1.0f - R.x, 1.0f - R.y, 1.0f - R.z);
         attenuation = T * (1.0f / fmaxf(1.0f - avg_R, 0.001f));
         return true;
      }
   }

   __device__ __forceinline__ f3 emission() const { return f3(0.0f, 0.0f, 0.0f); }
};

} // namespace Materials
