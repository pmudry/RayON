#pragma once

#include "color.hpp"
#include "hittable.hpp"
#include "mis_utils.hpp"
#include "ray.hpp"
#include "vec3.hpp"

#include <cmath>

//==============================================================================
// SCATTER RECORD — returned by scatter_mis()
//==============================================================================

/**
 * @brief Result of sampling a scatter direction from a material.
 *
 * For MIS: bsdf_value stores f(wo,wi) (NOT divided by pdf), and pdf stores
 * the probability density of the sampled direction.  Throughput update:
 *   throughput *= bsdf_value * cos_theta / pdf
 * For specular (delta) BSDFs is_specular=true and the attenuation is stored in
 * bsdf_value directly (pdf=1, cos_theta factor absorbed into bsdf_value).
 */
struct ScatterRecord
{
   Ray   scattered;    ///< Outgoing scattered ray
   Color bsdf_value;   ///< f(wo, wi) — NOT divided by pdf
   double pdf;         ///< PDF of the sampled direction (unused when is_specular=true)
   bool  is_specular;  ///< True for delta BSDFs (mirror, glass) — skip MIS
};

//==============================================================================
// MATERIAL BASE CLASS
//==============================================================================

class Material
{
 public:
   virtual ~Material() = default;

   // Legacy interface — kept for backward compatibility
   virtual bool scatter(const Ray &r_in, const Hit_record &rec, Color &attenuation, Ray &scattered) const
   {
      return false;
   }

   // --- MIS interface ---

   /// Emitted radiance at the hit point (default: no emission)
   virtual Color emitted(const Hit_record &) const { return Color(0, 0, 0); }

   /**
    * @brief Sample a scatter direction and return ScatterRecord.
    *
    * Default implementation wraps the legacy scatter() as a specular bounce so
    * that materials not yet converted to MIS still work correctly (they skip NEE
    * and the BSDF is applied as a whole-path weight).
    */
   virtual bool scatter_mis(const Ray &r_in, const Hit_record &rec, ScatterRecord &srec) const
   {
      Color attenuation;
      Ray scattered;
      if (!scatter(r_in, rec, attenuation, scattered))
         return false;
      srec.scattered   = scattered;
      srec.bsdf_value  = attenuation;
      srec.pdf         = 1.0;
      srec.is_specular = true; // treat unimplemented materials as delta (skip MIS)
      return true;
   }

   /// Evaluate f(wo, wi) — the BSDF at a given incoming direction wi (default: 0)
   virtual Color eval_bsdf(const Ray &, const Hit_record &, const Vec3 &) const { return Color(0, 0, 0); }

   /// PDF of sampling direction wi via BSDF scatter (default: 0 — delta or unimplemented)
   virtual double scatter_pdf(const Ray &, const Hit_record &, const Vec3 &) const { return 0.0; }
};

//==============================================================================
// CONSTANT — flat colour, path terminating (debug visualisation)
//==============================================================================

class Constant : public Material
{
 public:
   explicit Constant(const Color &a) : color(a) {}

   bool scatter(const Ray &, const Hit_record &rec, Color &attenuation, Ray &scattered) const override
   {
      attenuation = color;
      scattered   = Ray(rec.p, Vec3(0, 0, 0));
      return true;
   }

 public:
   Color color;
};

//==============================================================================
// SHOW NORMALS — debug normal visualisation, path terminating
//==============================================================================

class ShowNormals : public Material
{
 public:
   explicit ShowNormals(const Color &a) : albedo(a) {}

   bool scatter(const Ray &, const Hit_record &rec, Color &attenuation, Ray &scattered) const override
   {
      attenuation = 0.5 * (rec.normal + Vec3_ONES);
      scattered   = Ray(rec.p, Vec3_ZEROES);
      return true;
   }

 public:
   Color albedo;
};

//==============================================================================
// LAMBERTIAN — diffuse material with full MIS support
//==============================================================================

class Lambertian : public Material
{
 public:
   explicit Lambertian(const Color &a) : albedo(a) {}

   // Legacy interface
   bool scatter(const Ray &, const Hit_record &rec, Color &attenuation, Ray &scattered) const override
   {
      Vec3 dir = rec.normal + Vec3::random_in_unit_sphere();
      if (dir.near_zero())
         dir = rec.normal;
      scattered   = Ray(rec.p, dir);
      attenuation = albedo;
      return true;
   }

   // MIS interface — cosine-weighted hemisphere sampling
   bool scatter_mis(const Ray &, const Hit_record &rec, ScatterRecord &srec) const override
   {
      // Build ONB from normal
      Vec3 w = unit_vector(rec.normal);
      Vec3 a = (std::fabs(w.x()) > 0.9) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
      Vec3 v = unit_vector(cross(w, a));
      Vec3 u = cross(v, w);

      // Cosine-weighted hemisphere (Malley's method)
      double u1  = RndGen::random_double();
      double u2  = RndGen::random_double();
      double r   = std::sqrt(u1);
      // Note: CPU materials use double-precision constants (M_PI). The corresponding
      // GPU implementation uses CUDART_PI_F (float) for the same math.
      double phi = 2.0 * M_PI * u2;

      Vec3 dir = r * std::cos(phi) * u + r * std::sin(phi) * v + std::sqrt(std::max(0.0, 1.0 - u1)) * w;
      dir       = unit_vector(dir);

      srec.scattered   = Ray(rec.p, dir);
      srec.bsdf_value  = albedo / M_PI;
      srec.pdf         = std::max(0.0, dot(dir, rec.normal)) / M_PI;
      srec.is_specular = false;
      return true;
   }

   Color eval_bsdf(const Ray &, const Hit_record &, const Vec3 &) const override { return albedo / M_PI; }

   double scatter_pdf(const Ray &, const Hit_record &rec, const Vec3 &wi) const override
   {
      double cosine = dot(unit_vector(wi), rec.normal);
      return std::max(0.0, cosine / M_PI);
   }

 public:
   Color albedo;
};

//==============================================================================
// LIGHT — emissive area light (no scattering)
//==============================================================================

class Light : public Material
{
 public:
   explicit Light(const Color &emission) : emission_color(emission) {}

   bool scatter(const Ray &, const Hit_record &, Color &, Ray &) const override { return false; }

   Color emitted(const Hit_record &) const override { return emission_color; }

   bool scatter_mis(const Ray &, const Hit_record &, ScatterRecord &) const override { return false; }

 public:
   Color emission_color;
};

//==============================================================================
// THIN-FILM — soap-bubble iridescence (specular, skips MIS)
//==============================================================================

/**
 * @brief Thin-film interference material (soap bubbles, oil slicks)
 *
 * Uses an analytic RGB approximation of thin-film interference.
 * The phase difference from the film causes constructive/destructive
 * interference at different wavelengths, producing iridescent colors.
 *
 * Reference wavelengths for RGB: R=650nm, G=550nm, B=450nm
 */
class ThinFilm : public Material
{
 public:
   ThinFilm(float thickness, float film_ior, float exterior_ior = 1.0f)
       : film_thickness(thickness), film_ior(film_ior), exterior_ior(exterior_ior)
   {
   }

   bool scatter(const Ray &r_in, const Hit_record &rec, Color &attenuation, Ray &scattered) const override
   {
      Vec3 unit_dir   = unit_vector(r_in.direction());
      double cos_i    = std::fmin(dot(-unit_dir, rec.normal), 1.0);
      double sin_i    = std::sqrt(1.0 - cos_i * cos_i);
      double sin_t    = (exterior_ior / film_ior) * sin_i;
      double cos_t    = std::sqrt(std::fmax(0.0, 1.0 - sin_t * sin_t));
      double wls[3]   = {650.0, 550.0, 450.0};
      double R[3];

      for (int ch = 0; ch < 3; ch++)
      {
         double opd   = 2.0 * film_ior * film_thickness * cos_t;
         double delta = 2.0 * M_PI * opd / wls[ch];
         double r01   = (exterior_ior - film_ior) / (exterior_ior + film_ior);
         double R01   = r01 * r01;
         double R12   = R01;
         double cd    = std::cos(delta);
         double num   = R01 + R12 + 2.0 * std::sqrt(R01 * R12) * cd;
         double den   = 1.0 + R01 * R12 + 2.0 * std::sqrt(R01 * R12) * cd;
         R[ch]        = num / den;
      }

      Vec3 reflected = unit_dir - 2.0 * dot(unit_dir, rec.normal) * rec.normal;
      scattered      = Ray(rec.p, reflected);
      attenuation    = Color(R[0], R[1], R[2]);
      return true;
   }

   bool scatter_mis(const Ray &r_in, const Hit_record &rec, ScatterRecord &srec) const override
   {
      Color attenuation;
      Ray scattered;
      if (!scatter(r_in, rec, attenuation, scattered))
         return false;
      srec.scattered   = scattered;
      srec.bsdf_value  = attenuation;
      srec.pdf         = 1.0;
      srec.is_specular = true;
      return true;
   }

 public:
   float film_thickness;
   float film_ior;
   float exterior_ior;
};
