#pragma once

#include "color.hpp"
#include "hittable.hpp"
#include "ray.hpp"
#include "vec3.hpp"

class Material
{
 public:
   virtual ~Material() = default;

   virtual bool scatter(const Ray &r_in, const Hit_record &rec, Color &attenuation, Ray &scattered) const
   {
      return false;
   }
};

class Constant : public Material
{
 public:
   Constant(const Color &a) : color(a) {}

   virtual bool scatter(const Ray &r_in, const Hit_record &rec, Color &attenuation, Ray &scattered) const override
   {
      attenuation = color;
      scattered = Ray(rec.p, Vec3(0, 0, 0)); // No scattering, the ray is absorbed
      return true;
   }

 public:
   Color color;
};

class ShowNormals : public Material
{
 public:
   ShowNormals(const Color &a) : albedo(a) {}

   virtual bool scatter(const Ray &r_in, const Hit_record &rec, Color &attenuation, Ray &scattered) const override
   {
      attenuation = 0.5 * (rec.normal + Vec3_ONES);
      scattered = Ray(rec.p, Vec3_ZEROES); // No scattering
      return true;
   }

 public:
   Color albedo;
};

class Lambertian : public Material
{
 public:
   Lambertian(const Color &a) : albedo(a) {}

   virtual bool scatter(const Ray &r_in, const Hit_record &rec, Color &attenuation, Ray &scattered) const override
   {
      Vec3 scatter_direction = rec.normal + Vec3::random_in_unit_sphere();

      // Catch degenerate scatter direction
      if (scatter_direction.near_zero())
         scatter_direction = rec.normal;

      scattered = Ray(rec.p, scatter_direction);

      attenuation = albedo;
      return true;
   }

 public:
   Color albedo; // The amount of reflected light, 0 for no reflection, 1 for full reflection
};

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
       : film_thickness(thickness), film_ior(film_ior), exterior_ior(exterior_ior) {}

   virtual bool scatter(const Ray &r_in, const Hit_record &rec, Color &attenuation, Ray &scattered) const override
   {
      // Incident direction
      Vec3 unit_dir = unit_vector(r_in.direction());
      double cos_theta_i = fmin(dot(-unit_dir, rec.normal), 1.0);
      
      // Snell's law: angle inside the film
      double sin_theta_i = sqrt(1.0 - cos_theta_i * cos_theta_i);
      double sin_theta_t = (exterior_ior / film_ior) * sin_theta_i;
      double cos_theta_t = sqrt(fmax(0.0, 1.0 - sin_theta_t * sin_theta_t));
      
      // Compute thin-film reflectance for RGB reference wavelengths
      // R=650nm, G=550nm, B=450nm
      double wavelengths[3] = {650.0, 550.0, 450.0};
      double R[3];
      
      for (int ch = 0; ch < 3; ch++) {
         // Optical path difference (OPD) = 2 * n_film * d * cos(theta_t)
         double opd = 2.0 * film_ior * film_thickness * cos_theta_t;
         // Phase difference in radians
         double delta = 2.0 * M_PI * opd / wavelengths[ch];
         
         // Fresnel reflectance at air-film interface (unpolarized, Schlick approx)
         double r01 = (exterior_ior - film_ior) / (exterior_ior + film_ior);
         double R01 = r01 * r01;
         
         // Fresnel reflectance at film-air interface (symmetric for same medium)
         double R12 = R01;
         
         // Airy formula for total thin-film reflectance
         double cos_delta = cos(delta);
         double numerator = R01 + R12 + 2.0 * sqrt(R01 * R12) * cos_delta;
         double denominator = 1.0 + R01 * R12 + 2.0 * sqrt(R01 * R12) * cos_delta;
         R[ch] = numerator / denominator;
      }
      
      // Probabilistic reflection vs transmission (use average reflectance)
      double avg_R = (R[0] + R[1] + R[2]) / 3.0;
      
      // Reflect
      Vec3 reflected = unit_dir - 2.0 * dot(unit_dir, rec.normal) * rec.normal;
      scattered = Ray(rec.p, reflected);
      attenuation = Color(R[0], R[1], R[2]) * (1.0 / fmax(avg_R, 0.001));
      // Weight by 1/avg_R since we always reflect (importance sampling weight)
      // Multiply by avg_R probability → net attenuation = Color(R[0], R[1], R[2])
      attenuation = Color(R[0], R[1], R[2]);
      return true;
   }

 public:
   float film_thickness;  // nm
   float film_ior;
   float exterior_ior;
};
