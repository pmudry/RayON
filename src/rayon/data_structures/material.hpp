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
