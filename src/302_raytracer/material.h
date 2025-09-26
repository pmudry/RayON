#pragma once

#include "color.h"
#include "hittable.h"
#include "ray.h"

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
   Constant(const Color &a) : albedo(a) {}

   virtual bool scatter(const Ray &r_in, const Hit_record &rec, Color &attenuation, Ray &scattered) const override
   {            
      attenuation = albedo;
      scattered = Ray(rec.p, Vec3(0,0,0)); // No scattering, the ray is absorbed
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
      Vec3 scatter_direction = rec.normal + Vec3::random_in_hemisphere(rec.normal);

      // Catch degenerate scatter direction
      if (scatter_direction.near_zero())
         scatter_direction = rec.normal;

      scattered = Ray(rec.p, scatter_direction);

      // If an object is white, it reflects all light (attenuation=1), black object absorbs all light (attenuation=0)
      attenuation = albedo;
      return true;
   }

 public:
   Color albedo;
};

class Metal : public Material
{
 public:
   Metal(const Color &a) : albedo(a) {}

   virtual bool scatter(const Ray &r_in, const Hit_record &rec, Color &attenuation, Ray &scattered) const override
   {
      Vec3 reflected = reflect(r_in.direction(), rec.normal);
      scattered = Ray(rec.p, reflected);
      attenuation = albedo;
      return true;
   }

 public:
   Vec3 reflect(const Vec3 &v, const Vec3 &n) const { return v - 2 * dot(v, n) * n; }

   Color albedo;
};