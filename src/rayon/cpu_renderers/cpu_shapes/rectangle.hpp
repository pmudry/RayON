#pragma once

#include "color.hpp"
#include "hittable.hpp"
#include "material.hpp"
#include "vec3.hpp"
#include <memory>

/**
 * Rectangle class that can serve as both a surface and an area light
 */
class Rectangle : public Hittable
{
 public:
   Rectangle(const Point3 &corner, const Vec3 &u, const Vec3 &v, shared_ptr<Material> mat = nullptr)
       : corner(corner), u(u), v(v), mat_ptr(mat), is_light(false), light_color(Vec3_ONES), light_intensity(1.0)
   {
      normal = unit_vector(cross(u, v));
      area   = u.length() * v.length();
   }

   Rectangle(const Point3 &corner, const Vec3 &u, const Vec3 &v, const Color &light_col, double intensity)
       : corner(corner), u(u), v(v), mat_ptr(nullptr), is_light(true), light_color(light_col),
         light_intensity(intensity)
   {
      normal = unit_vector(cross(u, v));
      area   = u.length() * v.length();
   }

   bool hit(const Ray &r, Interval ray_t, Hit_record &rec) const override
   {
      double denom = dot(normal, r.direction());
      if (fabs(denom) < 1e-8)
         return false;

      double t = dot(normal, corner - r.origin()) / denom;
      if (!ray_t.surrounds(t))
         return false;

      Point3 intersection = r.at(t);
      Vec3   p            = intersection - corner;

      double alpha = dot(p, u) / dot(u, u);
      double beta  = dot(p, v) / dot(v, v);

      if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0)
         return false;

      rec.t = t;
      rec.p = intersection;
      rec.set_face_normal(r, normal);
      rec.mat_ptr = mat_ptr;
      return true;
   }

   // --- NEE emitter sampling ---

   double pdf_value(const Point3 &origin, const Vec3 &direction) const override
   {
      // Cast a ray and check if it hits this rectangle
      Hit_record rec;
      if (!this->hit(Ray(origin, direction), Interval(0.0001, 1e10), rec))
         return 0.0;

      double distance_sq = rec.t * rec.t * direction.length_squared();
      double cosine      = std::fabs(dot(direction, rec.normal) / direction.length());
      if (cosine < 1e-8)
         return 0.0;
      return distance_sq / (cosine * area);
   }

   Vec3 random_direction(const Point3 &origin) const override
   {
      Point3 p = sample_point();
      return unit_vector(p - origin);
   }

   Point3 sample_point() const
   {
      double alpha = RndGen::random_double();
      double beta  = RndGen::random_double();
      return corner + alpha * u + beta * v;
   }

   double get_area() const { return area; }
   bool is_area_light() const { return is_light; }
   Color get_light_color() const { return light_color; }
   double get_light_intensity() const { return light_intensity; }

 private:
   Point3 corner;
   Vec3   u, v;
   Vec3   normal;
   double area;

   shared_ptr<Material> mat_ptr;

   bool  is_light;
   Color light_color;
   double light_intensity;
};
