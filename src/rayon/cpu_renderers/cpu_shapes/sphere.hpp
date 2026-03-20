/**
 * @class Sphere
 * @brief Represents a 3D sphere that can be intersected by rays in a ray tracer.
 *
 * The `Sphere` class is a concrete implementation of the `Hittable` interface,
 * representing a sphere in 3D space. It provides functionality to calculate
 * intersections between a ray and the sphere, which is a fundamental operation
 * in ray tracing.
 *
 * **Mathematical Representation:**
 * - A sphere is defined by its center point and radius.
 * - The intersection of a ray with the sphere is determined by solving a
 *   quadratic equation derived from the sphere and ray equations.
 *
 * **Usage:**
 * - The `hit` method determines if a ray intersects the sphere and provides
 *   details about the intersection point, normal, and other properties.
 *
 */
#pragma once

#include "hittable.hpp"
#include "material.hpp"

#include <cmath>

class Sphere : public Hittable
{
 public:
   Sphere(const Point3 &center, double radius, shared_ptr<Material> mat)
       : center(center), radius(std::fmax(0, radius)), mat(mat)
   {
   }

   bool hit(const Ray &r, Interval ray_t, Hit_record &rec) const override
   {
      Vec3 oc = center - r.origin();
      auto a  = r.direction().length_squared();
      auto h  = dot(r.direction(), oc);
      auto c  = oc.length_squared() - radius * radius;

      auto discriminant = h * h - a * c;
      if (discriminant < 0)
         return false;

      auto sqrtd = std::sqrt(discriminant);

      auto root = (h - sqrtd) / a;
      if (ray_t.surrounds(root) == false)
      {
         root = (h + sqrtd) / a;
         if (ray_t.surrounds(root) == false)
            return false;
      }

      rec.t      = root;
      rec.p      = r.at(rec.t);
      rec.normal = (rec.p - center) / radius;
      rec.mat_ptr = mat;

      return true;
   }

   // --- NEE emitter sampling ---

   double pdf_value(const Point3 &origin, const Vec3 &direction) const override
   {
      Vec3   to_center = center - origin;
      double dist_sq   = to_center.length_squared();
      Vec3   dir_unit  = unit_vector(direction);

      // If origin is inside the sphere, the full sphere surface is visible.
      // Any direction hits the sphere, so use the uniform full-sphere PDF.
      if (dist_sq <= radius * radius)
         return 1.0 / (4.0 * M_PI);

      double dist          = std::sqrt(dist_sq);
      double cos_theta_max = std::sqrt(std::max(0.0, 1.0 - (radius * radius) / dist_sq));

      // Return 0 when the direction lies outside the visible cone of the sphere.
      double cos_theta = dot(dir_unit, to_center) / dist;
      if (cos_theta < cos_theta_max)
         return 0.0;

      double solid_angle = 2.0 * M_PI * (1.0 - cos_theta_max);
      return (solid_angle > 0.0) ? (1.0 / solid_angle) : 0.0;
   }

   Vec3 random_direction(const Point3 &origin) const override
   {
      Vec3   to_center = center - origin;
      double dist_sq   = to_center.length_squared();

      // Inside the sphere every direction hits the surface — sample uniformly
      // over the full sphere to match the 1/(4π) PDF returned by pdf_value().
      if (dist_sq <= radius * radius)
         return unit_vector(Vec3::random_in_unit_sphere());

      double dist           = std::sqrt(dist_sq);
      double cos_theta_max  = std::sqrt(std::max(0.0, 1.0 - (radius * radius) / dist_sq));

      // Build ONB with w pointing toward sphere center
      Vec3 w = to_center / dist;
      Vec3 a = (std::fabs(w.x()) > 0.9) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
      Vec3 v = unit_vector(cross(w, a));
      Vec3 u = cross(v, w);

      // Sample uniformly in the visible cone
      double u1  = RndGen::random_double();
      double phi = 2.0 * M_PI * RndGen::random_double();
      double cos_theta = 1.0 - u1 * (1.0 - cos_theta_max);
      double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));

      return unit_vector(sin_theta * std::cos(phi) * u + sin_theta * std::sin(phi) * v + cos_theta * w);
   }

 private:
   Point3 center;
   double radius;
   shared_ptr<Material> mat;
};
