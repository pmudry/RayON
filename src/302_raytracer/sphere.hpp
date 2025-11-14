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
#include "utils.hpp"
#include "material.hpp"

class Sphere : public Hittable{
 public:
   Sphere(const Point3 &center, double radius, shared_ptr<Material> mat) : center(center), radius(std::fmax(0, radius)), mat(mat) {}

   /**
    * @brief Calculates the intersection point of a ray with a sphere
    *
    * This function determines if and where a ray intersects with a sphere by
    * solving the quadratic equation formed by substituting the ray equation
    * into the sphere equation.
    *
    * **Mathematical Background:**
    * - Sphere equation: `(C−P)⋅(C−P) = r²` where P is a point on the sphere
    * - Ray equation: `P = O + t*d` where O is origin, d is direction, t is
    * distance
    * - We then solve for t by substituting the ray equation into the sphere
    * equation
    * - Substitution yields quadratic: `at² + bt + c = 0` where:
    *   - `a = d⋅d` (direction vector dot product)
    *   - `b = −2*d⋅(C−O)` (relates direction to center-origin vector)
    *   - `c = (C−O)⋅(C−O) − r²` (distance from origin to center minus radius
    * squared) 
    *
    * @param center The center point of the sphere in 3D space
    * @param radius The radius of the sphere (must be positive)
    * @param r The ray to test for intersection
    *
    * @return The parameter t for the intersection point along the ray:
    *     - Returns `-1.0` if no intersection occurs (discriminant < 0)
    *     - Returns the farther intersection point when two intersections exist
    *     - The actual intersection point can be computed as `r.origin() + t *
    * r.direction()`
    *
    * @note When the discriminant is non-negative, this function returns the
    * larger t value, corresponding to the exit point of the ray from the sphere
    */
   bool hit(const Ray &r, Interval ray_t, Hit_record &rec) const override
   {
      Vec3 oc = center - r.origin();
      auto a = r.direction().length_squared();
      auto h = dot(r.direction(), oc);
      auto c = oc.length_squared() - radius * radius;

      auto discriminant = h * h - a * c;
      if (discriminant < 0)
         return false;

      auto sqrtd = std::sqrt(discriminant);

      // Find the nearest root that lies in the acceptable range.
      auto root = (h - sqrtd) / a;

      if (ray_t.surrounds(root) == false)
      {
         root = (h + sqrtd) / a;
         if (ray_t.surrounds(root) == false)
            return false;
      }

      rec.t = root;
      rec.p = r.at(rec.t);
      rec.normal = (rec.p - center) / radius;
      rec.mat_ptr = mat;

      return true;
   }

 private:
   Point3 center;
   double radius;
   shared_ptr<Material> mat;
};