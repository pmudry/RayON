/**
 * @class Ray
 * @brief Represents a ray in 3D space, defined by an origin point and a direction vector.
 *
 * The Ray class provides methods to access the origin and direction of the ray,
 * as well as to compute a point along the ray at a given parameter `t`.
 */
#pragma once

#include "vec3.h"

class Ray
{
 public:
   Ray() {}
   Ray(const Point3 &origin, const Vec3 &direction) : orig(origin), dir(direction) {}

   const Point3 &origin() const { return orig; }
   const Vec3 &direction() const { return dir; }

   Point3 at(double t) const { return orig + t * dir; }

 private:
   Point3 orig;
   Vec3 dir;
};