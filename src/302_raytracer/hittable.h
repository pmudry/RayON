/**
 * @class Hit_record
 * @brief Represents the details of a ray-object intersection in a ray tracer.
 *
 * This class stores information about a ray's intersection with an object,
 * including the hit point, the surface normal at the hit point, the distance
 * along the ray where the hit occurred, and whether the hit was on the front
 * face of the object. It also provides functionality to set the surface normal
 * based on the ray's direction and the outward normal of the surface.
 *
 * @fn void Hit_record::set_face_normal(const ray &r, const Vec3 &outward_normal)
 * Sets the surface normal vector for the hit record. Determines whether the
 * ray hit the front face or the back face of the object and adjusts the normal
 * vector accordingly.
 *
 * @param r The ray that intersected the object.
 * @param outward_normal The outward normal vector of the surface at the hit
 *        point. This vector is assumed to have unit length.
 */
#pragma once

#include "utils.h"

class Hit_record
{
 public:
   Point3 p;         // The point where the ray hits the object
   Vec3 normal;      // The normal vector at the hit point
   double t;         // The ray distance at the hit point
   bool frontFacing; // True if the ray hits the front face of the object

   bool isMirror = false;

   void set_face_normal(const Ray &r, const Vec3 &outward_normal)
   {
      // Sets the hit record normal vector.
      // NOTE: the parameter `outward_normal` is assumed to have unit length.
      frontFacing = dot(r.direction(), outward_normal) < 0;
      normal = frontFacing ? outward_normal : -outward_normal;
   }
};

class Hittable
{
 public:
   virtual ~Hittable() = default;

   virtual bool hit(const Ray &r, Interval ray_t, Hit_record &rec) const = 0;
};