#pragma once

#include "hittable.hpp"
#include "utils.hpp"
#include "color.hpp"
#include "material.hpp"
#include <memory>

/**
 * Rectangle class that can serve as both a surface and an area light
 */
class Rectangle : public Hittable
{
 public:
   Rectangle(const Point3 &corner, const Vec3 &u, const Vec3 &v, shared_ptr<Material> mat = nullptr)
       : corner(corner), u(u), v(v), mat_ptr(mat), is_light(false), light_color(Color(1, 1, 1)), light_intensity(1.0)
   {
      normal = unit_vector(cross(u, v));
      area = u.length() * v.length();
   }

   // Constructor for area light
   Rectangle(const Point3 &corner, const Vec3 &u, const Vec3 &v, const Color &light_col, double intensity)
       : corner(corner), u(u), v(v), mat_ptr(nullptr), is_light(true), light_color(light_col), light_intensity(intensity)
   {
      normal = unit_vector(cross(u, v));
      area = u.length() * v.length();
   }

   bool hit(const Ray &r, Interval ray_t, Hit_record &rec) const override
   {
      // Calculate the plane equation: normal · (P - corner) = 0
      // For ray P = origin + t * direction
      // normal · (origin + t * direction - corner) = 0
      // Solve for t: t = normal · (corner - origin) / (normal · direction)

      double denom = dot(normal, r.direction());

      // If denominator is close to 0, ray is parallel to plane
      if (fabs(denom) < 1e-8)
         return false;

      double t = dot(normal, corner - r.origin()) / denom;

      if (!ray_t.surrounds(t))
         return false;

      // Find the intersection point
      Point3 intersection = r.at(t);

      // Check if intersection is within rectangle bounds
      Vec3 p = intersection - corner;

      // Project onto rectangle's coordinate system
      double alpha = dot(p, u) / dot(u, u);
      double beta = dot(p, v) / dot(v, v);

      // Check if point is inside rectangle (0 <= alpha <= 1, 0 <= beta <= 1)
      if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0)
         return false;

      // We have a valid hit
      rec.t = t;
      rec.p = intersection;
      rec.set_face_normal(r, normal);
      rec.mat_ptr = mat_ptr;

      return true;
   }

   // Sample a random point on the rectangle for area lighting
   Point3 sample_point() const
   {
      double alpha = RndGen::random_double();
      double beta = RndGen::random_double();
      return corner + alpha * u + beta * v;
   }

   // Get the area of the rectangle
   double get_area() const { return area; }

   // Check if this rectangle is a light source
   bool is_area_light() const { return is_light; }

   // Get light properties
   Color get_light_color() const { return light_color; }
   double get_light_intensity() const { return light_intensity; }

 private:
   Point3 corner; // One corner of the rectangle
   Vec3 u, v;     // Two edges of the rectangle from corner
   Vec3 normal;   // Normal vector to the rectangle
   double area;   // Area of the rectangle
   
   shared_ptr<Material> mat_ptr; // Material pointer

   // Light properties
   bool is_light;
   Color light_color;
   double light_intensity;
};