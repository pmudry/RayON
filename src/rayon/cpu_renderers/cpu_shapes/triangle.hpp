/**
 * @file triangle.hpp
 * @brief CPU-side triangle intersection using Möller–Trumbore algorithm
 */
#pragma once

#include "hittable.hpp"
#include "material.hpp"

class TriangleShape : public Hittable
{
 public:
   TriangleShape(const Point3 &v0, const Point3 &v1, const Point3 &v2, shared_ptr<Material> mat)
       : v0(v0), v1(v1), v2(v2), has_normals(false), mat(mat)
   {
   }

   TriangleShape(const Point3 &v0, const Point3 &v1, const Point3 &v2,
                 const Vec3 &n0, const Vec3 &n1, const Vec3 &n2, shared_ptr<Material> mat)
       : v0(v0), v1(v1), v2(v2), n0(n0), n1(n1), n2(n2), has_normals(true), mat(mat)
   {
   }

   bool hit(const Ray &r, Interval ray_t, Hit_record &rec) const override
   {
      Vec3 edge1 = v1 - v0;
      Vec3 edge2 = v2 - v0;
      Vec3 h = cross(r.direction(), edge2);
      double a = dot(edge1, h);

      if (fabs(a) < 1e-8)
         return false;

      double f = 1.0 / a;
      Vec3 s = r.origin() - v0;
      double u = f * dot(s, h);
      if (u < 0.0 || u > 1.0)
         return false;

      Vec3 q = cross(s, edge1);
      double v = f * dot(r.direction(), q);
      if (v < 0.0 || u + v > 1.0)
         return false;

      double t = f * dot(edge2, q);
      if (!ray_t.surrounds(t))
         return false;

      rec.t = t;
      rec.p = r.at(t);
      rec.mat_ptr = mat;

      // Always use geometric normal for front-face determination
      Vec3 geo_normal = unit_vector(cross(edge1, edge2));

      Vec3 shading_normal;
      if (has_normals)
      {
         double w = 1.0 - u - v;
         shading_normal = unit_vector(w * n0 + u * n1 + v * n2);
         // Ensure smooth normal is on the same hemisphere as geometric normal
         if (dot(shading_normal, geo_normal) < 0.0)
            shading_normal = -shading_normal;
      }
      else
      {
         shading_normal = geo_normal;
      }

      rec.frontFacing = dot(r.direction(), geo_normal) < 0;
      rec.normal = rec.frontFacing ? shading_normal : -shading_normal;
      return true;
   }

 private:
   Point3 v0, v1, v2;
   Vec3 n0, n1, n2;
   bool has_normals;
   shared_ptr<Material> mat;
};
