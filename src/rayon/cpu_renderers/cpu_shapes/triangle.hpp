#pragma once

#include "../../data_structures/hittable.hpp"
#include "../../data_structures/material.hpp"
#include "../../data_structures/vec3.hpp"
#include <cmath>

class TriangleShape : public Hittable
{
 public:
   TriangleShape(const Vec3 &_v0, const Vec3 &_v1, const Vec3 &_v2, shared_ptr<Material> _mat)
       : v0(_v0), v1(_v1), v2(_v2), n0(0,0,0), n1(0,0,0), n2(0,0,0), has_normals(false), mat_ptr(_mat)
   {
   }

   TriangleShape(const Vec3 &_v0, const Vec3 &_v1, const Vec3 &_v2, const Vec3 &_n0, const Vec3 &_n1, const Vec3 &_n2,
            shared_ptr<Material> _mat)
       : v0(_v0), v1(_v1), v2(_v2), n0(_n0), n1(_n1), n2(_n2), has_normals(true), mat_ptr(_mat)
   {
   }

   /**
    * @brief Möller–Trumbore intersection algorithm (or barycentric method ;)
    */
   bool hit(const Ray &r, Interval ray_t, Hit_record &rec) const override
   {
      Vec3 e1 = v1 - v0;
      Vec3 e2 = v2 - v0;

      Vec3 p = cross(r.direction(), e2);

      double det = dot(e1, p);

      if (std::fabs(det) < 1e-8)
         return false;

      double inv_det = 1.0 / det;

      Vec3 t_vec = r.origin() - v0;

      double u = dot(t_vec, p) * inv_det;

      if (u < 0.0 || u > 1.0)
         return false;

      Vec3 q = cross(t_vec, e1);

      double v = dot(r.direction(), q) * inv_det;

      if (v < 0.0 || u + v > 1.0)
         return false;

      double t = dot(e2, q) * inv_det;

      if (!ray_t.surrounds(t))
         return false;

      rec.t = t;
      rec.p = r.at(t);

      Vec3 outward_normal;
      if (has_normals)
      {
          double w = 1.0 - u - v;
          outward_normal = unit_vector(w * n0 + u * n1 + v * n2);
      }
      else
      {
          outward_normal = unit_vector(cross(e1, e2));
      }

      rec.set_face_normal(r, outward_normal);
      rec.mat_ptr = mat_ptr;

      return true;
   }

 public:
   Vec3 v0, v1, v2;
   Vec3 n0, n1, n2;
   bool has_normals;
   shared_ptr<Material> mat_ptr;
};
