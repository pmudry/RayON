#pragma once

#include "hittable.h"
#include "utils.h"

// This one was completely written using vibe coding, no human involvement at all
class Cube : public Hittable
{
 public:
   Cube(const Point3 &center, double side_length, const Vec3 &rotation = Vec3(0, 0, 0))
       : center(center), side_length(std::fmax(0, side_length)), rotation(rotation)
   {
      compute_rotation_matrix();                  
   }

   // Set rotation in degrees around x, y, z axes
   void set_rotation(const Vec3 &rot)
   {
      rotation = rot;
      compute_rotation_matrix();
   }

 private:
   Point3 center;
   double side_length;   
   Vec3 rotation; // in degrees
   
   // Rotation matrix (3x3)
   double rot_matrix[3][3];

   // Compute rotation matrix from Euler angles (XYZ order)
   void compute_rotation_matrix()
   {
      double rx = rotation.x() * PI / 180.0;
      double ry = rotation.y() * PI / 180.0;
      double rz = rotation.z() * PI / 180.0;
      double cx = cos(rx), sx = sin(rx);
      double cy = cos(ry), sy = sin(ry);
      double cz = cos(rz), sz = sin(rz);

      // Rotation matrices for each axis
      double Rx[3][3] = {{1, 0, 0}, {0, cx, -sx}, {0, sx, cx}};
      double Ry[3][3] = {{cy, 0, sy}, {0, 1, 0}, {-sy, 0, cy}};
      double Rz[3][3] = {{cz, -sz, 0}, {sz, cz, 0}, {0, 0, 1}};

      // Combined rotation: Rz * Ry * Rx
      double Rzy[3][3];
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
         {
            Rzy[i][j] = 0;
            for (int k = 0; k < 3; ++k)
               Rzy[i][j] += Rz[i][k] * Ry[k][j];
         }
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
         {
            rot_matrix[i][j] = 0;
            for (int k = 0; k < 3; ++k)
               rot_matrix[i][j] += Rzy[i][k] * Rx[k][j];
         }
   }

   // Apply rotation matrix to a vector
   Vec3 rotate_point(const Vec3 &p, bool inverse = false) const
   {
      double m[3][3];
      if (!inverse)
      {
         for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
               m[i][j] = rot_matrix[i][j];
      }
      else
      {
         // Transpose for inverse rotation
         for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
               m[i][j] = rot_matrix[j][i];
      }
      double x = m[0][0] * p.x() + m[0][1] * p.y() + m[0][2] * p.z();
      double y = m[1][0] * p.x() + m[1][1] * p.y() + m[1][2] * p.z();
      double z = m[2][0] * p.x() + m[2][1] * p.y() + m[2][2] * p.z();
      return Vec3(x, y, z);
   }

   /**
    * @brief Calculates the intersection point of a ray with a cube
    *
    * This function determines if and where a ray intersects with an axis-aligned cube
    * centered at `center` with edge length `side_length`.
    *
    * @param center The center point of the cube in 3D space
    * @param side_length The length of the cube's sides (must be positive)
    * @param r The ray to test for intersection
    *
    * @return True if the ray hits the cube, false otherwise
    *         The intersection point and normal are stored in `rec`.
    */
   bool hit(const Ray &r, Interval ray_t, Hit_record &rec) const override
   {
      // Transform ray to cube's local (rotated) space
      Vec3 local_origin = rotate_point(r.origin() - center, true);
      Vec3 local_dir = rotate_point(r.direction(), true);
      Ray local_ray(local_origin, local_dir);

      double half = side_length / 2.0;
      Interval x_interval(-half, half);
      Interval y_interval(-half, half);
      Interval z_interval(-half, half);

      double tmin = ray_t.min;
      double tmax = ray_t.max;

      // X slab
      if (std::abs(local_ray.direction().x()) > 1e-8)
      {
         double tx1 = (x_interval.min - local_ray.origin().x()) / local_ray.direction().x();
         double tx2 = (x_interval.max - local_ray.origin().x()) / local_ray.direction().x();
         tmin = std::max(tmin, std::min(tx1, tx2));
         tmax = std::min(tmax, std::max(tx1, tx2));
      }
      else if (local_ray.origin().x() < x_interval.min || local_ray.origin().x() > x_interval.max)
      {
         return false;
      }
      // Y slab
      if (std::abs(local_ray.direction().y()) > 1e-8)
      {
         double ty1 = (y_interval.min - local_ray.origin().y()) / local_ray.direction().y();
         double ty2 = (y_interval.max - local_ray.origin().y()) / local_ray.direction().y();
         tmin = std::max(tmin, std::min(ty1, ty2));
         tmax = std::min(tmax, std::max(ty1, ty2));
      }
      else if (local_ray.origin().y() < y_interval.min || local_ray.origin().y() > y_interval.max)
      {
         return false;
      }
      // Z slab
      if (std::abs(local_ray.direction().z()) > 1e-8)
      {
         double tz1 = (z_interval.min - local_ray.origin().z()) / local_ray.direction().z();
         double tz2 = (z_interval.max - local_ray.origin().z()) / local_ray.direction().z();
         tmin = std::max(tmin, std::min(tz1, tz2));
         tmax = std::min(tmax, std::max(tz1, tz2));
      }
      else if (local_ray.origin().z() < z_interval.min || local_ray.origin().z() > z_interval.max)
      {
         return false;
      }

      if (tmax < tmin || tmax < ray_t.min || tmin > ray_t.max)
         return false;

      rec.t = tmin;
      // Compute hit point in local space
      Vec3 local_hit = local_ray.at(rec.t);
      // Transform hit point back to world space
      rec.p = rotate_point(local_hit, false) + center;
      // Compute normal in local space
      Vec3 normal(0, 0, 0);
      double eps = 1e-6;
      if (std::abs(local_hit.x() - x_interval.min) < eps)
         normal = Vec3(-1, 0, 0);
      else if (std::abs(local_hit.x() - x_interval.max) < eps)
         normal = Vec3(1, 0, 0);
      else if (std::abs(local_hit.y() - y_interval.min) < eps)
         normal = Vec3(0, -1, 0);
      else if (std::abs(local_hit.y() - y_interval.max) < eps)
         normal = Vec3(0, 1, 0);
      else if (std::abs(local_hit.z() - z_interval.min) < eps)
         normal = Vec3(0, 0, -1);
      else if (std::abs(local_hit.z() - z_interval.max) < eps)
         normal = Vec3(0, 0, 1);
      // Rotate normal back to world space
      rec.normal = rotate_point(normal, false);
      return true;
   }
   
};
