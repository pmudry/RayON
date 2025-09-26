#pragma once

#include "rnd_gen.h"
#include <cmath>
#include <iostream>

class Vec3
{
 public:
   double e[3];

   Vec3() : e{0, 0, 0} {}
   Vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

   double x() const { return e[0]; }
   double y() const { return e[1]; }
   double z() const { return e[2]; }

   Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
   double operator[](int i) const { return e[i]; }
   double &operator[](int i) { return e[i]; }

   Vec3 &operator+=(const Vec3 &v)
   {
      e[0] += v.e[0];
      e[1] += v.e[1];
      e[2] += v.e[2];
      return *this;
   }

   Vec3 &operator*=(double t)
   {
      e[0] *= t;
      e[1] *= t;
      e[2] *= t;
      return *this;
   }

   Vec3 &operator/=(double t) { return *this *= 1 / t; }

   double length() const { return std::sqrt(length_squared()); }

   double length_squared() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }

   bool near_zero() const
   {
      // Return true if the vector is close to zero in all dimensions.
      auto s = 1e-8;
      return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
   }

   static Vec3 random() { return Vec3(RndGen::random_double(), RndGen::random_double(), RndGen::random_double()); }

   static Vec3 random(double min, double max)
   {
      return Vec3(RndGen::random_double(min, max), RndGen::random_double(min, max), RndGen::random_double(min, max));
   }

   friend Vec3 unit_vector(const Vec3 &v);
   friend double dot(const Vec3 &u, const Vec3 &v);

   static Vec3 random_unit_vector() { return unit_vector(random(-1, 1)); }

   static Vec3 random_in_unit_sphere()
   {
      while (true)
      {
         auto p = Vec3::random(-1, 1);
         auto l = p.length_squared();

         // Reject the vectors outside the unit sphere or
         // too close to zero length
         if (l >= 1 || l < 1e-160)
            continue;
         return p;
      }
   }

   static Vec3 random_in_hemisphere(const Vec3 &normal)
   {
      Vec3 in_unit_sphere = random_in_unit_sphere();
      if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
         return in_unit_sphere;
      else
         return -in_unit_sphere;
   }
};

// Point3 is just an alias for vec3, but useful for geometric clarity in the code.
using Point3 = Vec3;

inline std::ostream &operator<<(std::ostream &out, const Vec3 &v)
{
   return out << "vec: " << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline Vec3 operator+(const Vec3 &u, const Vec3 &v) { return Vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]); }

inline Vec3 operator-(const Vec3 &u, const Vec3 &v) { return Vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]); }

inline Vec3 operator*(const Vec3 &u, const Vec3 &v) { return Vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]); }

inline Vec3 operator*(double t, const Vec3 &v) { return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]); }

inline Vec3 operator*(const Vec3 &v, double t) { return t * v; }

inline Vec3 operator/(const Vec3 &v, double t) { return (1 / t) * v; }

inline double dot(const Vec3 &u, const Vec3 &v) { return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2]; }

inline Vec3 cross(const Vec3 &u, const Vec3 &v)
{
   return Vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1], u.e[2] * v.e[0] - u.e[0] * v.e[2], u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline Vec3 unit_vector(const Vec3 &v) { return v / v.length(); }