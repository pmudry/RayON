#pragma once
/**
 * @file sdf_primitives.cuh
 * @brief Ray-marched signed distance primitives shared by the CUDA and OptiX kernels.
 *
 * Distance functions after Íñigo Quilez (https://iquilezles.org/articles/distfunctions/), ported
 * from the legacy CPU renderer. A primitive is a shape type, a centre, three shape parameters and
 * an Euler rotation (X, Y, Z order, radians). Points are brought into the shape's local frame
 * (centre at the origin, rotation undone) before the distance is evaluated.
 *
 * sdf_march() sphere-traces a ray against one primitive: it first clips the ray to the shape's
 * bounding sphere, then steps by the signed distance until it is below the surface tolerance.
 * The normal is the central-difference gradient of the distance.
 *
 * Plain floats only, so both back ends can use it with their own vector types. The shape type
 * codes mirror Scene::SDFType (scene_description.hpp); keep the two in sync.
 */

#include <cuda_runtime.h>

enum SdfShape : int
{
   SDF_SHAPE_SPHERE = 0,
   SDF_SHAPE_BOX,
   SDF_SHAPE_TORUS,
   SDF_SHAPE_CAPSULE,
   SDF_SHAPE_CYLINDER,
   SDF_SHAPE_PLANE,
   SDF_SHAPE_MANDELBULB,
   SDF_SHAPE_DEATH_STAR,
   SDF_SHAPE_CUT_HOLLOW_SPHERE,
   SDF_SHAPE_OCTAHEDRON,
   SDF_SHAPE_PYRAMID,
   SDF_SHAPE_CUSTOM
};

/// One primitive, as the kernels see it
struct SdfPrim
{
   int   shape;
   float cx, cy, cz; ///< Centre (world space)
   float p0, p1, p2; ///< Shape parameters (meaning depends on the shape, see sdf_bound_radius)
   float rx, ry, rz; ///< Euler rotation in radians, applied in X, Y, Z order
};

constexpr int   SDF_MAX_STEPS     = 320;
constexpr float SDF_HIT_EPSILON   = 1e-4f; ///< Surface tolerance (scene units)
constexpr float SDF_NORMAL_EPS    = 2e-4f; ///< Central-difference step for the normal

__host__ __device__ __forceinline__ float sdf_clampf(float v, float lo, float hi) { return fminf(fmaxf(v, lo), hi); }
__host__ __device__ __forceinline__ float sdf_len3(float x, float y, float z) { return sqrtf(x * x + y * y + z * z); }
__host__ __device__ __forceinline__ float sdf_len2(float x, float y) { return sqrtf(x * x + y * y); }

/// Radius of a sphere around the centre that contains the shape whatever its rotation
/// (used for the bounding boxes on the host and the ray clipping in the kernels)
__host__ __device__ __forceinline__ float sdf_bound_radius(int shape, float p0, float p1, float p2)
{
   float r;
   switch (shape)
   {
   case SDF_SHAPE_SPHERE: r = p0; break;                                   // radius
   case SDF_SHAPE_BOX: r = sdf_len3(p0, p1, p2); break;                   // half extents
   case SDF_SHAPE_TORUS: r = p0 + p1; break;                               // major + minor radius
   case SDF_SHAPE_CAPSULE: r = p0 + 0.5f * p1; break;                      // radius, height (along y)
   case SDF_SHAPE_CYLINDER: r = sdf_len2(p0, 0.5f * p1); break;            // radius, height (along y)
   case SDF_SHAPE_MANDELBULB: r = 1.3f; break;                             // the set lives inside |z| < 1.3
   case SDF_SHAPE_DEATH_STAR: r = p0; break;                               // main radius
   case SDF_SHAPE_CUT_HOLLOW_SPHERE: r = p0 + p2; break;                   // radius + thickness
   case SDF_SHAPE_OCTAHEDRON: r = p0; break;                               // size
   case SDF_SHAPE_PYRAMID: r = sdf_len2(0.7072f, p0); break;               // base half-diagonal, height
   default: r = fmaxf(fmaxf(fabsf(p0), fabsf(p1)), fabsf(p2)); break;
   }
   return 1.05f * r + 1e-3f;
}

/// Signed distance of a point in the shape's local frame (centre at the origin, no rotation)
__device__ __forceinline__ float sdf_local(int shape, float p0, float p1, float p2, float x, float y, float z)
{
   switch (shape)
   {
   case SDF_SHAPE_SPHERE:
      return sdf_len3(x, y, z) - p0;

   case SDF_SHAPE_BOX:
   {
      const float qx = fabsf(x) - p0, qy = fabsf(y) - p1, qz = fabsf(z) - p2;
      const float outside = sdf_len3(fmaxf(qx, 0.0f), fmaxf(qy, 0.0f), fmaxf(qz, 0.0f));
      const float inside  = fminf(fmaxf(qx, fmaxf(qy, qz)), 0.0f);
      return outside + inside;
   }

   case SDF_SHAPE_TORUS:
   {
      const float qx = sdf_len2(x, z) - p0;
      return sdf_len2(qx, y) - p1;
   }

   case SDF_SHAPE_CAPSULE:
   {
      // Segment along y from -h/2 to +h/2, radius p0
      const float h  = 0.5f * p1;
      const float yy = y - sdf_clampf(y, -h, h);
      return sdf_len3(x, yy, z) - p0;
   }

   case SDF_SHAPE_CYLINDER:
   {
      const float d_xz     = sdf_len2(x, z) - p0;
      const float d_y      = fabsf(y) - 0.5f * p1;
      const float outside  = sdf_len2(fmaxf(d_xz, 0.0f), fmaxf(d_y, 0.0f));
      const float inside   = fminf(fmaxf(d_xz, d_y), 0.0f);
      return outside + inside;
   }

   case SDF_SHAPE_PLANE:
      return y; // horizontal plane through the centre

   case SDF_SHAPE_MANDELBULB:
   {
      const float power = p0 > 0.0f ? p0 : 8.0f;
      const int   iters = p1 > 0.0f ? (int)p1 : 15;
      float zx = x, zy = y, zz = z;
      float dr = 1.0f, r = 0.0f;
      for (int i = 0; i < iters; ++i)
      {
         r = sdf_len3(zx, zy, zz);
         if (r > 2.0f)
            break;
         float theta = acosf(zz / r);
         float phi   = atan2f(zy, zx);
         dr          = powf(r, power - 1.0f) * power * dr + 1.0f;
         const float zr = powf(r, power);
         theta *= power;
         phi *= power;
         const float st = sinf(theta);
         zx = zr * st * cosf(phi) + x;
         zy = zr * sinf(phi) * st + y;
         zz = zr * cosf(theta) + z;
      }
      return 0.5f * logf(fmaxf(r, 1e-6f)) * r / dr;
   }

   case SDF_SHAPE_DEATH_STAR:
   {
      // Sphere of radius ra with a sphere of radius rb cut out, its centre d away along +x
      const float ra = p0, rb = p1, d = p2;
      const float a   = (ra * ra - rb * rb + d * d) / (2.0f * d);
      const float b   = sqrtf(fmaxf(ra * ra - a * a, 0.0f));
      const float px  = x, py = sdf_len2(y, z);
      if (px * b - py * a > d * fmaxf(b - py, 0.0f))
         return sdf_len2(px - a, py - b);
      return fmaxf(sdf_len2(px, py) - ra, -(sdf_len2(px - d, py)) + rb);
   }

   case SDF_SHAPE_CUT_HOLLOW_SPHERE:
   {
      // Hollow sphere of radius r and thickness t, cut by the plane y = h (the cap is removed)
      const float r = p0, h = p1, t = p2;
      const float w  = sqrtf(fmaxf(r * r - h * h, 0.0f));
      const float qx = sdf_len2(x, z), qy = y;
      const float dist = (h * qx < w * qy) ? sdf_len2(qx - w, qy - h) : fabsf(sdf_len2(qx, qy) - r);
      return dist - t;
   }

   case SDF_SHAPE_OCTAHEDRON:
   {
      const float s  = p0;
      const float ax = fabsf(x), ay = fabsf(y), az = fabsf(z);
      const float m  = ax + ay + az - s;
      float qx, qy, qz;
      if (3.0f * ax < m)
         qx = ax, qy = ay, qz = az;
      else if (3.0f * ay < m)
         qx = ay, qy = az, qz = ax;
      else if (3.0f * az < m)
         qx = az, qy = ax, qz = ay;
      else
         return m * 0.57735027f;
      const float k = sdf_clampf(0.5f * (qz - qy + s), 0.0f, s);
      return sdf_len3(qx, qy - s + k, qz - k);
   }

   case SDF_SHAPE_PYRAMID:
   {
      // Square base of side 1 on the plane y = 0, apex at height h
      const float h  = p0;
      const float m2 = h * h + 0.25f;
      float px = fabsf(x), py = y, pz = fabsf(z);
      if (pz > px)
      {
         const float tmp = px;
         px = pz;
         pz = tmp;
      }
      px -= 0.5f;
      pz -= 0.5f;
      const float qx = pz, qy = h * py - 0.5f * px, qz = h * px + 0.5f * py;
      const float s  = fmaxf(-qx, 0.0f);
      const float t  = sdf_clampf((qy - 0.5f * pz) / (m2 + 0.25f), 0.0f, 1.0f);
      const float a  = m2 * (qx + s) * (qx + s) + qy * qy;
      const float b  = m2 * (qx + 0.5f * t) * (qx + 0.5f * t) + (qy - m2 * t) * (qy - m2 * t);
      const float d2 = (fminf(qy, -qx * m2 - qy * 0.5f) > 0.0f) ? 0.0f : fminf(a, b);
      const float res = sqrtf((d2 + qz * qz) / m2);
      return res * ((qz > 0.0f || py < 0.0f) ? 1.0f : -1.0f);
   }

   default:
      return 1e30f; // unknown shape: never hit
   }
}

/// Signed distance of a world-space point to the primitive
__device__ __forceinline__ float sdf_eval(const SdfPrim &s, float x, float y, float z)
{
   x -= s.cx;
   y -= s.cy;
   z -= s.cz;
   // Undo the rotation: inverse Euler, so Z then Y then X with negated angles
   if (s.rz != 0.0f)
   {
      const float c = cosf(s.rz), sn = sinf(s.rz);
      const float nx = c * x + sn * y, ny = -sn * x + c * y;
      x = nx;
      y = ny;
   }
   if (s.ry != 0.0f)
   {
      const float c = cosf(s.ry), sn = sinf(s.ry);
      const float nx = c * x - sn * z, nz = sn * x + c * z;
      x = nx;
      z = nz;
   }
   if (s.rx != 0.0f)
   {
      const float c = cosf(s.rx), sn = sinf(s.rx);
      const float ny = c * y + sn * z, nz = -sn * y + c * z;
      y = ny;
      z = nz;
   }
   return sdf_local(s.shape, s.p0, s.p1, s.p2, x, y, z);
}

/**
 * @brief Sphere-trace a ray against one primitive.
 * @param ox,oy,oz  Ray origin (world)
 * @param dx,dy,dz  Ray direction, unit length
 * @param tmin,tmax Valid parameter range
 * @param t_hit     Parameter of the hit
 * @param nx,ny,nz  Outward unit normal at the hit (gradient of the distance)
 * @return true on a hit
 */
__device__ __forceinline__ bool sdf_march(const SdfPrim &s, float ox, float oy, float oz, float dx, float dy, float dz,
                                          float tmin, float tmax, float &t_hit, float &nx, float &ny, float &nz)
{
   // Clip to the bounding sphere: no need to march through empty space around the shape
   const float R   = sdf_bound_radius(s.shape, s.p0, s.p1, s.p2);
   const float ocx = ox - s.cx, ocy = oy - s.cy, ocz = oz - s.cz;
   const float b    = ocx * dx + ocy * dy + ocz * dz;
   const float c    = ocx * ocx + ocy * ocy + ocz * ocz - R * R;
   const float disc = b * b - c;
   if (disc < 0.0f)
      return false;
   const float sq   = sqrtf(disc);
   const float t_in = fmaxf(tmin, -b - sq);
   const float t_out = fminf(tmax, -b + sq);
   if (t_in > t_out)
      return false;

   float t = t_in;
   // A ray leaving the surface (bounce or shadow ray) starts inside the tolerance band: hits
   // only count once the ray has moved clear of the band, otherwise it would re-hit its origin
   bool armed = false;
   for (int i = 0; i < SDF_MAX_STEPS; ++i)
   {
      float x = ox + t * dx, y = oy + t * dy, z = oz + t * dz;
      float d = sdf_eval(s, x, y, z);
      // Tolerance grows slowly with the distance travelled (float precision)
      const float eps = SDF_HIT_EPSILON * fmaxf(1.0f, 0.1f * t);
      if (!armed)
      {
         // Rays leaving a surface (bounces, shadow rays, refraction into the shape) start in
         // or next to the band: step until the distance has grown clear of it, on either side
         if (fabsf(d) > 1.5f * eps)
            armed = true;
         else
         {
            t += fmaxf(fabsf(d), 2.0f * eps);
            if (t > t_out)
               return false;
            continue;
         }
      }
      if (fabsf(d) < eps)
      {
         // Refine: one more signed step lands on the surface (the distance is exact along the
         // normal and a lower bound along the ray), so the hit sits on it rather than in front
         t += d;
         x = ox + t * dx; y = oy + t * dy; z = oz + t * dz;
         t_hit = t;
         const float e = SDF_NORMAL_EPS;
         nx = sdf_eval(s, x + e, y, z) - sdf_eval(s, x - e, y, z);
         ny = sdf_eval(s, x, y + e, z) - sdf_eval(s, x, y - e, z);
         nz = sdf_eval(s, x, y, z + e) - sdf_eval(s, x, y, z - e);
         const float len = sdf_len3(nx, ny, nz);
         if (len > 0.0f)
         {
            nx /= len;
            ny /= len;
            nz /= len;
         }
         else
         {
            nx = -dx;
            ny = -dy;
            nz = -dz;
         }
         return true;
      }
      // Inside the shape (ray started inside): march on |d| so the far side is found
      t += fmaxf(fabsf(d), 0.5f * eps);
      if (t > t_out)
         return false;
   }
   return false;
}
