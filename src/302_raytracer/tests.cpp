/**
 * @file cuda_float3.cuh
 * @brief Simple 3D vector math operations optimized for CUDA
 *
 * Provides a lightweight f3 struct with common vector operations
 * needed for ray tracing computations on the GPU.
 */
#pragma once

#include <cassert>
#include <cmath>
#include <stdio.h>

#define __host__
#define __device__
#define __forceinline__

//==============================================================================
// VECTOR MATH AND UTILITY STRUCTURES
//==============================================================================

/**
 * @brief Simple 2D vector structure for DOF calculations
 */
struct f2
{
   float x, y;

   __host__ __device__ f2() : x(0), y(0) {}
   __host__ __device__ f2(float x_, float y_) : x(x_), y(y_) {}

   __host__ __device__ f2 operator+(const f2 &other) const { return f2(x + other.x, y + other.y); }

   __host__ __device__ f2 operator-(const f2 &other) const { return f2(x - other.x, y - other.y); }

   __host__ __device__ f2 operator*(float t) const { return f2(x * t, y * t); }

   __host__ __device__ f2 operator/(float t) const { return f2(x / t, y / t); }
};

/** @brief Scalar multiplication from left */
__device__ f2 operator*(float t, const f2 &v) { return f2(t * v.x, t * v.y); }

/**
 * @brief Simple 3D vector structure optimized for CUDA
 * Provides basic vector operations for ray tracing computations
 */
struct f3
{
   float x, y, z;

   __host__ __device__ f3() : x(0), y(0), z(0) {}
   __host__ __device__ f3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

   __host__ __device__ f3 operator+(const f3 &other) const { return f3(x + other.x, y + other.y, z + other.z); }

   __host__ __device__ f3 operator-(const f3 &other) const { return f3(x - other.x, y - other.y, z - other.z); }

   __host__ __device__ f3 operator*(float t) const { return f3(x * t, y * t, z * t); }

   __host__ __device__ f3 operator/(float t) const { return f3(x / t, y / t, z / t); }

   __host__ __device__ f3 operator-() const { return f3(-x, -y, -z); }

   __host__ __device__ float length() const { return sqrtf(x * x + y * y + z * z); }

   __host__ __device__ float length_squared() const { return x * x + y * y + z * z; }
};

/** @brief Scalar multiplication from left */
__device__ __forceinline__ f3 operator*(float t, const f3 &v) { return v * t; }

/** @brief Compute dot product of two vectors */
__device__ __forceinline__ float dot(const f3 &a, const f3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

/** @brief Compute cross product of two vectors */
__device__ __forceinline__ f3 cross(const f3 &a, const f3 &b)
{
   return f3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

/** @brief Normalize a vector to unit length */
__device__ __forceinline__ f3 normalize(const f3 &v) { return v / v.length(); }

/** @brief Convert a normal to a debug RGB color */
__device__ __forceinline__ f3 normal_to_color(const f3 &n)
{
   return f3(0.5f * (n.x + 1.0f), 0.5f * (n.y + 1.0f), 0.5f * (n.z + 1.0f));
}

/** @brief Create grayscale color from single value */
__device__ __forceinline__ f3 grayscale(float v)
{
   v = fmaxf(0.0f, fminf(1.0f, v));
   return f3(v, v, v);
}

void build_orthonormal_basis(const f3 &n, f3 &u, f3 &v)
{
   // from "Building an Orthonormal Basis, Pixar" / Shirley
   if (fabs(n.x) > fabs(n.z))
   {
      u = normalize(f3(-n.y, n.x, 0.0f));
   }
   else
   {
      u = normalize(f3(0.0f, -n.z, n.y));
   }
   v = cross(n, u);
}

void printfVector(const f3 &vec, const char *name) { printf("%s: [%f, %f, %f]\n", name, vec.x, vec.y, vec.z); }

using namespace std;

int main()
{

   f3 u, v, w;

   for (int i = 0; i < 10000; i++)
   {
      w = f3(rand() / (float)RAND_MAX * 2.0f - 1.0f, rand() / (float)RAND_MAX * 2.0f - 1.0f,
             rand() / (float)RAND_MAX * 2.0f - 1.0f);
      w = normalize(w);

      printfVector(w, "w");
      assert(fabs(w.length() - 1.0f) < 1e-6);

      build_orthonormal_basis(w, u, v);
    //   cout << dot(u, v) << endl;
    //   cout << dot(u, w) << endl;
    //   cout << dot(v, w) << endl;

      assert(dot(u, v) < 1e-6);
      assert(dot(u, w) < 1e-6);
      assert(dot(v, w) < 1e-6);
   }

   printfVector(u, "u");
   printfVector(v, "v");
   printfVector(w, "w");
   return 0;
}