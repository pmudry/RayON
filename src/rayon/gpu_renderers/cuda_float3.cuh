/**
 * @file cuda_float3.cuh
 * @brief Simple 3D vector math operations optimized for CUDA
 *
 * Provides a lightweight f3 struct with common vector operations
 * needed for ray tracing computations on the GPU.
 */
#pragma once

#include <cmath>
#include <cuda_runtime.h>

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
__device__ __forceinline__ f2 operator*(float t, const f2 &v) { return f2(t * v.x, t * v.y); }

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

   __host__ __device__ f3 operator*(f3 other) const { return f3(x * other.x, y * other.y, z * other.z); }

   __host__ __device__ f3 operator/(float t) const { return f3(x / t, y / t, z / t); }

   __host__ __device__ f3 operator-() const { return f3(-x, -y, -z); }

   __host__ __device__ float length() const { return sqrtf(x * x + y * y + z * z); }

   __host__ __device__ float length_squared() const { return x * x + y * y + z * z; }
};

const f3 f3_ZEROES(0.0f, 0.0f, 0.0f);
const f3 f3_ONES(1.0f, 1.0f, 1.0f);

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
