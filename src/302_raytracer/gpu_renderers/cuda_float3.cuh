/**
 * @file cuda_float3.cuh
 * @brief Simple 3D vector math operations optimized for CUDA
 *
 * Provides a lightweight float3_simple struct with common vector operations
 * needed for ray tracing computations on the GPU.
 */
#pragma once

#include <cuda_runtime.h>
#include <cmath>

//==============================================================================
// VECTOR MATH AND UTILITY STRUCTURES
//==============================================================================

/**
 * @brief Simple 3D vector structure optimized for CUDA
 * Provides basic vector operations for ray tracing computations
 */
struct float3_simple
{
   float x, y, z;

   __host__ __device__ float3_simple() : x(0), y(0), z(0) {}
   __host__ __device__ float3_simple(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

   __host__ __device__ float3_simple operator+(const float3_simple &other) const
   {
      return float3_simple(x + other.x, y + other.y, z + other.z);
   }

   __host__ __device__ float3_simple operator-(const float3_simple &other) const
   {
      return float3_simple(x - other.x, y - other.y, z - other.z);
   }

   __host__ __device__ float3_simple operator*(float t) const { return float3_simple(x * t, y * t, z * t); }

   __host__ __device__ float3_simple operator/(float t) const { return float3_simple(x / t, y / t, z / t); }

   __host__ __device__ float3_simple operator-() const { return float3_simple(-x, -y, -z); }

   __host__ __device__ float length() const { return sqrtf(x * x + y * y + z * z); }

   __host__ __device__ float length_squared() const { return x * x + y * y + z * z; }
};

/** @brief Scalar multiplication from left */
__device__ inline float3_simple operator*(float t, const float3_simple &v)
{
   return v * t;
}

/** @brief Compute dot product of two vectors */
__device__ inline float dot(const float3_simple &a, const float3_simple &b)
{
   return a.x * b.x + a.y * b.y + a.z * b.z;
}

/** @brief Compute cross product of two vectors */
__device__ inline float3_simple cross(const float3_simple &a, const float3_simple &b)
{
   return float3_simple(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

/** @brief Normalize a vector to unit length */
__device__ inline float3_simple unit_vector(const float3_simple &v)
{
   return v / v.length();
}

/** @brief Convert a normal to a debug RGB color */
__device__ inline float3_simple normal_to_color(const float3_simple &n)
{
   return float3_simple(0.5f * (n.x + 1.0f), 0.5f * (n.y + 1.0f), 0.5f * (n.z + 1.0f));
}

/** @brief Create grayscale color from single value */
__device__ inline float3_simple grayscale(float v)
{
   v = fmaxf(0.0f, fminf(1.0f, v));
   return float3_simple(v, v, v);
}
