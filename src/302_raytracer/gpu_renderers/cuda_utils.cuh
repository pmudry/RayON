#pragma once
#include "cuda_float3.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

//==============================================================================
// RANDOMNESS
//==============================================================================
/**
 * @brief Initialize random states for all threads
 * This kernel should be called once at startup to initialize the shared random state array
 * @param rand_states Array of random states (one per thread/pixel)
 * @param num_states Total number of states to initialize
 * @param seed Base seed for random number generation
 */
// Forward declaration only. Implemented in cuda_utils.cu to avoid multiple definition at device link.
__global__ void init_random_states(curandState *rand_states, int num_states, unsigned long long seed, int width);

/**
 * @brief Generate random float in range [0,1) using fast PCG generator
 * Replaces slow curand_uniform with much faster custom RNG
 */
static __device__ inline float rand_float(curandState *state)
{
   // Use curandState as storage for our FastRNG state
   // We reinterpret the curandState pointer as containing our simple uint state
   unsigned int *fast_state = (unsigned int *)state;

   // PCG hash inline for maximum speed
   *fast_state = *fast_state * 747796405u + 2891336453u;
   unsigned int word = ((*fast_state >> ((*fast_state >> 28u) + 4u)) ^ *fast_state) * 277803737u;
   unsigned int result = (word >> 22u) ^ word;

   return result * (1.0f / 4294967296.0f);
}

/**
 * @brief Generate a random vector with components in [-1, 1]
 * @param state Random state for random number generation
 * @return Random vector as f3 (not normalized)
 */
static __device__ inline f3 randUnitVector(curandState *state)
{
   float x = 2.0f * rand_float(state) - 1.0f;
   float y = 2.0f * rand_float(state) - 1.0f;
   float z = 2.0f * rand_float(state) - 1.0f;

   float length = sqrtf(x * x + y * y + z * z);

   // Avoid division by zero (extremely rare)
   if (length > 1e-8f)
   {
      x /= length;
      y /= length;
      z /= length;
   }
   return f3(x, y, z);
}

/**
 * @brief Generate a random unit vector uniformly distributed on the unit sphere
 * @param state Random state for random number generation
 * @return Random unit vector as f3
 */
static __device__ inline f3 randOnUnitSphere(curandState *state)
{
   float theta = 2.0f * M_PI * rand_float(state);      // Azimuth [0, 2π]
   float phi = acosf(1.0f - 2.0f * rand_float(state)); // Polar [0, π]

   // Convert spherical to cartesian coordinates
   float x = sinf(phi) * cosf(theta);
   float y = sinf(phi) * sinf(theta);
   float z = cosf(phi);

   return f3(x, y, z);
}

/**
 * @brief Generate random position on sphere surface using spherical coordinates
 * @param state Random state for random number generation
 * @param center Sphere center
 * @param radius Sphere radius
 * @return Random point on sphere surface
 */
static __device__ inline f3 randPosInSphere(curandState *state, f3 center, float radius)
{
   float theta = 2.0f * M_PI * rand_float(state);      // Azimuth [0, 2π]
   float phi = acosf(1.0f - 2.0f * rand_float(state)); // Polar [0, π]

   // Convert spherical to cartesian coordinates
   float x = sinf(phi) * cosf(theta);
   float y = sinf(phi) * sinf(theta);
   float z = cosf(phi);

   return center + f3(x * radius, y * radius, z * radius);
}

static __device__ inline void build_orthonormal_basis(const f3 &n, f3 &u, f3 &v)
{
   // from "Building an Orthonormal Basis, Pixar" / Shirley
   if (fabs(n.x) > fabs(n.z))   
      u = normalize(f3(-n.y, n.x, 0.0f));   
   else   
      u = normalize(f3(0.0f, -n.z, n.y));   
   v = cross(n, u);
}

//==============================================================================
// Geometry transformations
//==============================================================================
/**
 * @brief Convert 3D point to spherical coordinates with 90-degree rotation around x-axis
 * @param p 3D point on sphere surface
 * @param theta Output azimuthal angle (0 to 2π)
 * @param phi Output polar angle (0 to π)
 */
static __device__ inline void cartesianToSpherical(f3 p, float &theta, float &phi)
{
   float r = p.length();
   if (r < 1e-6f)
   {
      theta = 0.0f;
      phi = 0.0f;
      return;
   }

   // Apply 90-degree rotation around x-axis: (x,y,z) -> (x,-z,y)
   f3 rotated = f3(p.x, -p.z, p.y);

   theta = atan2f(rotated.y, rotated.x);
   if (theta < 0.0f)
      theta += 2.0f * M_PI;
   phi = acosf(rotated.z / r);
}
