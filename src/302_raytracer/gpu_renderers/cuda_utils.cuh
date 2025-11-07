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
__global__ void init_random_states(curandState *rand_states, int num_states, unsigned long long seed, int width)
{
   // Support both 1D and 2D grid launches
   int idx;
   if (gridDim.y == 1)
   {
      // 1D launch
      idx = blockIdx.x * blockDim.x + threadIdx.x;
   }
   else
   {
      // 2D launch - compute proper 1D index
      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;
      idx = y * width + x;
   }

   if (idx < num_states)
   {
      // Initialize each state with a unique seed and sequence
      // Using idx for sequence ensures different random streams per pixel
      curand_init(seed, idx, 0, &rand_states[idx]);
   }
}

/** @brief Generate random float in range [0,1) using CUDA's curand */
__device__ float rand_float(curandState *state) { return curand_uniform(state); }

/**
 * @brief Generate random position on sphere surface using spherical coordinates
 * @param state Random state for random number generation
 * @param center Sphere center
 * @param radius Sphere radius
 * @return Random point on sphere surface
 */
__device__ float3_simple randPosInSphere(curandState *state, float3_simple center, float radius)
{
   float theta = 2.0f * M_PI * rand_float(state);      // Azimuth [0, 2π]
   float phi = acosf(1.0f - 2.0f * rand_float(state)); // Polar [0, π]

   // Convert spherical to cartesian coordinates
   float x = sinf(phi) * cosf(theta);
   float y = sinf(phi) * sinf(theta);
   float z = cosf(phi);

   return center + float3_simple(x * radius, y * radius, z * radius);
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
__device__ void cartesianToSpherical(float3_simple p, float &theta, float &phi)
{
   float r = p.length();
   if (r < 1e-6f)
   {
      theta = 0.0f;
      phi = 0.0f;
      return;
   }

   // Apply 90-degree rotation around x-axis: (x,y,z) -> (x,-z,y)
   float3_simple rotated = float3_simple(p.x, -p.z, p.y);

   theta = atan2f(rotated.y, rotated.x);
   if (theta < 0.0f)
      theta += 2.0f * M_PI;
   phi = acosf(rotated.z / r);
}
