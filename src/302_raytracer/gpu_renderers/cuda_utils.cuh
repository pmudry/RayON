#pragma once
#include "cuda_float3.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

//==============================================================================
// RANDOMNESS
//==============================================================================

/**
 * @brief Fast RNG state using PCG (Permuted Congruential Generator)
 * Much faster than curand_uniform (~5x speedup)
 */
struct FastRNG {
    unsigned int state;
    
    __device__ FastRNG(unsigned int seed) : state(seed) {}
    
    // PCG hash function - fast and high quality
    __device__ inline unsigned int pcg_hash() {
        state = state * 747796405u + 2891336453u;
        unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }
    
    // Generate float in [0, 1)
    __device__ inline float next() {
        return pcg_hash() * (1.0f / 4294967296.0f);
    }
};

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
static __device__ inline float rand_float(curandState *state) { 
    // Use curandState as storage for our FastRNG state
    // We reinterpret the curandState pointer as containing our simple uint state
    unsigned int *fast_state = (unsigned int*)state;
    
    // PCG hash inline for maximum speed
    *fast_state = *fast_state * 747796405u + 2891336453u;
    unsigned int word = ((*fast_state >> ((*fast_state >> 28u) + 4u)) ^ *fast_state) * 277803737u;
    unsigned int result = (word >> 22u) ^ word;
    
    return result * (1.0f / 4294967296.0f);
}

/**
 * @brief Generate random position on sphere surface using spherical coordinates
 * @param state Random state for random number generation
 * @param center Sphere center
 * @param radius Sphere radius
 * @return Random point on sphere surface
 */
static __device__ inline float3_simple randPosInSphere(curandState *state, float3_simple center, float radius)
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
static __device__ inline void cartesianToSpherical(float3_simple p, float &theta, float &phi)
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
