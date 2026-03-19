#pragma once
#include "cuda_float3.cuh"
#include "sobol_sampler.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

//==============================================================================
// ERROR CHECKING
//==============================================================================
#define CUDA_CHECK(call)                                                                                               \
   do                                                                                                                  \
   {                                                                                                                   \
      cudaError_t err = (call);                                                                                        \
      if (err != cudaSuccess)                                                                                          \
      {                                                                                                                \
         printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);                            \
      }                                                                                                                \
   } while (0)

//==============================================================================
// RANDOMNESS
//==============================================================================

/**
 * @brief Sobol sampler state overlaid on curandState (which is 48 bytes on GPU).
 * Only 16 bytes are used, leaving plenty of room in the 48-byte curandState.
 */
struct SobolSamplerState
{
   uint32_t pixel_hash;  ///< Per-pixel stable hash (from pixel coords + scene seed)
   uint32_t sample_idx;  ///< Gray-code encoded sample index — kept for backward-compat rand_float() path
   uint32_t dim_idx;     ///< Auto-increments on every rand_float() call; resets to 0 each sample
   uint32_t pcg_seed;    ///< PCG fallback seed used when dim_idx >= SOBOL_MAX_DIM
   uint32_t sample_n;    ///< Raw sample index (not Gray-coded) — used by rand_float2() / sobol_2d_sample()
};
static_assert(sizeof(SobolSamplerState) <= sizeof(curandState),
              "SobolSamplerState must fit inside curandState (48 bytes)");

// ---------------------------------------------------------------------------
// Sobol effect IDs for rand_float2().
// Each distinct sampling use within a path (AA jitter, DOF, per-material
// scatter, Russian roulette) gets its own ID so sobol_2d_sample() produces
// an independent, well-stratified 2D sequence for every use.
// ---------------------------------------------------------------------------
constexpr uint32_t SOBOL_EFFECT_AA               = 0u;  ///< Camera ray anti-aliasing jitter
constexpr uint32_t SOBOL_EFFECT_DOF              = 1u;  ///< Depth-of-field aperture disk
constexpr uint32_t SOBOL_EFFECT_LAMBERTIAN       = 2u;  ///< Lambertian cosine hemisphere
constexpr uint32_t SOBOL_EFFECT_GLASS            = 3u;  ///< Glass/dielectric Fresnel decision
constexpr uint32_t SOBOL_EFFECT_ANISO_GGX        = 4u;  ///< Anisotropic GGX microfacet VNDF
constexpr uint32_t SOBOL_EFFECT_THIN_FILM        = 5u;  ///< Thin-film Fresnel decision
constexpr uint32_t SOBOL_EFFECT_CLEAR_COAT       = 6u;  ///< Clear-coat Fresnel lobe selection
constexpr uint32_t SOBOL_EFFECT_CLEAR_COAT_DIR   = 7u;  ///< Clear-coat specular perturbation
constexpr uint32_t SOBOL_EFFECT_CLEAR_COAT_DIFF  = 8u;  ///< Clear-coat base diffuse hemisphere
constexpr uint32_t SOBOL_EFFECT_ROUGH_MIRROR     = 9u;  ///< Rough mirror normal perturbation
constexpr uint32_t SOBOL_EFFECT_RR               = 10u; ///< Russian roulette path termination

/// Runtime toggle — defined in cuda_utils.cu, set via setSobolSampler() extern-C call.
extern __device__ bool g_use_sobol;

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
 * @brief Generate random float in [0,1) — Sobol' or PCG depending on g_use_sobol.
 *
 * Sobol path: evaluates dimension dim_idx of the scrambled Sobol sequence at
 * sample_idx stored in the SobolSamplerState overlay.  dim_idx is incremented
 * automatically.  Falls back to PCG when dim_idx >= SOBOL_MAX_DIM.
 *
 * PCG path: classic PCG32 one-liner, identical to the previous implementation.
 */
static __device__ inline float rand_float(curandState *state)
{
   if (g_use_sobol)
   {
      SobolSamplerState *ss = reinterpret_cast<SobolSamplerState *>(state);
      if (ss->dim_idx < (uint32_t)SOBOL_MAX_DIM)
      {
         float v = sobol_float(ss->sample_idx, (int)ss->dim_idx, ss->pixel_hash);
         ss->dim_idx++;
         return v;
      }
      // PCG fallback for dims beyond SOBOL_MAX_DIM
      ss->pcg_seed = ss->pcg_seed * 747796405u + 2891336453u;
      uint32_t word = ((ss->pcg_seed >> ((ss->pcg_seed >> 28u) + 4u)) ^ ss->pcg_seed) * 277803737u;
      ss->dim_idx++;
      return ((word >> 22u) ^ word) * (1.0f / 4294967296.0f);
   }
   // Original PCG path (fast, stateful, pseudorandom)
   unsigned int *fast_state = (unsigned int *)state;
   *fast_state = *fast_state * 747796405u + 2891336453u;
   unsigned int word = ((*fast_state >> ((*fast_state >> 28u) + 4u)) ^ *fast_state) * 277803737u;
   unsigned int result = (word >> 22u) ^ word;
   return result * (1.0f / 4294967296.0f);
}

/**
 * @brief Reset the Sobol state for a new sample.
 * Must be called at the top of each per-sample loop iteration.
 * In PCG mode this is a no-op.
 * @param state  Per-pixel random state (interpreted as SobolSamplerState)
 * @param n      Absolute sample index (0, 1, 2, ...) — stored raw for rand_float2()
 *               and also Gray-coded for the legacy rand_float() path.
 */
static __device__ inline void reset_sobol_state_for_sample(curandState *state, uint32_t n)
{
   if (g_use_sobol)
   {
      SobolSamplerState *ss = reinterpret_cast<SobolSamplerState *>(state);
      ss->sample_n   = n;            // raw index for sobol_2d_sample() via rand_float2()
      ss->sample_idx = sobol_gray(n); // Gray-coded index for legacy rand_float() path
      ss->dim_idx    = 0u;
   }
}

/**
 * @brief Get a well-stratified 2D sample for a specific (bounce, effect) use.
 *
 * In Sobol mode: calls sobol_2d_sample() with full per-(pixel,bounce,effect)
 * seeding and index shuffling — the reference ShaderToy approach.
 * In PCG mode: two independent rand_float() calls.
 *
 * @param state      Per-pixel random state
 * @param sample_n   Raw sample index (total_samples_so_far + s)
 * @param pixel_hash Per-pixel hash (from SobolSamplerState::pixel_hash)
 * @param bounce     Path depth (0 = first scatter)
 * @param effect     Use-case ID (SOBOL_EFFECT_* constant)
 */
static __device__ inline float2 rand_float2(curandState *state, uint32_t sample_n, uint32_t pixel_hash,
                                             uint32_t bounce, uint32_t effect)
{
   if (g_use_sobol)
      return sobol_2d_sample(sample_n, bounce, effect, pixel_hash);
   // PCG fallback: two stateful calls
   return {rand_float(state), rand_float(state)};
}

/**
 * @brief Area-preserving square-to-unit-sphere mapping.
 * Maps (u, v) ∈ [0,1)² to a uniformly distributed point on the unit sphere.
 * No rejection needed — safe to use with low-discrepancy sequences.
 */
__device__ __forceinline__ f3 sphere_from_square(float u, float v)
{
   float z   = 1.0f - 2.0f * u;
   float phi = 6.28318530717958647692f * v;  // 2π
   float r   = sqrtf(fmaxf(0.0f, 1.0f - z * z));
   return f3(r * cosf(phi), r * sinf(phi), z);
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
