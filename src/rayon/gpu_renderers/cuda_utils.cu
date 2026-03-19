// Define the Sobol direction-vector table in this translation unit only.
// All other CUDA TUs get an extern __constant__ declaration via cuda_utils.cuh.
#define SOBOL_DEFINE_DIRECTIONS
#include "cuda_utils.cuh"

//==============================================================================
// Sobol/PCG runtime toggle
// Defined here (one TU) so it can be accessed from all CUDA device files via
// `extern __device__ bool g_use_sobol` declared in cuda_utils.cuh.
// Default: Sobol enabled (better convergence, lower variance).
//==============================================================================
__device__ bool g_use_sobol = true;

/// Host-callable function to switch sampler at runtime.
extern "C" void setSobolSampler(bool use_sobol)
{
   cudaMemcpyToSymbol(g_use_sobol, &use_sobol, sizeof(bool));
}

//------------------------------------------------------------------------------
// pixel_hash_fn: maps a linear pixel index to a per-pixel scramble seed.
// Uses the MurmurHash3 fmix32 finalizer (defined in sobol_sampler.cuh)
// which guarantees full avalanche from any single-bit input change.
// The scene/session seed is mixed in so re-renders get fresh scrambles.
//------------------------------------------------------------------------------
static __device__ __forceinline__ uint32_t pixel_hash_fn(uint32_t linear_idx, unsigned long long seed)
{
   return fmix32(linear_idx ^ (uint32_t)(seed * 2654435761ull));
}

// Implement kernel in a single translation unit to avoid nvlink multiple definition errors
__global__ void init_random_states(curandState *rand_states, int num_states, unsigned long long seed, int width)
{
   // Support both 1D and 2D grid launches
   int idx;
   int x, y;
   if (gridDim.y == 1)
   {
      // 1D launch — derive 2D coords from idx for pixel hash
      idx = blockIdx.x * blockDim.x + threadIdx.x;
      x = idx % (width > 0 ? width : 1);
      y = idx / (width > 0 ? width : 1);
   }
   else
   {
      x = blockIdx.x * blockDim.x + threadIdx.x;
      y = blockIdx.y * blockDim.y + threadIdx.y;
      idx = y * width + x;
   }

   if (idx < num_states)
   {
      if (g_use_sobol)
      {
         // Initialise as SobolSamplerState (overlaid on curandState)
         SobolSamplerState *ss = reinterpret_cast<SobolSamplerState *>(&rand_states[idx]);
         ss->pixel_hash = pixel_hash_fn((uint32_t)idx, seed);
         ss->sample_n   = 0u;
         ss->sample_idx = 0u;
         ss->dim_idx    = 0u;
         // PCG fallback seed — unique per pixel, derived differently from pixel_hash
         ss->pcg_seed = ss->pixel_hash ^ 0xdeadbeef ^ (uint32_t)(seed >> 32);
      }
      else
      {
         // Original PCG initialisation
         unsigned int *fast_state = (unsigned int *)&rand_states[idx];
         *fast_state = (unsigned int)(seed + (unsigned long long)idx * 747796405u);
      }
   }
}
