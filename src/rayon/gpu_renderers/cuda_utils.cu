#include "cuda_utils.cuh"
#include <curand_kernel.h>
#include <stdio.h>

//==============================================================================
// KERNELS
//==============================================================================

// Implement kernel in a single translation unit to avoid nvlink multiple definition errors
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
      // Initialize fast RNG state - we repurpose curandState storage
      // Simple but effective: combine seed with index for per-pixel unique sequences
      unsigned int *fast_state = (unsigned int *)&rand_states[idx];
      *fast_state = (unsigned int)(seed + idx * 747796405u);
   }
}