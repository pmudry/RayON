#include "cuda_utils.cuh"
#include "cuda_metrics.hpp"
#include <curand_kernel.h>
#include <stdio.h>
#include <map>
#include <mutex>

//==============================================================================
// VRAM TRACKING IMPLEMENTATION
//==============================================================================
static size_t g_total_vram_allocated = 0;
static std::map<void*, size_t> g_allocations;
static std::mutex g_alloc_mutex;

cudaError_t cudaMallocTrackedInternal(void** devPtr, size_t size) {
    cudaError_t err = cudaMalloc(devPtr, size);
    if (err == cudaSuccess && devPtr != nullptr && *devPtr != nullptr) {
        std::lock_guard<std::mutex> lock(g_alloc_mutex);
        g_allocations[*devPtr] = size;
        g_total_vram_allocated += size;
    }
    return err;
}

cudaError_t cudaFreeTracked(void* devPtr) {
    if (devPtr == nullptr) return cudaSuccess;
    
    {
        std::lock_guard<std::mutex> lock(g_alloc_mutex);
        auto it = g_allocations.find(devPtr);
        if (it != g_allocations.end()) {
            g_total_vram_allocated -= it->second;
            g_allocations.erase(it);
        }
    }
    return cudaFree(devPtr);
}

size_t getTrackedVramUsage() {
    std::lock_guard<std::mutex> lock(g_alloc_mutex);
    return g_total_vram_allocated;
}

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

void getCudaDeviceMetrics(std::string& name, size_t& vram_used) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        name = prop.name;
    } else {
        name = "Unknown CUDA Device";
    }

    // Use our precise tracked value instead of system-wide estimate
    vram_used = getTrackedVramUsage();
}