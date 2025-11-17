// CUDA translation-unit wrapper ensuring cuda_raytracer.cuh is compiled once.
// All device-side helpers remain in the header so kernels can include them directly.
#include "cuda_raytracer.cuh"
