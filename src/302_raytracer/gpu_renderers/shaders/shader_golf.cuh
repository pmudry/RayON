#pragma once

#include "../cuda_float3.cuh"
#include "../cuda_utils.cuh"

// Forward declarations to avoid including shader_common.cuh here (prevents recursion/duplicates)
struct ray_simple;
struct hit_record_simple;

#include <cfloat>
#include <cmath>
#include <curand_kernel.h>

__device__ f3 fibonacci_point(int i, int n);


__device__ float distanceToNearestDimple(f3 p);


__device__ float hexagonalDimplePattern(f3 p);


__device__ float golfBallDisplacement(f3 p, f3 center, float radius);


__device__ bool hit_golf_ball_sphere(f3 center, float radius, const ray_simple &r, float t_min,
                                     float t_max, hit_record_simple &rec);
