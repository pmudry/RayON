#pragma once

#include "../cuda_float3.cuh"
#include "../cuda_utils.cuh"

// Forward declarations to avoid including shader_common.cuh here (prevents recursion/duplicates)
struct ray_simple;
struct hit_record_simple;

#include <cfloat>
#include <cmath>
#include <curand_kernel.h>

__device__ float3_simple fibonacci_point(int i, int n);


__device__ float distanceToNearestDimple(float3_simple p);


__device__ float hexagonalDimplePattern(float3_simple p);


__device__ float golfBallDisplacement(float3_simple p, float3_simple center, float radius);


__device__ bool hit_golf_ball_sphere(float3_simple center, float radius, const ray_simple &r, float t_min,
                                     float t_max, hit_record_simple &rec);
