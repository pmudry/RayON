// Accumulative rendering CUDA kernel declaration
#pragma once

#include "cuda_raytracer.cuh"
#include "cuda_scene.cuh"

__global__ void renderAccKernel(float4 *accum_buffer, const CudaScene::Scene *__restrict__ scene, int width, int height,
                                int samples_to_add, int total_samples_so_far, int max_depth, float cam_center_x,
                                float cam_center_y, float cam_center_z, float pixel00_x, float pixel00_y,
                                float pixel00_z, float delta_u_x, float delta_u_y, float delta_u_z, float delta_v_x,
                                float delta_v_y, float delta_v_z, unsigned long long *ray_count,
                                curandState *rand_states, float cam_u_x, float cam_u_y, float cam_u_z, float cam_v_x,
                                float cam_v_y, float cam_v_z);
