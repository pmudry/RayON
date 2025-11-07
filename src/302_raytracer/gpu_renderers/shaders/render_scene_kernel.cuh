// Scene-based rendering CUDA kernel declaration
#pragma once

#include "shader_common.cuh"

__global__ void renderKernelWithScene(unsigned char *image, CudaScene::Scene scene, int width, int height,
                                      int samples_per_pixel, int max_depth, float cam_center_x, float cam_center_y,
                                      float cam_center_z, float pixel00_x, float pixel00_y, float pixel00_z,
                                      float delta_u_x, float delta_u_y, float delta_u_z, float delta_v_x,
                                      float delta_v_y, float delta_v_z, unsigned long long *ray_count,
                                      curandState *rand_states);
