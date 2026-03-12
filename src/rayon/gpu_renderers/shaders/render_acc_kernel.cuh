// Accumulative rendering CUDA kernel declaration
#pragma once

#include "cuda_raytracer.cuh"
#include "cuda_scene.cuh"

//==============================================================================
// Gamma correction: converts float4 accumulation buffer to uint8 display image.
// When pixel_sample_counts is non-null (adaptive sampling), each pixel uses its
// own sample count for normalization instead of the global num_samples.
//==============================================================================
__global__ void gammaCorrectKernel(const float4 *__restrict__ accum_buffer, unsigned char *display_image, int width,
                                   int height, int num_samples, int channels, float gamma,
                                   const int *pixel_sample_counts = nullptr);

//==============================================================================
// Sample count heatmap: visualizes per-pixel sample counts using the Plasma
// colormap. Low sample counts (converged early) appear dark purple/blue,
// high counts (still sampling) appear bright yellow.
//==============================================================================
__global__ void sampleHeatmapKernel(const int *pixel_sample_counts, unsigned char *display_image, int width, int height,
                                    int channels, int max_samples_for_scale);

//==============================================================================
// Path tracing kernel: traces samples_to_add rays per pixel and accumulates.
// When pixel_sample_counts is non-null (adaptive sampling), pixels that have
// already converged are skipped entirely.
//==============================================================================
__global__ void renderAccKernel(float4 *accum_buffer, const CudaScene::Scene *__restrict__ scene, int width, int height,
                                int samples_to_add, int total_samples_so_far, int max_depth, float cam_center_x,
                                float cam_center_y, float cam_center_z, float pixel00_x, float pixel00_y,
                                float pixel00_z, float delta_u_x, float delta_u_y, float delta_u_z, float delta_v_x,
                                float delta_v_y, float delta_v_z, unsigned long long *ray_count,
                                curandState *rand_states, float cam_u_x, float cam_u_y, float cam_u_z, float cam_v_x,
                                float cam_v_y, float cam_v_z,
                                int *pixel_sample_counts = nullptr, int min_adaptive_samples = 32,
                                float adaptive_threshold = 0.01f);
