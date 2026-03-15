# Why RayTracingInVulkan is Faster than RayON's CUDA Renderer

## Overview

Both projects implement Peter Shirley's "Ray Tracing in One Weekend" series, but RayTracingInVulkan leverages Vulkan's `VK_KHR_ray_tracing_pipeline` extension to access dedicated hardware RT cores, while RayON performs all ray tracing in software on general-purpose CUDA cores.

## 1. Hardware-Accelerated BVH Traversal (Biggest Factor)

RayTracingInVulkan offloads BVH traversal and ray-triangle intersection to **dedicated RT cores** — fixed-function units purpose-built for this task. They run in parallel with shader cores and are 10-100x faster at BVH traversal than general-purpose compute.

RayON implements BVH traversal in a CUDA kernel (`cuda_raytracer.cuh`), using a manual stack-based loop. Every node test and intersection check consumes shader core cycles that could be doing shading work instead.

## 2. Less Warp Divergence

In the Vulkan RT pipeline, the GPU driver and RT hardware handle ray coherence scheduling automatically. Rays hitting similar geometry get batched together.

In RayON's CUDA kernels, neighboring pixels in a warp diverge heavily:
- Different BVH paths (interior vs. leaf nodes)
- Different material types (Lambertian vs. Glass vs. Metal via `switch` branches)
- Different bounce depths (Russian roulette termination)

This serializes execution within warps — the slowest thread dictates warp completion time.

## 3. Iterative Bounce Loop Optimization

RayTracingInVulkan sets `maxRayRecursionDepth = 1` and handles bounces iteratively in the ray generation shader. The driver optimizes this pattern because it knows the full ray lifecycle.

RayON also uses an iterative loop, but without driver-level scheduling optimizations. The CUDA kernel must manage all state (ray, attenuation, hit record) in registers, hitting the `-maxrregcount 80` cap and potentially spilling to slow local memory.

## 4. Memory Access Patterns

| Aspect | RayON (CUDA) | RayTracingInVulkan |
|--------|-------------|-------------------|
| Accumulation buffer | 3 separate float writes (scattered, non-coalesced) | `R32G32B32A32_SFLOAT` image (HW-optimized format) |
| BVH nodes | Random global memory reads via flat array | Driver-optimized acceleration structure layout |
| Scene data | Manual flat arrays with index indirection | Managed GPU buffers with optimal alignment |

## 5. Fast Math

RayON has `--use_fast_math` commented out in CMakeLists.txt, preserving IEEE precision at the cost of speed. Vulkan's GLSL shaders allow the driver to apply aggressive fast approximations for `sqrt`, `acos`, etc.

## Summary Comparison

| Aspect | RayON (CUDA) | RayTracingInVulkan |
|--------|-------------|-------------------|
| BVH traversal | Software (shader cores) | Hardware (RT cores) |
| Ray scheduling | Manual 1:1 pixel→thread | Driver-managed coherence |
| Warp divergence | High (materials, BVH, depth) | Minimized by HW scheduler |
| Memory layout | Manual flat arrays | Driver-optimized AS |
| Fast math | Disabled | Driver-optimized |

The RT cores alone account for the bulk of the difference. On an RTX GPU, they handle BVH traversal and triangle intersection essentially "for free" while shader cores focus purely on shading — RayON's shader cores must do both.

## Reference Performance

RayTracingInVulkan reports ~140 fps at 1280x720 with 8 rays/pixel on an RTX 2080 Ti using hardware RT.
