# CUDA Renderer Optimization Plan

## Context
RayON's CUDA renderer is significantly slower than a comparable Vulkan RT raytracer (RayTracingInVulkan) primarily because it performs BVH traversal and intersection in software on shader cores, while Vulkan uses dedicated RT cores. This plan catalogs actionable optimizations and assesses OptiX migration.

See `explanations/VULKAN_VS_CUDA_PERFORMANCE.md` for the detailed comparison.

## Optimization Options

### Option 1: Enable `--use_fast_math` — DONE
Enables fast `rsqrtf`, fused multiply-add, relaxed denormals. Negligible visual impact.
- **File**: `CMakeLists.txt`

### Option 2: Fix accumulation buffer memory layout — DONE
Kernel already uses `float4` coalesced reads/writes.
- **Files**: `gpu_renderers/shaders/render_acc_kernel.cu`

### Option 3: Increase occupancy / tune kernel launch (Medium, ~10-20% speedup)
Profile with `ncu`, test 512 threads/block, evaluate register pressure vs. occupancy tradeoff.
- **Files**: `renderer_cuda_device.cu`

### Option 4: Use texture memory for BVH/geometry (Medium, ~5-15% speedup)
Bind BVH node and geometry arrays as CUDA texture objects for better cache behavior during traversal.
- **Files**: `renderer_cuda_device.cu`, `scene_builder_cuda.cu`, `cuda_raytracer.cuh`

### Option 5: Compact BVH node layout (Medium, ~10-20% speedup)
Pack BVH nodes to 64-byte cache-line alignment. Store child AABBs together so both children can be tested with a single cache line fetch.
- **Files**: `cuda_scene.cuh`, `cuda_raytracer.cuh`, `scene_builder_cuda.cu`, `scenes/scene_description.hpp`

### Option 6: Russian roulette from bounce 1 — DONE
Already starts at bounce 1 with energy compensation in `cuda_raytracer.cuh`.

### Option 7: Wavefront path tracing (Hard, ~30-50% speedup)
Split monolithic kernel into separate stages (ray gen → intersect → shade per material → bounce). Eliminates most warp divergence. Major architectural change.
- **Files**: New kernel files, major refactor of `render_acc_kernel.cu`, `cuda_raytracer.cuh`

### Option 8: Persistent threads with work queue (Hard, ~15-25% speedup)
Replace 1:1 pixel-thread mapping with fixed thread count pulling from global queue. Better load balancing for heterogeneous scenes.
- **Files**: `render_acc_kernel.cu`, `renderer_cuda_device.cu`

### Option 9: Migrate to OptiX (Hard, ~5-10x speedup)
Use NVIDIA OptiX SDK to access hardware RT cores for BVH traversal and intersection. This is the only path to match Vulkan RT performance. See detailed assessment below.

### Option A: Eliminate D2H round-trip in progressive renderer — DONE
Accumulation buffer stays on GPU. GPU-side `gammaCorrectKernel` produces uint8 display image directly. Only the small uint8 image (3 bytes/pixel) is copied to host instead of the full float4 buffer (16 bytes/pixel). Also uses `cudaMemset` instead of free/realloc on camera change.
- **Files**: `render_acc_kernel.cu`, `renderer_cuda_device.cu`, `renderer_cuda_progressive_host.hpp`

### Option B: Cache device properties — DONE
`getOptimalBlockSize()` caches result in static variable instead of calling `cudaGetDeviceProperties()` every frame.
- **File**: `renderer_cuda_device.cu`

### Option C: CUDA streams for async display copy (Medium, ~10-15% latency hiding)
Overlap kernel execution with display buffer transfer using CUDA streams. Currently the pipeline is fully synchronous.
- **Files**: `renderer_cuda_device.cu`

### Option E: BVH child ordering by ray direction sign (Medium, ~5-15% speedup)
Replace expensive distance-to-center heuristic with ray direction sign along split axis. One comparison instead of two `length_squared()` computations per interior node.
- **Files**: `cuda_raytracer.cuh`

### Option F: Flatten material dispatch in ray_color (Medium, ~5-15% speedup)
Remove CRTP lambda dispatch (`dispatch_material_bool`) and replace with explicit switch. Reduces register pressure and gives `nvcc` better optimization control.
- **Files**: `cuda_raytracer.cuh`

## OptiX Migration Assessment

### What OptiX Provides
- Hardware-accelerated BVH build and traversal (RT cores)
- Built-in ray-triangle and ray-AABB intersection
- Automatic ray coherence scheduling (reduces warp divergence)
- Denoiser module (AI-based, optional bonus)
- SBT (Shader Binding Table) for per-geometry material dispatch

### What Gets Replaced
| Current Component | Replaced By | File Impact |
|---|---|---|
| `hit_scene()` + BVH traversal in `cuda_raytracer.cuh` | OptiX built-in traversal | **Removed** |
| `hit_aabb()`, `hit_sphere()`, `hit_rectangle()` | OptiX intersection programs | **Rewritten** as .cu programs |
| `renderAccKernel` in `render_acc_kernel.cu` | OptiX ray generation program | **Rewritten** |
| Material dispatch in `cuda_raytracer.cuh` | OptiX closest-hit programs | **Rewritten** per material |
| BVH node struct in `cuda_scene.cuh` | OptiX GAS/IAS (managed by SDK) | **Removed** |
| `buildBVH()` in `scene_description.hpp` | `optixAccelBuild()` | **Replaced** for GPU path |
| `scene_builder_cuda.cu` | New OptiX AS builder | **Major rewrite** |

### What Gets Kept (unchanged or minor adaptation)
- `renderer_interface.hpp` — abstract interface, add new OptiX renderer
- `renderer_cuda_progressive_host.hpp` — host-side logic (SDL loop, accumulation reset) reusable
- `camera_base.hpp` / `camera.hpp` — camera frame building stays the same
- `scene_description.hpp` — scene format feeds into OptiX AS builder instead of CUDA scene builder
- `render_coordinator.hpp` — just wire new renderer
- Material parameters and physics (reflect, refract, Schlick) — port to closest-hit programs
- RNG (PCG-based fast RNG) — use in ray generation program
- All CPU renderers — untouched
- Constants, CLI, YAML loader — untouched

### New Files Needed
```
src/rayon/gpu_renderers/optix/
├── renderer_optix_host.hpp          # New IRenderer implementation
├── renderer_optix_progressive.hpp   # Interactive variant
├── optix_pipeline.cu                # Pipeline setup, SBT, module compilation
├── optix_scene_builder.cu           # SceneDescription → OptiX GAS/IAS
├── programs/
│   ├── raygen.cu                    # Ray generation (replaces renderAccKernel)
│   ├── miss.cu                      # Background color
│   ├── closest_hit_sphere.cu        # Sphere shading
│   ├── closest_hit_rectangle.cu     # Rectangle shading
│   ├── intersection_sphere.cu       # Custom sphere intersection
│   ├── intersection_rectangle.cu    # Custom rectangle intersection
│   └── intersection_sdf.cu          # SDF ray marching (custom)
```

### Build System Changes
- CMakeLists.txt: Find OptiX SDK (`find_package(OptiX)` or manual path)
- Compile OptiX programs to PTX/OptiX IR at build time
- Link against `optix` library
- OptiX is header-only on host side but needs CUDA toolkit

### Effort Estimate
- **Core pipeline** (GAS build, ray gen, miss, basic sphere closest-hit): ~3-5 days
- **All geometry types** (rectangle, cube, displaced sphere, SDF): ~2-3 days
- **All materials** (port scatter logic to closest-hit): ~1-2 days
- **Progressive/interactive mode** (accumulation, SDL integration): ~1-2 days
- **Testing and debugging**: ~2-3 days
- **Total**: ~10-15 days of focused work

### Risk: SDF Shapes
OptiX doesn't natively support SDF ray marching. Custom intersection programs can implement sphere tracing, but they run on shader cores (not RT cores), similar to Vulkan's procedural geometry. SDF performance won't improve with OptiX — only standard geometry benefits.

### Recommended Migration Strategy
1. Implement OptiX as a **new renderer** alongside existing CUDA renderers (don't replace)
2. Start with spheres-only scene to validate pipeline
3. Add geometry types incrementally
4. Port progressive/interactive mode last
5. Keep existing CUDA renderer as fallback for non-RTX GPUs

## Verification
- Compare rendered output between CUDA and OptiX renderers for identical scenes (pixel-level regression)
- Benchmark with `default_scene.yaml`, `cornell_box.yaml`, `bvh_test_scene.yaml`
- Test interactive mode FPS with OptiX progressive renderer
- Verify on both RTX 2080 and newer GPUs
