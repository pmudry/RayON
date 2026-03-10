# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RayON is a high-performance path tracer (C++17/CUDA) with CPU, multi-threaded, and GPU backends, plus interactive SDL2 real-time rendering. Educational project for ISC 302 HPC course, based on "Ray Tracing in One Weekend." Version 1.2.4, licensed GNU AGPL v3.

## Build Commands

```bash
# Full clean build (from project root)
cd build && cmake .. --fresh && make -j && cd ..

# Incremental build
make -C build -j

# Debug build
cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j && cd ..

# Run (from build/)
./rayon -m <0-3> -s <samples> -r <resolution> --scene <yaml_file>
#   -m 0: CPU sequential, 1: CPU parallel, 2: CUDA, 3: CUDA+SDL interactive
```

After adding new files/directories under `src/`, re-run `cmake .. --fresh` to update includes and `.clangd`.

## Code Style

- **Formatter**: `.clang-format` — LLVM base, 120-column limit, 3-space indent, Allman braces, no tabs
- **Static analysis**: `.clang-tidy` runs automatically if installed (disable with `ENABLE_CLANG_TIDY=OFF`)
- **Header-only**: Most classes live in `.hpp` files; `.cu` files only for CUDA kernels
- **Flat includes**: `#include "camera.hpp"` not `#include "../camera/camera.hpp"` (CMake auto-discovers include dirs)

## Architecture

### "Build Once, Render Anywhere" Scene System

Central hub: `Scene::SceneDescription` (`src/rayon/scenes/scene_description.hpp`).

1. **Host-side**: SceneDescription built on CPU (from YAML or programmatically)
2. **CPU path**: Converted to polymorphic `Hittable_list` via `CPUSceneBuilder`
3. **GPU path**: Converted to flat `CudaScene::Scene` struct via `CudaSceneBuilder::buildGPUScene()`

GPU cannot use virtual functions, so CPU uses polymorphic classes while GPU uses flat structs with enum-based type dispatch.

### Renderer Backends

All unified through `Camera` class (virtual inheritance from all renderer bases to avoid diamond problem with `CameraBase`):

| Backend | File | Notes |
|---------|------|-------|
| CPU Sequential | `cpu_renderers/renderer_cpu_single_thread.hpp` | Reference/debug |
| CPU Parallel | `cpu_renderers/renderer_cpu_parallel.hpp` | `std::async` |
| CUDA | `gpu_renderers/renderer_cuda_host.hpp` | Batch |
| CUDA Progressive | `gpu_renderers/renderer_cuda_progressive_host.hpp` | Interactive SDL2 |

### Key Source Layout

```
src/rayon/
├── main.cc                    # Entry point, CLI parsing
├── constants.hpp              # Version, resolution, quality defaults
├── camera/                    # Camera + SDL interactive controls
├── data_structures/           # vec3, ray, hittable, material, color, interval
├── cpu_renderers/             # CPU backends + cpu_shapes/
├── gpu_renderers/             # CUDA backends + shaders/
│   └── shaders/               # Kernels: render_scene_kernel.cu, render_acc_kernel.cu, shader_golf.cu
├── render/                    # RenderCoordinator, IRenderer interface, RenderTarget
└── scenes/                    # SceneDescription, SceneBuilder, SceneFactory, YAML loader
```

### BVH Acceleration

- Built on CPU with Surface Area Heuristic: `SceneDescription::buildBVH()`
- Traversed on GPU with iterative stack in `shader_common.cuh`
- Enable: `scene_desc.use_bvh = true` or `use_bvh: true` in YAML

### Precision Split

CPU uses `double` (Vec3), GPU uses `float` — conversion at kernel boundary.

## Adding New Geometry or Materials

**Geometry**: Add enum to `GeometryType` in `scene_description.hpp` → add to `GeometryDesc` → implement CPU intersection as `Hittable` subclass → implement GPU intersection in `shader_common.cuh::intersect_geometry()` → add factory method to `SceneDescription`.

**Material**: Add enum to `MaterialType` → add params to `MaterialDesc` → implement CPU scattering in `material.hpp` → implement GPU evaluation in `shader_common.cuh::evaluate_material()`.

## CUDA Patterns

- C++/CUDA boundary: device functions in `.cu`, exposed via `extern "C"`
- All CUDA calls wrapped with `CUDA_CHECK()` macro (`cuda_utils.cu`)
- Kernel block size: 32×8 (128 threads) for memory coalescing
- Separable compilation enabled for cross-file device functions
- `curand` states persist across progressive frames (allocated once, reused)

## Gotchas

- No exceptions, STL containers, or virtual functions in GPU kernels
- BVH: both tree structure and geometry must be copied to device
- After adding files to `src/`, must `cmake .. --fresh` to regenerate includes and compile_commands.json
- Random states must persist across frames for progressive rendering
- Gamma correction differs between interactive display and saved images
