# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RayON is a high-performance path tracer (C++17/CUDA) with GPU (CUDA and OptiX) backends, plus interactive SDL2 real-time rendering. Educational project for ISC 302 HPC course, based on "Ray Tracing in One Weekend." Version 1.5.5, licensed GNU GPL v3.

> **Note:** The original CPU rendering backends (sequential and multi-threaded) have been moved to the `legacy/cpu-renderer` branch.

## Build Commands

```bash
# Full clean build (from project root)
cd build && cmake .. --fresh && make -j && cd ..

# Incremental build
make -C build -j

# Debug build
cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j && cd ..

# Run (from build/)
./rayon -m <2-5> -s <samples> -r <resolution> --scene <yaml_file>
#   -m 2: CUDA, 3: CUDA+SDL interactive,
#       4: OptiX offline (if built with OPTIX), 5: OptiX+SDL interactive (if built with OPTIX)
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
2. **GPU path**: Converted to flat `CudaScene::Scene` struct via `CudaSceneBuilder::buildGPUScene()`

GPU cannot use virtual functions, so it uses flat structs with enum-based type dispatch.

### Renderer Backends

All unified through `Camera` class (virtual inheritance from all renderer bases to avoid diamond problem with `CameraBase`):

| Backend | File | Notes |
|---------|------|-------|
| CUDA | `gpu_renderers/renderer_cuda_host.hpp` | Batch |
| CUDA Progressive | `gpu_renderers/renderer_cuda_progressive_host.hpp` | Interactive SDL2 |
| OptiX | `gpu_renderers/renderer_optix_host.hpp` | Batch (if built with OPTIX) |
| OptiX Progressive | `gpu_renderers/renderer_optix_progressive_host.hpp` | Interactive SDL2 (if built with OPTIX) |

### Key Source Layout

```
src/rayon/
├── main.cc                    # Entry point, CLI parsing
├── constants.hpp              # Version, resolution, quality defaults
├── camera/                    # Camera + SDL interactive controls
├── data_structures/           # vec3, ray, hittable, material, color, interval, rnd_gen
├── gpu_renderers/             # CUDA backends + materials/ + shaders/ + optix/
│   ├── cuda_raytracer.cu      # Main CUDA ray tracing kernel + intersection/shading logic
│   ├── materials/             # GPU material system (material_dispatcher.cuh, legacy/)
│   ├── shaders/               # Additional kernels: render_acc_kernel.cu, shader_golf.cu
│   └── optix/                 # OptiX programs and renderer (if built with OPTIX)
├── render/                    # RenderCoordinator, IRenderer interface, RenderTarget
└── scenes/                    # SceneDescription, SceneBuilder, SceneFactory, YAML loader
```

### BVH Acceleration

- Built on CPU with Surface Area Heuristic: `SceneDescription::buildBVH()`
- Traversed on GPU with iterative stack in `cuda_raytracer.cuh`
- Enable: `scene_desc.use_bvh = true` or `use_bvh: true` in YAML

### Precision Split

CPU uses `double` (Vec3), GPU uses `float` — conversion at kernel boundary.

## Adding New Geometry or Materials

**Geometry**: Add enum to `GeometryType` in `scene_description.hpp` → add to `GeometryDesc` → implement GPU intersection in `cuda_raytracer.cuh::intersect_geometry()` → add factory method to `SceneDescription`.

**Material**: Add enum to `MaterialType` → add params to `MaterialDesc` → implement GPU evaluation in `gpu_renderers/materials/material_dispatcher.cuh`.

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

## Command Line Arguments

```bash
-m <method>              # Rendering method: 2=CUDA, 3=CUDA+SDL,
                         #   4=OptiX offline (if built with OPTIX), 5=OptiX+SDL (if built with OPTIX)
-s <samples>             # Samples per pixel for offline modes (default: SAMPLES_PER_PIXEL)
-r <WxH>|<height>        # Resolution: WxH (e.g. 1920x1080) or height for 16:9 (e.g. 720)
--scene <yaml_file>      # Load scene from YAML (files live under resources/scenes/)
--samples-per-batch <n>  # Max samples per interactive batch (auto-scales to hit --target-fps)
--target-fps <fps>       # Interactive mode target FPS (default: 60)
--adaptive-depth         # Progressively increase max ray bounce depth
--no-adaptive-sampling   # Disable converged-pixel skipping
--no-auto-accumulate     # Disable automatic sample accumulation when stationary
--theme <name>           # GUI theme: light, classic, nord, dracula, gruvbox, catppuccin
--menu                   # Show interactive method selection menu
```

## Material & Geometry Types

### Material Types (enum `MaterialType` in `scene_description.hpp`)
LAMBERTIAN, METAL, MIRROR, ROUGH_MIRROR, GLASS, DIELECTRIC, LIGHT, CONSTANT, SHOW_NORMALS,
SDF_MATERIAL, ANISOTROPIC_METAL, THIN_FILM, CLEAR_COAT

Procedural patterns (stored in `MaterialDesc`): FIBONACCI_DOTS, CHECKERBOARD, STRIPES

### Geometry Types (enum `GeometryType` in `scene_description.hpp`)
SPHERE, DISPLACED_SPHERE, RECTANGLE, TRIANGLE, OBJ_MESH, CUBE, BVHNODE, SDF_PRIMITIVE

## Scene Files

YAML scene files live in `resources/scenes/`. The format is documented in `resources/scenes/SCENE_FORMAT.md`.
Notable scenes: `default_scene.yaml`, `01_bvh_test_scene.yaml`, `05_material_laboratory.yaml`,
`11_soap_bubbles.yaml` (thin-film), `12_clearcoat_pokemonball.yaml` (clear-coat).
