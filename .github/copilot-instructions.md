# RayON - AI Coding Agent Instructions

## Documentation Sync Policy

**After every significant code change, update the relevant page(s) in `website/docs/`.**

| Code change | Doc page(s) to update |
|---|---|
| New CLI flag in `main.cc` | `website/docs/getting-started.md` — command reference table |
| New material type / enum | `website/docs/how-it-works/materials.md`, `website/docs/features/scenes.md` |
| New geometry type | `website/docs/architecture/scene-system.md` |
| New YAML key / scene format | `website/docs/features/scenes.md` |
| New SDF shape | `website/docs/features/sdf-shapes.md` |
| Performance change / benchmark result | `website/docs/performance.md` |
| BVH changes | `website/docs/how-it-works/bvh.md` |
| CUDA kernel / architecture change | `website/docs/architecture/cuda-renderer.md` |
| Interactive / SDL change | `website/docs/features/interactive.md` |
| OBJ / mesh loading change | `website/docs/features/obj-loading.md` |
| New render sample worthy of gallery | `website/docs/gallery.md` |

After editing docs, always run `cd website && uv run mkdocs build --strict` to verify no broken references.

## Project Overview
High-performance path tracer (C++17/CUDA) with GPU (CUDA and OptiX) backends, plus interactive SDL2 real-time rendering. Educational ISC 302 HPC project based on "Ray Tracing in One Weekend." Version 1.5.5, licensed GNU GPL v3.

> **Note:** The original CPU rendering backends (sequential and multi-threaded) have been moved to the `legacy/cpu-renderer` branch.

## Architecture: The Unified Scene System

**Critical Concept**: The project uses a "build once, render anywhere" architecture centered around `Scene::SceneDescription` (`src/rayon/scenes/scene_description.hpp`).

### Scene Flow
1. **Host-side construction**: `Scene::SceneDescription` built on CPU (from YAML or programmatically in `main.cc`)
2. **GPU rendering**: Converted to flat `CudaScene::Scene` structure via `CudaSceneBuilder::buildGPUScene()` in `scene_builder_cuda.cu`

**Why**: GPU cannot use virtual functions or polymorphism, so it uses flat structs with enum-based type discrimination.

### BVH Acceleration
- **Built on CPU**: `SceneDescription::buildBVH()` uses Surface Area Heuristic (SAH) for optimal partitioning
- **Traversed on GPU**: Iterative stack-based traversal in `cuda_raytracer.cuh`
- **Enable with**: `scene_desc.use_bvh = true` or `use_bvh: true` in YAML scenes
- See `explanations/BVH_ACCELERATION.md` for implementation details

## Build System

### CMake Configuration
```bash
# From project root (clean build recommended)
cd build
cmake .. --fresh -DCMAKE_EXPORT_COMPILE_COMMANDS=1 && cp compile_commands.json ..
make -j8  # Or use ninja for faster builds
```

### VS Code Tasks
Use predefined tasks (Ctrl+Shift+P → Run Task):
- **CMake: fresh**: Clean CMake rebuild
- **Make build**: Incremental compilation
- **Launch**: Build and run

### Build Types
- **Release** (default): `-O3` optimization, `-DCMAKE_BUILD_TYPE=Release`
- **Debug**: Debug symbols, `-DCMAKE_BUILD_TYPE=Debug`

### Key CMake Patterns
- **CUDA optional**: Falls back gracefully if `CMAKE_CUDA_COMPILER` not found
- **SDL2 optional**: Real-time display disabled without `SDL2_FOUND` define
- **Compiler choice**: Set `USE_CLANG=OFF` for GCC instead of default Clang
- **CUDA architectures**: `all-major` for broad GPU compatibility
- **Include directories**: Flat includes (`#include "camera.hpp"` not `#include "../camera/camera.hpp"`)

## CUDA Programming Patterns

### C++ ↔ CUDA Boundary
- **Principle**: Keep CUDA code in `.cu` files, expose via `extern "C"` functions
- **Example**: `renderer_cuda_device.cu` exports `renderPixelsCUDAAccumulative()`, called from C++ `renderer_cuda_host.hpp`
- **State transfer**: Use `cudaMemcpyToSymbol()` for global GPU constants (see `setLightIntensity()`)

### GPU Memory Management
```cpp
// Typical pattern in renderer_cuda_device.cu
CudaScene::Scene* allocateAndTransferScene(const Scene::SceneDescription& desc) {
    CudaScene::Scene* d_scene;
    cudaMalloc(&d_scene, sizeof(CudaScene::Scene));
    // ... convert and copy materials, geometries, BVH
    return d_scene;
}
```

### Kernel Launch Configuration
- **Standard grid**: `dim3 block_size(32, 8)` (256 threads) - rectangular for memory coalescing
- **Separable compilation**: Required for device functions across files (`CUDA_SEPARABLE_COMPILATION ON`)
- **Random states**: Persistent device memory for `curand` states across accumulative renders

## Interactive SDL Rendering

### Progressive Accumulation Mode
Enabled with `-m 3` (CUDA+SDL) or `-m 5` (OptiX+SDL, if built with OPTIX):
```bash
./rayon -m 3 --target-fps 60 --samples-per-batch 50 --adaptive-depth
```

**Controls** (implemented in `camera/sdl/sdl_gui_handler.hpp` / `sdl_gui_controls.hpp`):
- **Left mouse**: Orbit camera (rotate around look-at)
- **Right mouse**: Pan (translate look-at point)
- **Mouse wheel**: Zoom (distance from look-at)
- **Space**: Force re-render
- **GUI sliders**: Adjust samples, light intensity, DOF, etc.

**Accumulation logic** (`renderer_cuda_progressive_host.hpp`):
- Low samples during camera motion (auto-scaled to hit `--target-fps`)
- Automatic accumulation when stationary (up to `INTERACTIVE_MAX_SPP`)
- Adaptive depth increases ray bounce limit progressively

## Scene Definition

### YAML Scene Files
Scene files live in `resources/scenes/`. The format is documented in `resources/scenes/SCENE_FORMAT.md`.

```bash
./rayon --scene resources/scenes/default_scene.yaml
```

**Structure** (abbreviated):
```yaml
camera:
  position: [0, 1, 5]
  look_at: [0, 0, 0]
  fov: 60

settings:
  use_bvh: true
  background_color: [0.05, 0.05, 0.08]

materials:
  - name: "mat_name"
    type: "lambertian"  # mirror, rough_mirror, metal, glass, dielectric, light,
                        # anisotropic_metal, thin_film, clear_coat, show_normals
    albedo: [r, g, b]
    roughness: 0.3  # for rough_mirror / metal

geometry:
  - type: "sphere"
    material: "mat_name"
    center: [x, y, z]
    radius: r
```

**Loader**: `src/rayon/scenes/yaml_scene_loader.cc` - lightweight custom parser (no external deps)

### Programmatic Scene Building
Scenes can also be built in code via `SceneDescription` API:
```cpp
int mat = scene_desc.addMaterial(MaterialDesc::roughMirror(Vec3(1,0.85,0.47), 0.03));
scene_desc.addSphere(Vec3(-3.5, 0.45, -1.8), 0.8, mat);
scene_desc.addRectangle(Vec3(-1,3,-2), Vec3(2.5,0,0), Vec3(0,0,1.5), light_mat); // Area light
```

## Code Organization Conventions

### Header-Only Implementation
Most classes are header-only for template/device code compatibility. Implementation in `.cu` files for CUDA kernels only.

### Virtual Inheritance Pattern
`Camera` class uses virtual inheritance to combine rendering backends:
```cpp
class Camera : public RendererCUDA, public RendererCUDAProgressive
```
All inherit virtually from `CameraBase` to avoid diamond problem.

### Renderer Separation
- `camera/camera_base.hpp`: Core camera parameters (look_from, look_at, FOV, pixel_delta_u/v)
- `gpu_renderers/renderer_cuda_host.hpp`: One-shot CUDA render
- `gpu_renderers/renderer_cuda_progressive_host.hpp`: Interactive SDL + accumulation

### Shader / GPU Code Organization
CUDA device code split into modules under `src/rayon/gpu_renderers/`:
- `cuda_raytracer.cuh`: Core ray tracing logic — ray-geometry intersection, BVH traversal, material dispatch
- `cuda_raytracer.cu`: Main CUDA kernel entry points
- `materials/material_dispatcher.cuh`: Material evaluation dispatcher
- `materials/legacy/`: Individual material implementations (lambertian, glass, rough_mirror, etc.)
- `shaders/render_acc_kernel.cu`: Accumulative progressive kernel
- `shaders/shader_golf.cu`: SDF ray marching for procedural shapes

## Material & Geometry System

### Material Types (enum `MaterialType` in `src/rayon/scenes/scene_description.hpp`)
- **LAMBERTIAN**: Diffuse (cosine-weighted hemisphere sampling)
- **METAL**: Metallic with fuzziness
- **MIRROR**: Perfect specular reflection
- **ROUGH_MIRROR**: Microfacet with roughness parameter
- **GLASS / DIELECTRIC**: Refraction with Schlick's approximation
- **LIGHT**: Emissive (importance sampled in area lights)
- **CONSTANT / SHOW_NORMALS**: Debug/diagnostic materials
- **ANISOTROPIC_METAL**: Physically-based anisotropic conductor (GGX)
- **THIN_FILM**: Thin-film interference (soap bubbles, oil slicks)
- **CLEAR_COAT**: Two-layer: glossy dielectric coat over diffuse base
- **SDF_MATERIAL**: Used for ray-marched SDF objects

Procedural patterns (`ProceduralPattern` enum): FIBONACCI_DOTS, CHECKERBOARD, STRIPES

### Geometry Types (enum `GeometryType` in `src/rayon/scenes/scene_description.hpp`)
- **Primitives**: SPHERE, DISPLACED_SPHERE (golf ball), RECTANGLE, CUBE, TRIANGLE
- **Mesh**: OBJ_MESH — loaded via `src/rayon/scenes/obj_loader.hpp`
- **Ray marched**: SDF_PRIMITIVE (torus, octahedron, pyramid) — see `shaders/shader_golf.cu`
- **Acceleration**: BVHNODE (internal, not added by users)

### SDF Shapes
Ray marched using sphere tracing in `shaders/shader_golf.cu`. Rotation support via Euler angles. Examples: `addSDFTorus()`, `addSDFOctahedron()`.

## Performance Characteristics

### Expected Speedups
- **With BVH**: 5-50× improvement for complex scenes (100+ objects)
- **CUDA vs CPU baseline**: ~100-500× (depends on GPU, scene complexity)

### Optimization Flags
- **CUDA**: `--use_fast_math` (disabled by default), `-O3`, `--expt-relaxed-constexpr`
- **CPU**: `-O3` in Release, architecture-specific optimizations via Clang/GCC
- **Parallel builds**: CMake uses `N-2` cores automatically

## Common Development Patterns

### Adding New Geometry
1. Add enum to `GeometryType` in `src/rayon/scenes/scene_description.hpp`
2. Add struct to `GeometryDesc` union
3. Implement GPU intersection in `gpu_renderers/cuda_raytracer.cuh::intersect_geometry()`
4. Add factory method `SceneDescription::addMyShape()`

### Adding New Material
1. Add enum to `MaterialType` in `src/rayon/scenes/scene_description.hpp`
2. Add parameters to `MaterialDesc` struct
3. Add GPU evaluation in `gpu_renderers/materials/material_dispatcher.cuh`

### Debugging CUDA Kernels
- **Compile Debug**: `-DCMAKE_BUILD_TYPE=Debug` enables `-lineinfo` for cuda-gdb
- **Check errors**: All CUDA calls wrapped with `CUDA_CHECK()` macro in `cuda_utils.cuh`
- **Atomic counters**: Ray count tracked via `atomicAdd()` for validation

## Command Line Arguments
```bash
-m <method>              # 2=CUDA, 3=CUDA+SDL,
                         #   4=OptiX offline (if built with OPTIX), 5=OptiX+SDL (if built with OPTIX)
-s <samples>             # Samples per pixel (default: SAMPLES_PER_PIXEL)
-r <WxH>|<height>        # Resolution: e.g. 1920x1080 or 720 (16:9)
--scene <yaml_file>      # Load scene from YAML (files in resources/scenes/)
--samples-per-batch <n>  # Max samples per interactive batch (auto-scales to hit --target-fps)
--target-fps <fps>       # Interactive mode target FPS (default: 60)
--adaptive-depth         # Progressive max_depth increase
--no-adaptive-sampling   # Disable converged-pixel skipping
--no-auto-accumulate     # Disable automatic sample accumulation when stationary
--theme <name>           # GUI theme: light, classic, nord, dracula, gruvbox, catppuccin
--menu                   # Show interactive method selection menu
```

## Testing Scenes
Provided in `resources/scenes/`:
- `default_scene.yaml`: Full-featured default scene
- `01_bvh_test_scene.yaml`: BVH acceleration testing with many objects
- `05_material_laboratory.yaml`: All material types showcase
- `11_soap_bubbles.yaml`: Thin-film interference (iridescent bubbles)
- `12_clearcoat_pokemonball.yaml`: Clear-coat material demo
- `bvh_stress_courtyard.yaml`: High-triangle-count BVH stress test

## Key Files Reference
- **Main entry**: `src/rayon/main.cc`
- **Scene hub**: `src/rayon/scenes/scene_description.hpp` — unified scene format (read this first!)
- **Scene factory**: `src/rayon/scenes/scene_factory.hpp` — functions to create scenes (from YAML or programmatically)
- **GPU scene**: `src/rayon/gpu_renderers/cuda_scene.cuh` — flat GPU-friendly scene structs
- **CUDA ray tracer**: `src/rayon/gpu_renderers/cuda_raytracer.cuh` — intersection, BVH traversal, shading
- **Material dispatcher**: `src/rayon/gpu_renderers/materials/material_dispatcher.cuh`
- **YAML parser**: `src/rayon/scenes/yaml_scene_loader.cc`
- **Scene YAML format**: `resources/scenes/SCENE_FORMAT.md`

## Common Gotchas
- **Device code restrictions**: No exceptions, no STL containers, no virtual functions in GPU kernels
- **Random states**: Must persist across frames for progressive rendering (allocated once, reused)
- **BVH transfer**: Both tree structure and geometry must be copied to device
- **Float precision**: GPU uses `float`, CPU uses `double` - conversion happens at kernel boundary
- **Compile commands**: Must regenerate with `--fresh` after adding new files to CMakeLists.txt

## Documentation
Detailed explanations in `explanations/`:
- `CUDA_RENDER_EXPLANATION.md`: Kernel launch mechanics
- `BVH_ACCELERATION.md`: SAH algorithm and traversal
- `PROGRESSIVE_SDL_RENDERING.md`: Interactive mode architecture
- `YAML_SCENE_LOADER.md`: Scene file format
