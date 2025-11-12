# 302 Raytracer - AI Coding Agent Instructions

## Project Overview
High-performance path tracer with CPU, multi-threaded, and CUDA GPU implementations. Educational CS302 HPC project based on "Ray Tracing in One Weekend" series, featuring interactive SDL real-time rendering with progressive sampling.

## Architecture: The Unified Scene System

**Critical Concept**: The project uses a "build once, render anywhere" architecture centered around `Scene::SceneDescription` (`src/302_raytracer/scene_description.h`).

### Scene Flow
1. **Host-side construction**: `Scene::SceneDescription` built on CPU (from YAML or programmatically in `main.cc::create_scene_description()`)
2. **CPU rendering**: Converted to polymorphic `Hittable_list` via `CPUSceneBuilder::buildCPUScene()`
3. **GPU rendering**: Converted to flat `CudaScene::Scene` structure via `CUDASceneBuilder::buildGPUScene()` in `scene_builder_cuda.cu`

**Why**: GPU cannot use virtual functions or polymorphism, so CPU uses classes (Sphere, Rectangle, Material) while GPU uses flat structs with enum-based type discrimination.

### BVH Acceleration
- **Built on CPU**: `SceneDescription::buildBVH()` uses Surface Area Heuristic (SAH) for optimal partitioning
- **Traversed on GPU**: Iterative stack-based traversal in `shader_common.cuh::trace_ray_scene_bvh()`
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
- **Include directories**: Flat includes (`#include "camera.h"` not `#include "../camera/camera.h"`)

## CUDA Programming Patterns

### C++ ↔ CUDA Boundary
- **Principle**: Keep CUDA code in `.cu` files, expose via `extern "C"` functions
- **Example**: `renderer_cuda.cu` exports `renderPixelsCUDAAccumulative()`, called from C++ `renderer_cuda.h`
- **State transfer**: Use `cudaMemcpyToSymbol()` for global GPU constants (see `setLightIntensity()`)

### GPU Memory Management
```cpp
// Typical pattern in renderer_cuda.cu
CudaScene::Scene* allocateAndTransferScene(const Scene::SceneDescription& desc) {
    CudaScene::Scene* d_scene;
    cudaMalloc(&d_scene, sizeof(CudaScene::Scene));
    // ... convert and copy materials, geometries, BVH
    return d_scene;
}
```

### Kernel Launch Configuration
- **Standard grid**: `dim3 block_size(32, 4)` (128 threads) - rectangular for memory coalescing
- **Separable compilation**: Required for device functions across files (`CUDA_SEPARABLE_COMPILATION ON`)
- **Random states**: Persistent device memory for `curand` states across accumulative renders

## Interactive SDL Rendering

### Progressive Accumulation Mode
Enabled with option 3 in runtime menu (requires `SDL2_FOUND`):
```bash
./302_raytracer --target-fps 60 --start-samples 32 --adaptive-depth
```

**Controls** (implemented in `sdl_gui_handler.h` / `sdl_gui_controls.h`):
- **Left mouse**: Orbit camera (rotate around look-at)
- **Right mouse**: Pan (translate look-at point)
- **Mouse wheel**: Zoom (distance from look-at)
- **Space**: Force re-render
- **GUI sliders**: Adjust samples, light intensity, DOF, etc.

**Accumulation logic** (`renderer_cuda_progressive.h`):
- Low samples during camera motion (e.g., 8 samples)
- Automatic accumulation when stationary (up to `max_samples`)
- Adaptive depth increases ray bounce limit progressively

## Scene Definition

### YAML Scene Files
Load custom scenes: `./302_raytracer --scene resources/cornell_box.yaml`

**Structure** (see `resources/default_scene.yaml`):
```yaml
materials:
  - name: "mat_name"
    type: "lambertian"  # rough_mirror, glass, light, etc.
    albedo: [r, g, b]
    roughness: 0.3  # for rough_mirror

geometry:
  - type: "sphere"
    material: "mat_name"
    center: [x, y, z]
    radius: r
```

**Loader**: `yaml_scene_loader.cc` - lightweight custom parser (no external deps)

### Programmatic Scene Building
Add geometry in `main.cc::create_scene_description()`:
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
class Camera : public RendererCPU, public RendererCPUParallel, 
               public RendererCUDA, public RendererCUDAProgressive
```
All inherit virtually from `CameraBase` to avoid diamond problem.

### Renderer Separation
- `camera_base.h`: Core camera parameters (look_from, look_at, FOV, pixel_delta_u/v)
- `renderer_cpu.h`: Single-threaded `renderPixels()`
- `renderer_cpu_parallel.h`: Thread pool with `std::async`
- `renderer_cuda.h`: One-shot CUDA render
- `renderer_cuda_progressive.h`: Interactive SDL + accumulation

### Shader Organization
CUDA device code split into modules:
- `shader_common.cuh`: Material evaluation, ray-geometry intersection
- `render_scene_kernel.cu`: Main ray tracing kernel
- `render_acc_kernel.cu`: Accumulative progressive kernel
- `shader_golf.cu`: SDF ray marching for procedural shapes

## Material & Geometry System

### Material Types (enum in `scene_description.h`)
- **LAMBERTIAN**: Diffuse (cosine-weighted hemisphere sampling)
- **ROUGH_MIRROR**: Microfacet with roughness parameter
- **GLASS**: Refraction with Schlick's approximation
- **LIGHT**: Emissive (importance sampled in area lights)
- **Procedural patterns**: FIBONACCI_DOTS, CHECKERBOARD (stored in `MaterialDesc`)

### Geometry Types
- **Primitives**: SPHERE, RECTANGLE, CUBE, DISPLACED_SPHERE (golf ball)
- **Ray marched**: SDF_PRIMITIVE (torus, octahedron, pyramid) - see `sdf_shape.h`
- **TODO**: TRIANGLE_MESH (structure exists, not fully implemented)

### SDF Shapes
Ray marched using sphere tracing in `shader_golf.cu`. Rotation support via Euler angles. Examples: `addSDFTorus()`, `addSDFDeathStar()`, `addSDFOctahedron()`.

## Performance Characteristics

### Expected Speedups
- **CPU → Multi-threaded CPU**: ~8-16× (depends on core count)
- **CPU → CUDA**: ~100-500× (depends on GPU, scene complexity)
- **With BVH**: 5-50× improvement for complex scenes (100+ objects)

### Optimization Flags
- **CUDA**: `--use_fast_math` (disabled by default), `-O3`, `--expt-relaxed-constexpr`
- **CPU**: `-O3` in Release, architecture-specific optimizations via Clang/GCC
- **Parallel builds**: CMake uses `N-2` cores automatically

## Common Development Patterns

### Adding New Geometry
1. Add enum to `GeometryType` in `scene_description.h`
2. Add struct to `GeometryDesc` union
3. Implement CPU intersection in `hittable.h` subclass
4. Implement GPU intersection in `shader_common.cuh::intersect_geometry()`
5. Add factory method `SceneDescription::addMyShape()`

### Adding New Material
1. Add enum to `MaterialType`
2. Add parameters to `MaterialDesc` struct
3. Implement CPU scattering in `material.h` subclass
4. Implement GPU evaluation in `shader_common.cuh::evaluate_material()`

### Debugging CUDA Kernels
- **Compile Debug**: `-DCMAKE_BUILD_TYPE=Debug` enables `-lineinfo` for cuda-gdb
- **Check errors**: All CUDA calls wrapped with `CUDA_CHECK()` macro in `cuda_utils.cuh`
- **Atomic counters**: Ray count tracked via `atomicAdd()` for validation

## Command Line Arguments
```bash
-s <samples>           # Samples per pixel (default: constants.h::SAMPLES_PER_PIXEL)
-r <height>            # Resolution (2160/1080/720/360/180)
--scene <yaml_file>    # Load scene from YAML
--start-samples <n>    # Initial samples for interactive mode (default: 32)
--target-fps <fps>     # Interactive mode target FPS (default: 60)
--adaptive-depth       # Progressive max_depth increase
--no-auto-accumulate   # Disable auto sample accumulation when stationary
```

## Testing Scenes
Provided in `resources/`:
- `default_scene.yaml`: Full featured (11 materials, SDFs, area lights)
- `cornell_box.yaml`: Classic validation scene
- `simple_scene.yaml`: Minimal 3-sphere test
- `bvh_test_scene.yaml`: Performance testing with many objects

## Key Files Reference
- **Main entry**: `src/302_raytracer/main.cc`
- **Scene hub**: `src/302_raytracer/scene_description.h` (857 lines - read this first!)
- **GPU scene**: `src/302_raytracer/gpu_renderers/cuda_scene.cuh`
- **CUDA kernels**: `src/302_raytracer/gpu_renderers/shaders/render_scene_kernel.cu`
- **BVH builder**: `scene_description.h::buildBVH()` and `buildBVHRecursive()`
- **YAML parser**: `src/302_raytracer/yaml_scene_loader.cc`

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
