# Getting Started

This page walks you through everything needed to build RayON from source and run your first render.

---

## Prerequisites

| Dependency | Version | Required? | Notes |
|---|---|---|---|
| C++ compiler | C++17 | **Yes** | Clang ≥ 14 (default) or GCC ≥ 11 |
| CMake | ≥ 3.20 | **Yes** | Ninja also supported |
| CUDA Toolkit | ≥ 12.0 | Optional | Required for GPU renderer modes 2 & 3 |
| SDL2 | ≥ 2.0 | Optional | Required for interactive mode (3) and real-time display |
| SDL2_image | ≥ 2.0 | Optional | Companion to SDL2 for texture loading |
| Dear ImGui | bundled | — | Included under `src/external/` |
| stb | bundled | — | Included under `src/external/` |

> **Without CUDA** the GPU modes are silently disabled — all CPU modes still work.
> **Without SDL2** the interactive mode is disabled but all offline render modes work.

---

## Building from source

```bash
# 1. Clone
git clone https://github.com/pmudry/RayON.git
cd RayON

# 2. Configure (generates compile_commands.json for IDE autocompletion)
mkdir -p build && cd build
cmake .. --fresh -DCMAKE_EXPORT_COMPILE_COMMANDS=1

# Copy compile commands for IDE tools
cp compile_commands.json ..

# 3. Build (uses all cores minus two)
make -j$(nproc)
```

!!! tip "VS Code users"
    Use the pre-defined tasks via `Ctrl+Shift+P → Run Task`:

    - **CMake: fresh** — clean CMake reconfigure
    - **Make build** — incremental compilation
    - **Launch** — build and run

### Build variants

```bash
# Debug build (enables -lineinfo for cuda-gdb, disables -O3)
cmake .. --fresh -DCMAKE_BUILD_TYPE=Debug

# Release build (default — -O3, best performance)
cmake .. --fresh -DCMAKE_BUILD_TYPE=Release

# Force GCC instead of Clang
cmake .. --fresh -DUSE_CLANG=OFF
```

---

## Running RayON

```bash
cd build
./rayon
```

You will be prompted to choose a renderer:

```
Choose renderer:
  0 - CPU single-threaded
  1 - CPU multi-threaded
  2 - CUDA GPU (one-shot)
  3 - CUDA GPU interactive (requires SDL2)
```

Rendered images are saved automatically to `build/rendered_images/` as timestamped
`.png` + `.json` (metadata) + `.txt` (log) triplets. `latest.png` is always overwritten.

---

## Command-line options

```bash
./rayon [options]
```

| Flag | Default | Description |
|---|---|---|
| `--scene <yaml>` | built-in | Load a scene from a YAML file |
| `-s <n>` | 64 | Samples per pixel (offline modes) |
| `-r <h>` | 720 | Image height in pixels (2160/1080/720/360/180) |
| `--start-samples <n>` | 32 | Initial SPP in interactive mode |
| `--target-fps <n>` | 60 | Target frame rate for interactive mode |
| `--adaptive-depth` | off | Progressively increase max ray bounce depth |
| `--no-auto-accumulate` | off | Disable automatic sample accumulation when stationary |
| `--help` | — | Print help and exit |

### Examples

```bash
# High-quality offline render to PNG
./rayon --scene ../resources/scenes/09_color_bleed_box.yaml -s 2048 -r 1080

# Fast preview
./rayon --scene ../resources/scenes/default_scene.yaml -s 8 -r 360

# Interactive session, starting at 32 SPP, 30 fps budget
./rayon --scene ../resources/scenes/default_scene.yaml --start-samples 32 --target-fps 30
# > choose mode 3
```

---

## Example scenes

17 pre-built YAML scenes are available in `resources/scenes/`:

| File | Highlights |
|---|---|
| `default_scene.yaml` | Full-featured default — metals, glass, area light, SDF shapes |
| `09_color_bleed_box.yaml` | Cornell box with coloured wall bleeding |
| `05_material_laboratory.yaml` | Side-by-side showcase of all material types |
| `03_platonic_solids.yaml` | SDF platonic solids — icosahedron, octahedron, torus |
| `04_obj_dragon.yaml` | Stanford dragon mesh at ~100k triangles |
| `01_anisotropic_metals_test.yaml` | Anisotropic brushed-metal highlight patterns |
| `11_soap_bubbles.yaml` | Thin-film iridescent soap bubble material |
| `12_clearcoat_pokemonball.yaml` | Clearcoat layer over diffuse base |
| `bvh_stress_courtyard.yaml` | 300+ objects — BVH stress test |

See [YAML Scene Format](features/scenes.md) for full documentation on writing your own.

---

## Output files

Every render writes to `build/rendered_images/` with the timestamp as the filename prefix:

```
output_2026-03-14_15-30-00.png   ← the rendered image
output_2026-03-14_15-30-00.json  ← render metadata (samples, time, resolution, scene)
output_2026-03-14_15-30-00.txt   ← ASCII log
latest.png                        ← always points to the most recent render
```
