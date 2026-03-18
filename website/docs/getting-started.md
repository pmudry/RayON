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

When SDL2 is available, RayON defaults directly to **interactive mode (mode 3)** — no prompt.
When only CUDA is available, it defaults to the one-shot CUDA renderer (mode 2).

To pick a renderer explicitly, use `-m`:

```bash
./rayon -m 0          # CPU single-thread
./rayon -m 1          # CPU parallel
./rayon -m 2          # CUDA one-shot
./rayon -m 3          # CUDA interactive (SDL2 required)
```

Or add `--menu` to see the interactive selection prompt:

```bash
./rayon --menu
# Choose rendering method:
#   0. CPU sequential
#   1. CPU parallel
#   2. CUDA GPU (default)
#   3. CUDA GPU with interactive SDL display
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
| `-m <method>` | 3 (or 2) | Renderer: `0`=CPU seq, `1`=CPU par, `2`=CUDA, `3`=CUDA interactive, `4`=OptiX, `5`=OptiX interactive |
| `--menu` | — | Show interactive renderer-selection menu instead of using the default |
| `--scene <yaml>` | built-in | Load a scene from a YAML file |
| `-s <n>` | 64 | Samples per pixel (offline modes 0–2, 4) |
| `-r <h>` | 720 | Image height; aspect ratio 16:9 auto-applied (`-r 1080`) |
| `-r <WxH>` | — | Arbitrary resolution (`-r 1920x1080`, `-r 800x600`) |
| `--samples-per-batch <n>` | 50 | Quality ceiling for the adaptive batch scheduler (interactive mode) |
| `--target-fps <n>` | 60 | Target frame rate; batch size auto-scales every frame to meet this budget |
| `--adaptive-depth` | off | Progressively increase max ray bounce depth |
| `--no-adaptive-sampling` | off | Disable converged-pixel skipping (adaptive sampling) |
| `--no-auto-accumulate` | off | Disable automatic sample accumulation when stationary |
| `--no-hdr-cache` | off | Disable `.hdrcache` disk cache — always re-decode the `.hdr` file from scratch |
| `--theme <name>` | `nord` | ImGui colour theme: `light`, `classic`, `nord`, `dracula`, `gruvbox`, `catppuccin` |
| `-h`, `--help` | — | Print help and exit |

### Examples

```bash
# High-quality offline render to PNG
./rayon -m 2 --scene ../resources/scenes/09_color_bleed_box.yaml -s 2048 -r 1080

# Fast preview (offline)
./rayon -m 2 --scene ../resources/scenes/default_scene.yaml -s 8 -r 360

# Interactive session, 30 fps budget, adaptive depth
./rayon --scene ../resources/scenes/default_scene.yaml --target-fps 30 --adaptive-depth
# > interactive window opens directly (mode 3)

# Interactive with custom theme and no adaptive sampling
./rayon --theme dracula --no-adaptive-sampling
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
