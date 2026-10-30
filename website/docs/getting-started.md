# Getting Started

This page walks you through building RayON from source, running your first render, and the
command-line options.

---

## Prerequisites

| Dependency | Version | Required? | Notes |
|---|---|---|---|
| C++ compiler | C++17 | **Yes** | Clang (default, `USE_CLANG=ON`) or GCC |
| CMake | ≥ 3.20 | **Yes** | Generates `compile_commands.json` for clangd |
| CUDA Toolkit | ≥ 12.0 | **Yes** | Both back-ends are CUDA programs; the architecture is set to `native` |
| NVIDIA OptiX SDK | 8.0 or 9.0 | Optional | Enables modes 4 and 5 (hardware ray tracing). Found through `OptiX_INSTALL_DIR`, `~/Downloads/NVIDIA-OptiX-SDK-*`, `/usr/local/NVIDIA-OptiX-SDK` or `/opt/optix` |
| SDL2 | ≥ 2.0 | Optional | Enables the interactive modes 3 and 5 |
| Dear ImGui, stb | bundled | — | Included under `src/external/` |

Without SDL2 the offline modes still work. Without the OptiX SDK only the CUDA back-end is built.

---

## Building from source

```bash
git clone https://github.com/pmudry/RayON.git
cd RayON

mkdir -p build && cd build
cmake .. --fresh          # prints which of SDL2 and OptiX were found
make -j$(nproc)
```

!!! tip "VS Code users"
    Use the pre-defined tasks via `Ctrl+Shift+P → Run Task`: **CMake: fresh**, **Make build**
    and **Launch**. `clangd` picks up `compile_commands.json` and the `.clangd` file that CMake
    maintains with the include paths of every directory under `src/`.

### Build variants

```bash
cmake .. --fresh -DCMAKE_BUILD_TYPE=Debug      # -lineinfo for cuda-gdb and Nsight, no fast math
cmake .. --fresh -DCMAKE_BUILD_TYPE=Release    # default: -O3 and --use_fast_math
cmake .. --fresh -DUSE_CLANG=OFF               # GCC instead of Clang for the host code
cmake .. --fresh -DENABLE_CLANG_TIDY=ON        # static analysis during the build (.clang-tidy)
cmake .. --fresh -DENABLE_DIAGS=ON             # extra diagnostic output in the renderers
```

After adding files or directories under `src/`, re-run `cmake .. --fresh` so the include paths
and `compile_commands.json` are regenerated.

---

## Running RayON

!!! warning "Run from `build/`"
    The OptiX programs (`optix_programs.ptx`) and the HDR skies (`../resources/hdri/`) are
    located relative to the current directory. Every example below assumes `cd build`.

```bash
cd build
./rayon
```

With no `-m` the renderer picks the best mode it was built with: **5** (OptiX interactive) when
both OptiX and SDL2 are available, else 4, else 3, else 2.

```bash
./rayon -m 2          # CUDA offline
./rayon -m 3          # CUDA interactive (SDL2)
./rayon -m 4          # OptiX offline
./rayon -m 5          # OptiX interactive (SDL2 + OptiX)
./rayon --menu        # ask which mode to use
```

Offline renders are written to `build/rendered_images/` as a timestamped `.png` with a `.json`
metadata file and a `.txt` log next to it; `latest.png` always holds the most recent image. The
offline progress bar shows the elapsed time and an estimate of the time left.

---

## Command-line options

```bash
./rayon [options]
```

| Flag | Default | Description |
|---|---|---|
| `-m <method>` | 5 (3 without OptiX) | Renderer: `2` = CUDA offline, `3` = CUDA interactive, `4` = OptiX offline, `5` = OptiX interactive |
| `--menu` | — | Show the mode selection menu instead of using the default |
| `--scene <yaml>` | built-in | Load a scene from a YAML file |
| `-s <n>` | 64 | Samples per pixel (offline modes 2, 4) |
| `-r <h>` | 720 | Image height; the width follows the 16:9 aspect ratio (`-r 1080`) |
| `-r <WxH>` | — | Arbitrary resolution (`-r 1920x1080`, `-r 800x600`) |
| `--sampler <sobol\|pcg>` | `sobol` | GPU sampler: low-discrepancy Sobol' or classic PCG |
| `--no-hdr-cache` | off | Always re-decode `.hdr` files instead of reading the `.hdrcache` sidecar |
| `--sky <name>` | gradient | Interactive modes: start with this HDR sky of `resources/hdri/` (part of the file name, e.g. `--sky sunflowers`) |
| `--samples-per-batch <n>` | 8 | Interactive samples per batch; overrides the scene's `settings.samples_per_batch` |
| `--no-adaptive-sampling` | off | Disable converged-pixel skipping |
| `--no-auto-accumulate` | off | Do not keep adding samples while the camera is still |
| `--theme <name>` | `nord` | GUI theme: `light`, `classic`, `nord`, `dracula`, `gruvbox`, `catppuccin` |
| `-h`, `--help` | — | Print the help and exit |

### MIS / NEE options (all modes)

| Flag | Default | Description |
|---|---|---|
| `--no-mis` | MIS on | Disable multiple importance sampling and next event estimation (faster per sample, much noisier) |
| `--motion-gate-mis` | off | Disable MIS while the camera moves (interactive modes only) |
| `--nee-stride <N>` | `1` | Beyond the first bounce, trace the shadow ray on 1 of every N samples and scale its contribution by N |

The target frame rate while moving, the quality floors, the denoiser and the adaptive sampling
threshold are GUI settings, see [Interactive Mode](features/interactive.md).

### Examples

```bash
# High-quality offline render with OptiX
./rayon -m 4 --scene ../resources/scenes/09_color_bleed_box.yaml -s 2048 -r 1080

# Fast preview with CUDA
./rayon -m 2 --scene ../resources/scenes/default_scene.yaml -s 8 -r 360

# Interactive session on the cloud bunny
./rayon --scene ../resources/scenes/17_cloud_bunny.yaml

# Interactive with a custom theme and no adaptive sampling
./rayon --theme dracula --no-adaptive-sampling
```

### Interactive mode on a remote GPU server

The SDL window does not work through `ssh -X`. Start a VNC session on the server instead:

```bash
sudo apt install tigervnc-standalone-server openbox   # once, on the server
scripts/rayon_vnc.sh -r 1080 --scene resources/scenes/18_sdf_shapes.yaml
```

The script starts a headless VNC server sized to the render resolution, runs `rayon` inside it
and prints the `vncviewer` command to connect from your machine. `Ctrl-C` stops everything.

---

## Example scenes

The 24 scenes in `resources/scenes/` double as a feature tour. Load any of them with `--scene`.

| File | Highlights |
|---|---|
| `default_scene.yaml` | The built-in scene: rough mirrors, glass, anisotropic gold and copper, a Fibonacci-dotted sphere, an area light |
| `default_scene_no_ambient.yaml` | The same without the sky, lit by the area light alone |
| `01_anisotropic_metals_test.yaml` | 4×4 grid of aluminium spheres: anisotropy along X, roughness along Y |
| `01_anisotropic_scene_obj.yaml` | The same grid on cylinder meshes (OBJ), a check of anisotropic shading on polygons |
| `01_bvh_test_scene.yaml` | 150 spheres in a grid, a BVH sanity scene |
| `02_displacement_garden.yaml` | Procedurally displaced spheres (golf-ball dimples) |
| `03_platonic_solids.yaml` | The five Platonic solids as OBJ meshes with anisotropic metals |
| `04_obj_dragon.yaml`, `04_obj_statue.yaml`, `04_obj_test_scene.yaml` | OBJ loading: the xyzrgb dragon (250 k triangles), the Winged Victory statue, a test sphere |
| `05_material_laboratory.yaml` | Rough mirrors, anisotropic metals and glass side by side |
| `06_caustics_chapel.yaml` | Glass spheres in a dark room with a small bright light: caustics and the hardest case for sampling |
| `09_color_bleed_box.yaml` | Cornell box with red, green and blue walls |
| `10_isc_neons_bunny.yaml` | Five coloured neon strips lighting the Stanford bunny |
| `11_soap_bubbles.yaml` | Thin-film interference at several film thicknesses |
| `12_clearcoat_pokemonball.yaml` | Clear coat over a diffuse base, with Suzanne |
| `13_texture_test1.yaml`, `14_texture_test2.yaml` | UV-mapped OBJ models with image textures through MTL files |
| `15_mis_furnace_test.yaml` | Furnace test: energy conservation of the MIS estimator |
| `16_nee_small_light.yaml` | Five matte spheres under a tiny bulb: what next event estimation buys |
| `17_cloud_bunny.yaml` | The volume material: the bunny as a cumulus, procedural shape noise |
| `18_sdf_shapes.yaml` | Gallery of the ten ray-marched SDF primitives in a studio room |
| `19_vdb_bunny.yaml` | The OpenVDB `bunny_cloud` density grid (needs a one-time conversion, see [Volumes](features/volumes.md)) |
| `20_emissive_mesh.yaml` | A glowing OBJ bunny beside a tiny bulb: emissive meshes light the scene through bounces, NEE samples only the bulb |
| `bvh_stress_courtyard.yaml` | 385 metallic spheres in a pyramid, a BVH stress test |
| `99_gnd.yaml` | Scratch scene |

The format is documented in [YAML Scene Format](features/scenes.md).
