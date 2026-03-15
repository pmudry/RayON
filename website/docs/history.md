# Development History

A walkthrough of how RayON grew from a 53-line `main.cc` into a multi-backend interactive GPU
path tracer — told through the git history.

The project spans roughly six months, from a first "hello sphere" on a September evening to a
full NVIDIA OptiX hardware-accelerated renderer the following spring. Each milestone below
corresponds to a real commit (or small cluster of commits) and captures both what was built and
why it mattered at that stage.

---

## Milestone 1 — The First Pixel (Sept 7–8, 2025) { #first-pixel }

**Commit:** [`7901473`](https://github.com/pmudry/RayON/commit/7901473) → [`b6af112`](https://github.com/pmudry/RayON/commit/b6af112)

The repository starts with a CMakeLists, a `.gitignore`, and a `main.cc` that is exactly 53 lines
long. The core idea is already present — cast a ray for each pixel, check whether it hits a
sphere, and write a colour. The shading at this point is purely a linear remap of the ray-to-sphere
distance: no lighting, no materials, just depth turned into a grey gradient.

What makes this milestone interesting is what is *not* there yet: there is no `Vec3` class (3D
vectors are three naked floats), no camera abstraction, no material system, the sky gradient is
hardcoded. The image is written using `stb_image_write.h`, still the output library in use today.
The first render shows a single large sphere, obviously not perspective-correct, with depth shading
that makes it look like a shaded disc.

!!! quote "First commit message"
    *"first commit"* — September 7, 2025

**What changed:** project scaffolding, CMake, `stb_image`, first ray–sphere intersection, first
rendered pixel.

---

## Milestone 2 — Normals, Perspective & Reflections (Sept 8–11, 2025) { #normals }

**Commits:** [`a8f08e7`](https://github.com/pmudry/RayON/commit/a8f08e7) · [`4740f42`](https://github.com/pmudry/RayON/commit/4740f42)

Within the first four days, the renderer gains the key ingredients of a recognisable path tracer.

- **Normal-to-colour shading** — mapping the surface normal vector to RGB gives the characteristic
  blue-tinted sphere that features in nearly every "Ray Tracing in One Weekend" tutorial.
- **Camera refactoring** — `look_from`, `look_at`, and `vfov` are extracted into a proper class.
  Perspective is now correct.
- **Reflections** — a basic mirror material is added. The timing infrastructure also appears here:
  each render prints elapsed milliseconds.

By September 11, the scene already has multiple spheres with different materials, and the output
image is recognisably a physically-based render rather than a depth map.

**What changed:** `Vec3` class, ray abstraction, camera class, normal-to-RGB shading, mirror
reflection, multi-sphere scene, wall-clock timing.

---

## Milestone 3 — First CUDA Kernel (Sept 15–16, 2025) { #first-cuda }

**Commit:** [`671f69d`](https://github.com/pmudry/RayON/commit/671f69d)

> *"Block rendering with CUDA — Final version to be cleaned-up — 38e9 rays in 1 minute"*

This is the GPU ignition moment. A first `.cu` file appears alongside `main.cc`, and the same
sphere scene that was taking seconds on the CPU is handed off to blocks of CUDA threads. The commit
message records a key benchmark: **38 thousand rays per minute** — not fast by later standards,
but proof that the GPU path was working.

The CUDA code at this stage is rough. All scene data is encoded as literals inside the kernel.
There is no material system on the GPU side yet — just a hardcoded normal-to-colour calculation
mirroring the CPU. The architecture of a separate `.cu` file talking to a `.h` host file through
an `extern "C"` boundary — a pattern that scales all the way to the OptiX integration months
later — is established here.

The documentation in `explanations/CUDA_RENDER_EXPLANATION.md` is also written at this commit,
showing that the explanation-driven development style starts early.

**What changed:** first `.cu` CUDA kernel, GPU ray-sphere intersection, `extern "C"` host/device
boundary, CUDA block/grid launch configuration, first GPU timing measurement.

---

## Milestone 4 — Golf Ball & Procedural Displacement (Sept 18–19, 2025) { #golf-ball }

**Commits:** [`44b3421`](https://github.com/pmudry/RayON/commit/44b3421) · [`8f76e3a`](https://github.com/pmudry/RayON/commit/8f76e3a) · [`2fa8d0f`](https://github.com/pmudry/RayON/commit/2fa8d0f)

> *"Geo displacement"* then *"With dots !"* then *"Very satisfying one"*

The first genuinely fun visual moment: a sphere whose surface is perturbed by a procedural
displacement function, producing the characteristic dimpled pattern of a golf ball. The commit
messages read like a lab notebook — brief, excited, chronological.

The tiled CUDA renderer (rendering the image in tiles rather than all pixels at once) is merged
with the regular CUDA path at this point, reducing memory pressure for larger resolutions. A
big refactoring pass also separates the two renderers properly so neither needs to know how the
other works.

**What changed:** procedural displacement mapping (Fibonacci dot pattern), tiled CUDA rendering,
merged CUDA renderer, renderer separation into distinct compilation units.

---

## Milestone 5 — Material System Refactoring (Sept 26, 2025) { #materials }

**Commits:** [`fb88041`](https://github.com/pmudry/RayON/commit/fb88041) · [`511956`](https://github.com/pmudry/RayON/commit/511956) · [`9134c1c`](https://github.com/pmudry/RayON/commit/9134c1c)

A refactoring week brings the CPU material system to the shape it recognisably holds today.
Lambertian diffuse, metals, and a first "constant" (flat-colour) material are given proper
abstractions. The parallel CPU renderer gets a progress bar. Normals are now visualisable as
colours separately from proper shading.

This is also the point where the CPU path becomes clearly a reference implementation for
correctness — run slowly, check against it, trust its output.

**What changed:** proper `Material` base class, Lambertian/Metal subclasses, constant material,
normal-as-colour diagnostic mode, parallel CPU renderer with progress bar.

---

## Milestone 6 — Interactive SDL Window (Nov 3–5, 2025) { #interactive }

**Commits:** [`cd710bb`](https://github.com/pmudry/RayON/commit/cd710bb) · [`47da519`](https://github.com/pmudry/RayON/commit/47da519) · [`c9bf459`](https://github.com/pmudry/RayON/commit/c9bf459)

> *"Add progressive SDL2 rendering with real-time quality improvements"*  
> *"Add interactive camera controls to SDL progressive rendering"*  
> *"Even better accumulative samples"*

This is one of the most significant feature additions in the project's history: a live SDL2 window
that shows the render improving over time. The first version displays the image after each of the
sample stages 1 → 4 → 16 → 64 → 256 → 1024, with the user able to press ESC and keep the current
quality image. Two days later, interactive camera controls (mouse orbit, pan, scroll-to-zoom)
are added.

The `renderPixelsSDLProgressive()` method — originally just 141 lines in `camera.h` — becomes the
foundation for everything interactive that follows. The key architectural decision made here is
that the GPU accumulation buffer lives on the device and is never unnecessarily copied to host
until display time; a pattern that contributes directly to the 2× speedup achieved four months
later.

**What changed:** SDL2 window integration, progressive sample-over-time display, interactive
camera orbit/pan/zoom, accumulation buffer on device, real-time quality/speed tradeoff.

---

## Milestone 7 — The Big November Sprint: BVH, SDF, DOF, Cornell Box (Nov 7, 2025) { #november-sprint }

**Commits (all on the same day):**
[`2dfd645`](https://github.com/pmudry/RayON/commit/2dfd645) BVH ·
[`254adfa`](https://github.com/pmudry/RayON/commit/254adfa) SDF shapes ·
[`042fc38`](https://github.com/pmudry/RayON/commit/042fc38) SDF rotations ·
[`58b0e34`](https://github.com/pmudry/RayON/commit/58b0e34) Depth-of-field ·
[`91f0872`](https://github.com/pmudry/RayON/commit/91f0872) YAML scene loader + Cornell box

November 7 is the project's single densest development day. The BVH acceleration structure
(Bounding Volume Hierarchy with Surface Area Heuristic), SDF ray-marched procedural shapes,
depth-of-field with a thin-lens model, and the YAML scene file loader all land within hours of
each other. The classic Cornell box scene also appears as a validation target.

The BVH commit alone adds 833 lines across 7 files, including a full SAH implementation and
iterative stack-based GPU traversal. The technical documentation in `explanations/BVH_ACCELERATION.md`
(189 lines) is written in the same commit.

The SDF shapes (torus, octahedron, death-star) use sphere-tracing ray marching and work on both
CPU and GPU. The rotation support arrives two commits later the same day.

The YAML loader gives scenes a stable, human-readable format that can be shared and versioned
alongside the renderer.

**What changed:** BVH with SAH (CPU build, GPU traversal), SDF procedural shapes with rotations,
depth of field, YAML scene format, Cornell box reference scene, `bvh_test_scene.yaml` with 178
objects for performance testing.

---

## Milestone 8 — Lambertian Sampling Deep Dive (Nov 13–15, 2025) { #lambertian }

**Commits:** [`c37a0a1`](https://github.com/pmudry/RayON/commit/c37a0a1) · [`ce5c2d1`](https://github.com/pmudry/RayON/commit/ce5c2d1) · [`dd39b2f`](https://github.com/pmudry/RayON/commit/dd39b2f)

> *"Lambertian new implementation, using cosine weighted sampling — Included functions for building
> orthonormal bases"*  
> *"Refactor Lambertian material to include Owen scrambling for improved randomness and
> stratification in sampling"*

A careful revisit of the Lambertian diffuse model, motivated by discrepancies in the rendered
images. Two implementations are compared side by side: the naïve hemisphere sampling and a
cosine-weighted version using an orthonormal basis constructed from the surface normal. Owen
scrambling — a quasi-random sequence technique — is added to reduce low-frequency noise patterns
at low sample counts.

This milestone is typical of the educational philosophy behind RayON: not just making it work,
but understanding *why* it works, and documenting the difference an algorithmic choice makes on
the output.

**What changed:** cosine-weighted hemisphere sampling, orthonormal basis construction,
Owen scrambling for stratified quasi-random sampling, two-implementation comparison.

---

## Milestone 9 — A Name is Born: RayON (Nov 17, 2025) { #naming }

**Commits:** [`a52ee87`](https://github.com/pmudry/RayON/commit/a52ee87) · [`ce66234`](https://github.com/pmudry/RayON/commit/ce66234) · [`1b8ff65`](https://github.com/pmudry/RayON/commit/1b8ff65)

> *"A project name!"*

After ten weeks of development under the working title "302_raytracer", the project becomes
**RayON** — a nod to the ray tracing core and the HES-SO Valais backdrop, where the project
serves as a teaching vehicle for the CS302 HPC course. The name change is accompanied by the
addition of the GNU GPL v3 licence and a major README overhaul.

The `clang-tidy` static analyser is integrated into the build at this milestone, adding automated
code quality checks that run at compile time.

**What changed:** project renamed to RayON, GPL v3 licence, README rewrite, clang-tidy
integration, build system quality-of-life improvements.

---

## Milestone 10 — Multi-Platform CUDA (Nov 27, 2025) { #multiplatform }

**Commit:** [`ada3e77`](https://github.com/pmudry/RayON/commit/ada3e77)

> *"Tested on AMD64 Linux and fixed for older GPU issue with memory location (no longer shared for
> RTX2080) — Fixed clangd for Linux ARM64 (ignore CUDA_ARCHITECTURE)"*

The renderer is tested on two hardware platforms back to back: an AMD x86-64 Linux machine with an
older RTX 2080 GPU, and an ARM64 Linux machine. Both surface different bugs. The RTX 2080 cannot
use unified memory for some allocations that worked transparently on newer hardware; those
allocations are rerouted. The ARM64 clangd integration requires a guard to skip the
`CUDA_ARCHITECTURE` flag (which is x86-only).

This is the last commit of 2025, representing a natural plateau — the renderer is stable,
multi-platform, and has most of its core features.

**What changed:** RTX 2080 memory allocation fix, ARM64 clangd compatibility, better diagnostics
for cross-platform debugging, JSON render statistics with timestamps.

---

## Milestone 11 — CUDA 2.24× Speedup & Dear ImGui (March 9–10, 2026) { #cuda-speedup }

**Commits:** [`1f5929a`](https://github.com/pmudry/RayON/commit/1f5929a) · [`dafa369`](https://github.com/pmudry/RayON/commit/dafa369) · [`264e525`](https://github.com/pmudry/RayON/commit/264e525) · [`433000f`](https://github.com/pmudry/RayON/commit/433000f)

> *"CUDA renderer optimizations: 2.24x speedup (6.08s → 2.71s)"*  
> *"Replace hand-made SDL GUI with Dear ImGui for interactive renderer"*

After a three-month pause, development resumes with an intense performance sprint.

**CUDA optimisations** (two commits): the D2H (device-to-host) round-trip that was being performed
every frame is eliminated. The material array is flattened into a plain struct-of-arrays layout
that maps directly to GPU cache lines. BVH traversal is restructured to reduce warp divergence.
The result: the benchmark scene drops from 6.08 s to 2.71 s — a **2.24× improvement** with no
change to output quality.

**Dear ImGui** replaces the hand-written SDL2 button/slider panel that had been accumulating
technical debt since November. The new GUI has collapsible sections, live performance graphs,
and proper input capture. SDL2_TTF (font rendering) is dropped as a dependency. The version
number jumps to 1.5.0.

**What changed:** eliminated D2H round-trip, flattened material arrays, reduced BVH warp
divergence, Dear ImGui integration with collapsible panels and live SPP/ms graphs, version 1.5.0.

---

## Milestone 12 — Adaptive Sampling & Normal Visualisation (March 12, 2026) { #adaptive }

**Commits:** [`1bc7183`](https://github.com/pmudry/RayON/commit/1bc7183) · [`78bd8de`](https://github.com/pmudry/RayON/commit/78bd8de) · [`18c8498`](https://github.com/pmudry/RayON/commit/18c8498)

Per-pixel convergence tracking is implemented: each pixel records a running variance estimate
and stops accumulating samples early once it has converged. Noisy pixels (typically those with
difficult light paths) continue accumulating while already-converged areas freeze. The result is
an effective speedup for scenes where some regions converge quickly (flat walls, large lights)
while others are slow (specular caustics, glass edges).

The normal arrows overlay arrives in the same few days: a debug visualisation that draws a small
3D arrow from each surface intersection point in the direction of the shading normal. This makes
it immediately obvious when a mesh has flipped or missing normals — an invaluable diagnostic for
the triangle pipeline work about to begin.

Scene switching from within the interactive GUI is added, allowing different YAML scenes to be
loaded and rendered without restarting the program.

**What changed:** per-pixel adaptive convergence sampling, normal arrows 3D overlay, in-GUI
scene switching, scene selection controls in interactive mode.

---

## Milestone 13 — Triangle Pipeline & OBJ Loading (March 13, 2026) { #triangles }

**Commit:** [`e383e83`](https://github.com/pmudry/RayON/commit/e383e83)

> *"Triangle-based pipeline integration — Added OBJ models rendering and import."*

The renderer can now load `.obj` files and render proper triangle meshes. A Möller–Trumbore
intersection test is implemented for triangles on both CPU and GPU. Smooth normals (interpolated
from vertex normals in the `.obj` file) are supported. The Platonic solids (tetrahedron, cube,
octahedron, dodecahedron, icosahedron) appear as a showcase scene, each rendered with a
different material.

A Python script (`generate_platonic_solids.py`) generates the `.obj` files procedurally, and
the accompanying YAML scene file wires them together with the new OBJ loader. The project's
`resources/models/` directory starts filling up.

**What changed:** `triangle.hpp` intersection, `obj_loader.hpp` mesh parser, smooth normal
interpolation, five Platonic solid scene files, `generate_platonic_solids.py` generator.

---

## Milestone 14 — Anisotropic Metals, Thin-Film & Clear-Coat (March 13–14, 2026) { #anisotropic }

**Commits:** [`8623b79`](https://github.com/pmudry/RayON/commit/8623b79) · [`cca43c1`](https://github.com/pmudry/RayON/commit/cca43c1) · [`1f7a83d`](https://github.com/pmudry/RayON/commit/1f7a83d)

> *"Anisotropic metals, stage 1"*  
> *"Add thin-film interference material and CUDA streams"*  
> *"Clear-coat and thin-film shading — Car paint, soap bubbles, and demo scenes!"*

The most physically rich material work in the project lands in a two-day burst.

**Anisotropic metals** use a full GGX microfacet BRDF with separate tangential and bitangential
roughness parameters. The specular highlight stretches into a horizontal or vertical streak
depending on the anisotropy ratio — the effect that makes brushed aluminium look brushed.
A dedicated `microfacet_ggx.cuh` (213 lines) implements the distribution, geometry, and Fresnel
terms.

**Thin-film interference** models the iridescent colour shift seen on soap bubbles, oil films,
and camera lenses. The colour of the highlight shifts with viewing angle via an optical path
difference calculation over a thin dielectric layer. CUDA streams are added alongside this to
overlap kernel execution and memory transfers.

**Clear-coat** adds a specular dielectric layer on top of a base material — the standard
automotive paint model. A Shelby Cobra `.obj` model and a PokéBall are added to the resources
for demonstration.

**What changed:** GGX anisotropic BRDF, thin-film interference shader, clear-coat multi-layer
material, CUDA streams, Shelby Cobra and PokéBall OBJ models, dedicated demo scenes for each
material.

---

## Milestone 15 — NVIDIA OptiX: Hardware Ray Tracing (March 15, 2026) { #optix }

**Commits:** [`8ec565e`](https://github.com/pmudry/RayON/commit/8ec565e) · [`d831935`](https://github.com/pmudry/RayON/commit/d831935)

> *"First rendering in OptiX"*  
> *"First metrics for OptiX: 4x faster for non-interactive rendering on dragon scene! Impressive!"*

The final milestone integrates NVIDIA OptiX — the hardware-accelerated ray tracing API that
exposes the dedicated RT cores on Turing/Ampere/Ada GPUs. Instead of a custom BVH traversed by
CUDA threads, OptiX hands the acceleration structure and traversal off to fixed-function hardware.

The result on the dragon scene (a high-polygon mesh that taxes the BVH heavily): **4× faster
than CUDA** for offline rendering. The integration is non-trivial: OptiX programs are compiled
separately as PTX and loaded at runtime; a new `renderer_optix_progressive_host.hpp` wraps the
OptiX pipeline; all the same material evaluation code is shared with the CUDA path through
common `.cuh` headers.

The OptiX renderer is an optional build target — the renderer detects whether the OptiX SDK is
present at CMake time and falls back to CUDA gracefully if it is not.

**What changed:** full NVIDIA OptiX integration (`.ptx` programs, `OptixPipeline`,
`OptixShaderBindingTable`), hardware BVH traversal via RT cores, optional CMake build path,
4× speedup on complex mesh scenes.

---

## The Timeline at a Glance

```
Sep  7 ━━ First pixel (sphere, depth shading)
Sep  8 ━━ Normals, perspective, camera class
Sep 11 ━━ Reflections, timing, multi-sphere
Sep 16 ━━ First CUDA kernel (38k rays/min)
Sep 19 ━━ Golf ball, tiled CUDA, displacement
Sep 26 ━━ Material system, parallel CPU render
           ┄┄┄ (break) ┄┄┄
Nov  3 ━━ Interactive SDL2 window
Nov  5 ━━ Accumulative progressive rendering
Nov  7 ━━ BVH · SDF shapes · DOF · YAML · Cornell box
Nov 13 ━━ Lambertian sampling deep dive
Nov 17 ━━ RayON name, GPL licence, clang-tidy
Nov 27 ━━ Multi-platform (AMD64 + ARM64)
           ┄┄┄ (break) ┄┄┄
Mar  9 ━━ CUDA 2.24× speedup
Mar 10 ━━ Dear ImGui, version 1.5.0
Mar 12 ━━ Adaptive sampling, normal arrows
Mar 13 ━━ Triangle/OBJ pipeline, Platonic solids
Mar 13 ━━ Anisotropic metals, thin-film, clear-coat
Mar 15 ━━ NVIDIA OptiX — 4× speedup (hardware RT)
```

---

## Milestone Explorer (Planned)

Each milestone above has a corresponding **commit hash** that can be checked out, compiled, and
run to experience the renderer at that exact point in its evolution. A set of scripts in
`scripts/milestones/` is planned that will:

1. `git stash` any in-progress work, then `git checkout <hash>`
2. Run a clean CMake configure and build (with the correct flags for the feature set of that era)
3. Launch the renderer — either in **offline mode** (writing a reference PNG) or in
   **interactive mode** (opening the SDL2 window where available)
4. Restore the original branch on exit

The planned milestone checkpoints and their intended demo modes are:

| # | Date | Commit | Demo mode | What to see |
|---|------|--------|-----------|-------------|
| 1 | 2025-09-08 | `b6af112` | offline | First sphere — depth pseudo-shading |
| 2 | 2025-09-11 | `4740f42` | offline | Normals, reflections, multi-sphere |
| 3 | 2025-09-16 | `671f69d` | offline | First CUDA render |
| 4 | 2025-09-19 | `2fa8d0f` | offline | Golf ball displacement pattern |
| 5 | 2025-09-26 | `fb88041` | offline | Material system — diffuse + metal |
| 6 | 2025-11-05 | `c9bf459` | interactive | Progressive SDL2 accumulation |
| 7 | 2025-11-07 | `2dfd645` | interactive | BVH on, Cornell box scene |
| 8 | 2025-11-17 | `a52ee87` | interactive | RayON 1.0 — full feature set |
| 9 | 2025-11-27 | `ada3e77` | interactive | Multi-platform stable release |
| 10 | 2026-03-10 | `264e525` | interactive | Dear ImGui, v1.5.0 |
| 11 | 2026-03-13 | `8623b79` | interactive | Anisotropic metals scene |
| 12 | 2026-03-13 | `e383e83` | interactive | OBJ loading — Platonic solids |
| 13 | 2026-03-13 | `1f7a83d` | interactive | Thin-film / clear-coat materials |
| 14 | 2026-03-15 | `8ec565e` | offline | First OptiX render |
| 15 | 2026-03-15 | `d831935` | offline | OptiX 4× speedup on dragon mesh |

!!! info "Coming soon"
    The `scripts/milestones/` restore scripts are in progress. When available, running
    `scripts/milestones/goto_milestone.sh <N>` will check out, build, and launch the correct
    version automatically.
