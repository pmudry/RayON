---
hide:
  - toc
  - navigation
---

<div class="hero-banner">
  <img src="assets/images/samples/rayon_4k_render.png"
       alt="RayON — path traced scene of reflective and refractive spheres on a brushed-metal floor"
       loading="eager">
  <div class="hero-overlay">
    <h1>RayON</h1>
    <p>An interactive, high-performance CPU &amp; CUDA path tracer with real-time progressive sampling</p>
  </div>
</div>

## What is RayON?

RayON is an educational and experimental **path tracer** built in C++ with optional CUDA acceleration.
It started as a re-implementation of the classic
[Ray Tracing in One Weekend](https://github.com/RayTracing/raytracing.github.io) series and evolved
into a fully interactive renderer running at **100 Hz** on an NVIDIA DGX Spark.

Four rendering back-ends are available at runtime — no recompilation needed:

<div class="feature-grid" markdown>
<div class="feature-card" markdown>
**CPU Single-thread**

Reference implementation. Correct output, one pixel at a time. Good for debugging scene geometry and materials.
</div>
<div class="feature-card" markdown>
**CPU Multi-thread**

Splits the image into tiles and dispatches them across all available cores using `std::async`. Typical speedup: 8–16×.
</div>
<div class="feature-card" markdown>
**CUDA GPU**

One-shot CUDA kernel with 32×4 thread blocks, warp-friendly memory layout, and persistent `curand` states. ~100–500× faster than single-thread CPU.
</div>
<div class="feature-card" markdown>
**CUDA Interactive**

SDL2 window with progressive accumulation. Orbit, pan, zoom with the mouse. ImGui sliders for live DOF, samples, light intensity, and roughness.
</div>
<div class="feature-card" markdown>
**BVH Acceleration**

CPU-built, GPU-traversed Bounding Volume Hierarchy with Surface Area Heuristic (SAH) splitting. 5–50× speedup on scenes with 100+ objects.
</div>
<div class="feature-card" markdown>
**YAML Scenes**

Describe full scenes in plain YAML: materials, geometry, lights, camera, BVH flag. No recompile required. 17 example scenes included.
</div>
<div class="feature-card" markdown>
**OBJ Loading**

Import arbitrary triangle meshes from `.obj` files. Möller–Trumbore intersection with smooth per-vertex normals.
</div>
<div class="feature-card" markdown>
**SDF Shapes**

Ray-marched signed-distance-function primitives: torus, octahedron, death star, pyramid. Combined with analytical geometry in the same scene.
</div>
</div>

---

## Quick start

```bash
# Build (requires CMake ≥ 3.20, a C++17 compiler, and optionally CUDA + SDL2)
mkdir -p build && cd build
cmake .. --fresh
make -j$(nproc)

# Run
./rayon
# > Choose renderer: 0=CPU  1=CPU-parallel  2=CUDA  3=CUDA interactive
```

Load one of the bundled example scenes:

```bash
./rayon --scene ../resources/scenes/09_color_bleed_box.yaml -s 512 -r 1080
```

See [Getting Started](getting-started.md) for the full setup guide, or
[YAML Scene Format](features/scenes.md) to author your own scenes.

---

## Progressive rendering in action

Below is the same viewpoint at increasing sample counts. The image is accumulated on the GPU and
the display is updated live — no interruption to camera interaction.

<div class="progression-strip">
  <figure>
    <img src="assets/images/for_project/begin.png" alt="1 sample">
    <figcaption>1 SPP</figcaption>
  </figure>
  <figure>
    <img src="assets/images/for_project/end_8s.png" alt="8 samples">
    <figcaption>8 SPP</figcaption>
  </figure>
  <figure>
    <img src="assets/images/for_project/end_128s.png" alt="128 samples">
    <figcaption>128 SPP</figcaption>
  </figure>
  <figure>
    <img src="assets/images/for_project/end_256s.png" alt="256 samples">
    <figcaption>256 SPP</figcaption>
  </figure>
  <figure>
    <img src="assets/images/for_project/end_512s.png" alt="512 samples">
    <figcaption>512 SPP</figcaption>
  </figure>
  <figure>
    <img src="assets/images/for_project/end_2048s.png" alt="2048 samples">
    <figcaption>2048 SPP</figcaption>
  </figure>
</div>

Each doubling of the sample count halves visible noise — a fundamental property of Monte Carlo
integration where error decreases as \(\mathcal{O}(1/\sqrt{N})\).

---

## Explore the docs

| Section | What you'll find |
|---|---|
| [How It Works](how-it-works/index.md) | The math: ray equations, material models, BVH, sampling theory |
| [Architecture](architecture/index.md) | Code organization, CUDA renderer internals, progressive pipeline |
| [Features](features/index.md) | Interactive controls, YAML scene format, SDF shapes, OBJ loading |
| [Gallery](gallery.md) | Curated renders from all available scenes |
| [Performance](performance.md) | Benchmark results, speedup tables, tuning tips |
