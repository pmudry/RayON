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
into a fully interactive raytracer running at **> 100 FPS @ 720p** on an NVIDIA DGX Spark.

Four rendering back-ends are available at runtime — no recompilation needed:

<div class="feature-grid" markdown>
<div class="feature-card" markdown>
**CPU Single-thread**

Reference implementation on the CPU, one pixel at a time. It was the starting point but is in reality useless
</div>
<div class="feature-card" markdown>
**CPU Multi-thread**

Splits the image into tiles and dispatches them across all available cores using `std::async`. Typical speedup: 8–16×, depending on your CPU.
</div>
<div class="feature-card" markdown>
**CUDA GPU**

One-shot CUDA kernel with 32×4 thread blocks, warp-friendly memory layout, and persistent `curand` states. ~100–500× faster than single-thread CPU.
</div>
<div class="feature-card" markdown>
**CUDA Interactive**

SDL2 window with progressive accumulation. Orbit, pan, zoom with the mouse. ImGui sliders for live DOF, samples, light intensity, and roughness.
</div>
</div>

It also features **BVH Acceleration**, CPU-built, with GPU-traversed Bounding Volume Hierarchy with Surface Area Heuristic (SAH) splitting. This provides 5–50× speedup on scenes with 100+ objects.

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

## Explore the docs

| Section | What you'll find |
|---|---|
| [How It Works](how-it-works/index.md) | The math: ray equations, material models, BVH, sampling theory |
| [Architecture](architecture/index.md) | Code organization, CUDA renderer internals, progressive pipeline |
| [Features](features/index.md) | Interactive controls, YAML scene format, SDF shapes, OBJ loading |
| [Gallery](gallery.md) | Curated renders from all available scenes |
| [Performance](performance.md) | Benchmark results, speedup tables, tuning tips |
