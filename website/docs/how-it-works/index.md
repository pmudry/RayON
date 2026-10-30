# How It Works

This section covers the mathematical and algorithmic foundations of RayON's rendering pipeline.
If you want to understand *why* the renderer produces the images it does, start here.

---

<div class="feature-grid" markdown>
<div class="feature-card" markdown>
**[Ray Tracing & Path Tracing](ray-tracing.md)**

The rendering equation, Monte Carlo estimation, the bounce loop, and why noise decreases with more samples.
</div>
<div class="feature-card" markdown>
**[Materials](materials.md)**

Lambertian, mirrors, glass, GGX conductors, thin films, clear coat, lights and the volume material, with the math and a YAML snippet for each.
</div>
<div class="feature-card" markdown>
**[BVH Acceleration](bvh.md)**

Surface Area Heuristic tree construction, GPU iterative traversal, and 64-byte cache-line-aligned nodes.
</div>
<div class="feature-card" markdown>
**[Hemisphere Sampling](sampling.md)**

Why cosine-weighted sampling beats uniform sampling, and how the orthonormal basis maps directions to world space.
</div>
<div class="feature-card" markdown>
**[Sobol' Quasi-Random Sampling](sobol-sampling.md)**

How 128-dimensional Joe–Kuo direction vectors and Owen scrambling deliver faster convergence than pseudo-random PCG.
</div>
<div class="feature-card" markdown>
**[Multiple Importance Sampling](mis.md)**

Next event estimation, the power heuristic, light sampling on rectangles and spheres, and how the GPU weights emitter hits.
</div>
<div class="feature-card" markdown>
**[Visual Comparisons](visual-comparisons.md)**

Four scenes with a slider: PCG vs Sobol', MIS off vs on, and both combined.
</div>
<div class="feature-card" markdown>
**[Performance Optimisations](optimizations.md)**

Every optimisation applied so far, from thread-block shape to hardware triangles, with its measured impact.
</div>
</div>
