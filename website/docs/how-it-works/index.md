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

Lambertian, mirror, rough mirror, glass, and area light — full math and YAML snippet for each.
</div>
<div class="feature-card" markdown>
**[BVH Acceleration](bvh.md)**

Surface Area Heuristic tree construction, GPU iterative traversal, and 64-byte cache-line-aligned nodes.
</div>
<div class="feature-card" markdown>
**[Hemisphere Sampling](sampling.md)**

Why cosine-weighted sampling beats uniform sampling, and how the orthonormal basis maps directions to world space.
</div>
</div>
