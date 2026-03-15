# Architecture

This section explains how RayON is structured internally — how the scene is built, how the
CUDA kernel launches work, and how the progressive renderer accumulates samples.

---

<div class="feature-grid" markdown>
<div class="feature-card" markdown>
**[Scene System](scene-system.md)**

The "build once, render anywhere" design: one neutral `SceneDescription` converted into CPU class
hierarchy or GPU flat structs.
</div>
<div class="feature-card" markdown>
**[CUDA Renderer](cuda-renderer.md)**

32×4 thread blocks, thread→pixel mapping, curand states, float accum buffer, and the C++/CUDA boundary.
</div>
<div class="feature-card" markdown>
**[Progressive Rendering](progressive-rendering.md)**

The accumulation loop, tiled rendering, sample stages, adaptive depth, and live parameter injection.
</div>
</div>
