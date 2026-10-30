# Architecture

This section explains how RayON is structured internally: how a scene is built and handed to
the two GPU back-ends, how the CUDA kernel is launched, and how the interactive renderer
accumulates samples and keeps its frame rate.

---

<div class="feature-grid" markdown>
<div class="feature-card" markdown>
**[Scene System](scene-system.md)**

The "build once, render anywhere" design: one neutral `SceneDescription` converted into flat GPU
structs for CUDA and into acceleration structures and a shader binding table for OptiX.
</div>
<div class="feature-card" markdown>
**[CUDA Renderer](cuda-renderer.md)**

32×8 thread blocks, thread→pixel mapping, random states, the float accumulation buffer, and the C++/CUDA boundary.
</div>
<div class="feature-card" markdown>
**[Progressive Rendering](progressive-rendering.md)**

The accumulation loop, adaptive sampling, the motion regulator and denoiser, the display pipeline and live parameter updates.
</div>
</div>
