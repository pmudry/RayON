# Features

Practical documentation: how to use each capability, which options exist, and where they live
in the YAML format or the GUI.

---

<div class="feature-grid" markdown>
<div class="feature-card" markdown>
**[Interactive Mode](interactive.md)**

The SDL2 window: mouse orbit, pan and zoom, keyboard shortcuts, every panel of the Dear ImGui
control panel, the motion regulator and the AI denoiser.
</div>
<div class="feature-card" markdown>
**[YAML Scene Format](scenes.md)**

Camera, settings, materials, geometry, procedural patterns, the `visible` flag, and the button
that writes the camera back into the file.
</div>
<div class="feature-card" markdown>
**[Lights & Shadows](lighting.md)**

Rectangle and sphere lights, hidden emitters, next event estimation, the light intensity and
falloff controls, and the scene that shows what NEE buys.
</div>
<div class="feature-card" markdown>
**[Volumes, Clouds & Smoke](volumes.md)**

The `volume` material: a scattering medium inside a mesh, shaped by procedural noise or by an
OpenVDB density grid, with volumetric shadow rays.
</div>
<div class="feature-card" markdown>
**[SDF Shapes](sdf-shapes.md)**

Ten ray-marched signed distance primitives, from the torus to the mandelbulb, with rotation and
named parameters, on both back-ends.
</div>
<div class="feature-card" markdown>
**[OBJ Mesh Loading](obj-loading.md)**

Wavefront meshes with MTL materials, smooth normals and UV textures; hardware triangles on OptiX,
a per-triangle BVH on CUDA.
</div>
<div class="feature-card" markdown>
**[HDR Sky Dome](hdr-sky.md)**

Equirectangular HDRI environments as the sky, cycled at runtime, stored as float16 textures with
a binary disk cache.
</div>
</div>
