# TODO List

## CUDA renderer
- [X] Textures loading (Venturi's style)
- [ ] Volumetric smoke ray-marching
- [ ] Depth map exporting
- [ ] New sphere / point light light types
- [ ] SDF integration in CUDA
- [ ] SDL3 migration (won't probably fix)

## General
- [ ] Better skybox
- [ ] Skybox as HDR, dynamic loading

## Website
- [ ] History script testing

## Others things
- [ ] Impact of different rendering optimizations reporting -> find a way and report (for instance, color complexity per pixel)
- [ ] `sdl` texture blitting integration + performance assessment
- [ ] Performance regression
- [ ] Pipeline CI/CD pour runner les tests unitaires ci-dessus (si possible, besoin CUDA sur Github)
- [ ] Profiling and low-level optimizations + document changes using `nsight-systems` and `nsight-compute`
- [ ] Implement NVIDIA OptiX image denoising
- [ ] Save image as OpenEXR format

## Done

### Interactive renderer
- [x] Gamma correction in interactive renderer is wrong when displayed, but saved correctly -> it's related to how program handle color profiles.
- [x] Change of speeds in interactive renderer, as they are not very nice
- [x] Add a ray/second counter somewhere -> SPPS instead, as ray counting is too costly on gpu side

### CUDA renderer
- [x] Implement anisotropic metals / shading
- [x] Refactor constants f3_ones, f3_zero and others
- [x] Time per pixel shading for performance display
- [x] Normals as lines for spheres
- [x] Normals color toggle for interactive renderer

### General code organization
- [x] There are still discrepancies for the cuda renderers
    - [x] clarify renderer_cuda_host.hpp vs renderer_cuda_device.cu responsibilities
    - [x] cuda_raytracer name is badly chosen

### General
- [x] Doxygen documentation

### Scenes
- [x] YAML scene should take camera positions
- [x] Dynamic scenes loading

### Optimisations
- [x] Fast maths

### Others things
- [x] `ImGUI` GUI integration for controls -> reimplement existing controls + add more
- [x] Implement benchmarks for static renderer on typical scenes (multiple renders + average)
- [x] arbitrary resolution (other than 16/9) 


# Image recovery 

Permanently lost (14 images — never committed to git, not on disk anywhere):

```
Image	Milestone
dev/m1_first_pixel.png	M1 — First pixel
dev/m1_sphere_depth.png	M1 — First pixel
dev/m2_normals.png	M2 — Normals
dev/output_circular_spheres.png	M5 — Materials
dev/output0_multicore.png	M5 — Materials
dev/interactive.png	M6 — Interactive SDL
dev/adaptive_sampling0/1/2.png	M12 — Adaptive sampling
dev/first_triangle_step_0/2/3.png	M13 — Triangles
samples/Screenshot from 2026-03-15 00-33-03.png	M15 — OptiX
samples/Screenshot from 2026-03-15 00-54-56.png	M15 — OptiX
These were local screenshots/renders that were referenced in the doc but never git add-ed. You'd need to regenerate them — either by checking out the old milestone commits and running the renderer, or by re-running the current renderer for the OptiX screenshots.
```

in dev: 

```
m1_first_pixel.png, m1_sphere_depth.png — run milestone 1 (b6af112)
m2_normals.png — run milestone 2 (4740f42)
output_circular_spheres.png, output0_multicore.png — run milestone 5 (fb88041)
interactive.png — screenshot of milestone 6 SDL window (c9bf459)
adaptive_sampling0.png, adaptive_sampling1.png, adaptive_sampling2.png — progressive screenshots from milestone 12
first_triangle_step_0.png, first_triangle_step_2.png, first_triangle_step_3.png — milestone 13 triangle pipeline stages
samples
Screenshot from 2026-03-15 00-33-03.png — first OptiX render
Screenshot from 2026-03-15 00-54-56.png — OptiX dragon scene
```

The goto_milestone.sh script should make regenerating most of these straightforward. Once you have them, just drop them into the images directories and the hooks.py will sync them into images on the next build.