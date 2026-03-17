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