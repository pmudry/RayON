# TODO List

## Interactive renderer
- [x] Gamma correction in interactive renderer is wrong when displayed, but saved correctly -> it's related to how program handle color profiles.
- [x] Change of speeds in interactive renderer, as they are not very nice
- [x] Add a ray/second counter somewhere -> SPPS instead, as ray counting is too costly on gpu side

## CUDA renderer
- [ ] Implement anisotropic metals / shading
- [ ] SDL3 migration
- [x] Refactor constants f3_ones, f3_zero and others
- [ ] New sphere / point light light types
- [ ] Textures loading (Venturi's style)
- [ ] SDF integration in CUDA
- [x] Time per pixel shading for performance display
- [x] Normals as lines for spheres
- [ ] Volumetric smoke ray-marching
- [ ] Normals color toggle for interactive renderer
- [ ] Depth map exportingLet'

## General code organization
- [x] There are still discrepancies for the cuda renderers
    - [x] clarify renderer_cuda_host.hpp vs renderer_cuda_device.cu responsibilities
    - [x] cuda_raytracer name is badly chosen

## General
- [ ] Better skybox
- [ ] Skybox as HDR, dynamic loading
- [x] Doxygen documentation

## Scenes
- [x] YAML scene should take camera positions
- [x] Dynamic scenes loading

## Bug fixing
- [ ] Artifacts when rendering metallic ground (grazing angle)
- [ ] Artifacts when rendering glass (might be related to metallic ground somehow ?)

## Optimisations
- [x] Fast maths

## Bugfixing 
- [ ] Adaptive sampling is not converging correctly in some cases, maybe can be fixed (not sure though)

## Others things
- [x] `ImGUI` GUI integration for controls -> reimplement existing controls + add more
- [x] Implement benchmarks for static renderer on typical scenes (multiple renders + average)
- [ ] Impact of different rendering optimizations reporting -> find a way and report (for instance, color complexity per pixel)
- [ ] `sdl` texture blitting integration + performance assessment
- [ ] Tests unitaires qui détectent les régressions dans le renderer GPU et CPU en comparant avec des scènes de référence, notamment lors d'optimisations
- [ ] Pipeline CI/CD pour runner les tests unitaires ci-dessus (si possible, besoin CUDA sur Github)
- [ ] Profiling and low-level optimizations + document changes using `nsight-systems` and `nsight-compute`
- [ ] implement dynamic screen rescaling 
- [ ] Implement NVIDIA OptiX image denoising
- [ ] Save image as OpenEXR format