# TODO List

## Interactive renderer
- [ ] Gamma correction in interactive renderer is wrong when displayed, but saved correctly -> it's related to how program handle color profiles.
- [ ] Change of speeds in interactive renderer, as they are not very nice
- [ ] Add a ray/second counter somewhere

## CUDA renderer
- [ ] Implement anisotropic metals
- [ ] Refactor constants f3_ones, f3_zero and others
- [ ] New sphere light types
- [ ] Textures loading (Venturi's style)
- [ ] SDF integration
- [ ] Time per pixel shading for performance display
- [ ] Normals as lines for spheres
- [ ] Ray-marching

## General code organization
- [x] There are still discrepancies for the cuda renderers
    - [x] clarify renderer_cuda_host.hpp vs renderer_cuda_device.cu responsibilities
    - [x] cuda_raytracer name is badly chosen

## General
- [ ] Better skybox
- [ ] Skybox as HDR, dynamic loading
- [x] Doxygen documentation

## Scenes
- [ ] YAML scene should take camera positions
- [ ] Dynamic scenes loading

## Bug fixing
- [ ] Artifacts when rendering metallic ground (grazing angle)
- [ ] Artifacts when rendering glass (might be related to metallic ground somehow ?)

## Optimisations
- [ ] Fast maths

## Semester project
- [ ] `ImGUI` GUI integration for controls
- [ ] Impact of different rendering optimizations reporting
- [ ] Implement benchmarks for static renderer on typical scenes (multiple renders + average)
- [ ] Tests unitaires qui détectent les régressions dans le renderer GPU et CPU en comparant avec des scènes de référence, notamment lors d'optimisations
- [ ] Profiling and low-level optimizations + document changes
- [ ] Implement NVIDIA OptiX image denoising
- [ ] Pipeline CI/CD pour runner les tests unitaires ci-dessus (si possible, besoin CUDA sur Github)

- [ ] Save as OpenEXR format