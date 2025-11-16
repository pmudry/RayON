# TODO List

## Interactive renderer
[ ] Gamma correction in interactive renderer is wrong when displayed, but saved correctly
[ ] Change of speeds in interactive renderer, as they are not very nice
[ ] Add a ray/second counter somewhere

## CUDA renderer
[ ] Implement anisotropic metals
[ ] Refactor constants f3_ones, f3_zero and others
[ ] New sphere light types
[ ] Textures loading (Venturi's style)
[ ] SDF integration
[ ] Time per pixel shading for performance display
[ ] Normals as lines for spheres
[ ] Ray-marching

## General code organization
[ ] There are still discrepancies for the cuda renderers
    [ ] why renderer_cuda.cu AND .hpp
    [ ] cuda_raytracer name is badly chosen

## General
[ ] Better skybox
[ ] Skybox as HDR, dynamic loading
[ ] Doxygen documentation

## Scenes
[ ] YAML scene should take camera positions
[ ] Dynamic scenes loading

## Bug fixing
[ ] Artifacts when rendering metallic ground (grazing angle)
[ ] Artifacts when rendering glass (might be related to metallic ground somehow ?)

## Optimisations
[ ] Fast maths

## Projet de semestre
[ ] ImGUI à ajouter
[ ] Profiling and low-level optimizations
[ ] Impact of different rendering optimizations reporting
[ ] Implement benchmarks for static renderer on typical scenes (multiple renders + average)
