# Struct-Based Material System: Current Status

This note was updated to reflect the code currently present in the repository.

## Status Summary

The CRTP-based struct material framework exists in the codebase, but it is **not** currently the active CUDA shading path.

- The framework files are present under `src/rayon/gpu_renderers/materials/`.
- The active render path in `src/rayon/gpu_renderers/cuda_raytracer.cuh` uses a flattened `switch`-based material scatter implementation (`scatter_material(...)` + `ray_color(...)`).
- There is no `USE_NEW_MATERIAL_SYSTEM` toggle in the current tree.
- `shader_common.cuh` and `src/302_raytracer/...` paths referenced by the previous version of this document do not exist in this repository layout.

## What Exists Today

### Material framework files

Current directory:

```text
src/rayon/gpu_renderers/materials/
├── material_base.cuh
├── material_dispatcher.cuh
└── legacy/
        ├── lambertian.cuh
        ├── mirror.cuh
        ├── rough_mirror.cuh
        ├── glass.cuh
        ├── light.cuh
        ├── constant.cuh
        ├── show_normals.cuh
        ├── thin_film.cuh
        └── clear_coat.cuh
```

### Framework capabilities (implemented)

- `material_base.cuh` defines CRTP `MaterialBase<Derived>` and shared optical helpers.
- `material_dispatcher.cuh` defines `MaterialDescriptor`, `MaterialParamsUnion`, and template dispatch helpers:
    - `dispatch_material(...)`
    - `dispatch_material_bool(...)`
    - `dispatch_material_f3(...)`
- Per-material structs expose `scatter(...)` and `emission()` methods.

## What Is Not True Anymore

The previous version of this file contained claims that are currently inaccurate:

- It stated the system was integrated through `shader_common.cuh` with `#define USE_NEW_MATERIAL_SYSTEM`.
    - Current state: no such file/toggle in the active tree.
- It implied easy runtime switching between legacy and CRTP paths.
    - Current state: active CUDA path is hardwired in `cuda_raytracer.cuh`.
- It claimed zero-overhead parity as a guaranteed, validated result.
    - Current state: this may be plausible, but this repository snapshot does not include benchmark or validation evidence in this file.

## Current Active CUDA Material Path

In the active renderer path:

- `LegacyMaterialType` is defined in `src/rayon/gpu_renderers/cuda_raytracer.cuh`.
- Material behavior is selected in `scatter_material(...)` via `switch (rec.material)`.
- `ray_color(...)` calls that `scatter_material(...)` function directly.

This means the CRTP dispatcher framework is currently best described as **available scaffolding / alternative implementation**, not the production default path.

## If You Want To Re-evaluate the CRTP Path

Suggested process:

1. Wire `MaterialDescriptor` + `dispatch_material(...)` into the active CUDA shading path.
2. Build and render a fixed set of scenes with both implementations.
3. Compare image outputs (numerical or pixel diff with tolerance).
4. Compare performance (render time, occupancy, register pressure, rays/sec).
5. Keep this document updated with measured results instead of assumptions.

## Conclusion

The struct/CRTP material system is implemented at the file level, but the repository currently renders with the direct `switch` path in `cuda_raytracer.cuh`. Treat this system as present but not fully integrated into the active pipeline.

---

Updated: March 20, 2026
