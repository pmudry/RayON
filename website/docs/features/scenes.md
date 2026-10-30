# YAML Scene Format

Every scene can be described in a plain YAML file and loaded without recompilation. The format
is read by a small custom parser (`yaml_scene_loader.cc`), no external YAML library is needed.
The reference below matches `resources/scenes/SCENE_FORMAT.md` in the repository.

---

## Loading a scene

```bash
./rayon --scene ../resources/scenes/09_color_bleed_box.yaml -s 512 -r 1080
```

In interactive mode the **Scene** combo and the **Left / Right** keys switch between every YAML
file found in `resources/scenes/`. A scene loads in a few milliseconds; OBJ meshes and volume
grids take longer.

---

## File structure

Four optional top-level sections, in any order. Comments start with `#`, values may be quoted.

```yaml
camera:        # viewpoint
settings:      # render options
materials:     # named material library
geometry:      # the objects
```

Both `materials:` and `geometry:` entries accept the compact inline syntax:

```yaml
materials:
  - { name: floor, type: lambertian, albedo: [0.15, 0.15, 0.18] }
geometry:
  - { type: sphere, material: floor, center: [0, 1, 0], radius: 0.7 }
```

---

## `camera`

| Key | Default | Description |
|---|---|---|
| `position` | `[0, 0, 0]` | Camera origin |
| `look_at` | `[0, 0, -1]` | Point the camera looks at |
| `up` | `[0, 1, 0]` | World up vector |
| `fov` | `90` | Vertical field of view in degrees |

```yaml
camera:
  position: [0, 2, 8]
  look_at:  [0, 0.5, 0]
  fov: 45
```

!!! tip "Save the camera from the GUI"
    In interactive mode, frame the view you like and press **Save camera & lights to scene
    YAML** in the *Scene* section. The `camera:` lines and the light settings of the file are
    rewritten in place, comments included.

---

## `settings`

| Key | Default | Description |
|---|---|---|
| `background_color` | `[0.5, 0.7, 1.0]` | Top colour of the gradient sky (linear RGB) |
| `background_intensity` | `1.0` | Sky brightness (GUI *Ambient Light*) |
| `light_intensity` | `1.0` | Multiplier on every light's emission (GUI *Lights inten.*) |
| `light_falloff` | `2.0` | Distance exponent of the lights' reach, 2 = physical (GUI *Lights falloff*, see [Lights & Shadows](lighting.md)) |
| `indirect_clamp` | `50.0` | OptiX: per-sample ceiling on the light gathered after the first diffuse bounce, 0 = off (GUI *Indirect clamp*, see [Fireflies](lighting.md#fireflies)) |
| `use_bvh` | `false` | Build the BVH; recommended above ~50 objects and for any mesh |
| `adaptive_sampling` | `true` | Let converged pixels stop taking samples |
| `samples_per_batch` | `0` | Interactive batch size for this scene (`0` = renderer default of 8; `--samples-per-batch` wins) |

---

## `materials`

Each entry needs a `name` and a `type`. Geometry entries reference the name.

### Common fields

| Field | Default | Notes |
|---|---|---|
| `albedo` | `[0.7, 0.7, 0.7]` | Base colour |
| `emission` | `[0, 0, 0]` | Emitted radiance; alternatively `color` + `emission_intensity` |
| `roughness` | `0` | Used by the reflective types |
| `metallic` | `0` | Metallic factor |
| `texture` | — | Image file (relative to the scene file) that replaces the albedo through the mesh UVs |

### Material types

| `type` | What it is | Specific fields |
|---|---|---|
| `lambertian` | Ideal diffuse | `pattern` sub-block (below) |
| `mirror` | Perfect specular reflection | — |
| `rough_mirror`, `metal` | Fuzzy reflection | `roughness` |
| `glass`, `dielectric` | Refractive dielectric | `refractive_index` (1.5 for glass), `transmission` |
| `light` | Emitter | `emission`, or `color` + `emission_intensity` |
| `anisotropic_metal` | GGX microfacet conductor | `roughness`, `anisotropy`, `preset` (`gold`, `silver`, `copper`, `aluminum`) or `eta` + `k` |
| `thin_film` | Iridescent interference layer | `film_thickness` (nm), `film_ior` (1.33), `refractive_index` of the outside |
| `clear_coat` | Glossy dielectric coat over a diffuse base | `roughness` of the coat (0.05), `refractive_index` (1.5) |
| `volume`, `smoke`, `cloud` | Scattering medium inside a closed mesh or a density grid | see [Volumes](volumes.md) |
| `constant` | Flat unlit colour | — |
| `show_normals` | Normals as colours (debug) | — |

```yaml
materials:
  - name: "plaster"
    type: "lambertian"
    albedo: [0.8, 0.78, 0.72]

  - name: "brushed_gold"
    type: "anisotropic_metal"
    preset: "gold"
    roughness: 0.08
    anisotropy: 0.7

  - name: "bubble"
    type: "thin_film"
    film_thickness: 400.0
    film_ior: 1.33

  - name: "red_car_paint"
    type: "clear_coat"
    albedo: [0.85, 0.10, 0.08]
    roughness: 0.05
    refractive_index: 1.5

  - name: "warm_light"
    type: "light"
    color: [1.0, 0.875, 0.75]
    emission_intensity: 12
```

The physics behind each type is on the [Materials](../how-it-works/materials.md) page.

### Procedural patterns

A `lambertian` material can carry a `pattern` block: `fibonacci_dots` (dots on a Vogel spiral,
with `color`, `dot_count`, `dot_radius`) or `stripes` (alternating bands along the sphere's
local Y axis, with `color`).

```yaml
  - name: "pokeball"
    type: "lambertian"
    albedo: [0.9, 0.1, 0.1]
    pattern:
      type: "fibonacci_dots"
      color: [0.02, 0.02, 0.02]
      dot_count: 12
      dot_radius: 0.33
```

---

## `geometry`

Each entry needs a `type` and a `material`. Any entry accepts `visible: false`, which hides the
object from camera rays while it keeps lighting the scene (and keeps appearing in reflections).

### `sphere`

```yaml
  - type: "sphere"
    material: "glass"
    center: [0, 1, -2]
    radius: 0.8
```

### `rectangle`

A quad spanned by two edge vectors from a corner. Also the usual shape of an area light.

```yaml
  - type: "rectangle"
    material: "area_light"
    corner: [-1.5, 4, -1.5]
    u: [3, 0, 0]
    v: [0, 0, 3]
```

### `triangle`

`v0`, `v1`, `v2`, with optional per-vertex normals `n0`, `n1`, `n2` for smooth shading.

### `obj`

A Wavefront mesh, path relative to the scene file. `position` is applied after `scale`. The
`material` key is optional when the OBJ references an MTL file. See [OBJ Loading](obj-loading.md).

```yaml
  - type: "obj"
    material: "silver"
    file: "../models/stanford-bunny.obj"
    position: [0, 0, 0]
    scale: [1, 1, 1]
```

### `displaced_sphere`

A sphere with golf-ball dimples: `center`, `radius`, `displacement_scale` (0.2), `pattern_type` (0).

### `sdf`

A ray-marched signed distance primitive: `shape`, `center`, `rotation` (degrees, X then Y then
Z) and the shape's named parameters. See [SDF Shapes](sdf-shapes.md).

```yaml
  - type: "sdf"
    material: "steel"
    shape: "death_star"
    center: [-3.5, 1.2, -4.5]
    radius: 0.8
    cutout_radius: 0.5
    cutout_distance: 1.0
    rotation: [0, 54, 0]
```

---

## Complete minimal scene

```yaml
camera:
  position: [0, 2, 7]
  look_at: [0, 0, 0]
  fov: 45

settings:
  background_color: [0.05, 0.07, 0.12]
  background_intensity: 0.5
  use_bvh: true

materials:
  - name: "ground"
    type: "lambertian"
    albedo: [0.5, 0.5, 0.5]
  - name: "red_ball"
    type: "lambertian"
    albedo: [0.9, 0.1, 0.1]
  - name: "mirror_ball"
    type: "mirror"
  - name: "glass_ball"
    type: "glass"
    refractive_index: 1.5
  - name: "area_light"
    type: "light"
    emission: [10, 9, 8]

geometry:
  - type: "rectangle"
    material: "ground"
    corner: [-10, 0, -10]
    u: [20, 0, 0]
    v: [0, 0, 20]
  - type: "rectangle"
    material: "area_light"
    corner: [-1, 5, -1]
    u: [2, 0, 0]
    v: [0, 0, 2]
  - type: "sphere"
    material: "red_ball"
    center: [-1.5, 0.5, 0]
    radius: 0.5
  - type: "sphere"
    material: "mirror_ball"
    center: [0, 0.6, 0]
    radius: 0.6
  - type: "sphere"
    material: "glass_ball"
    center: [1.5, 0.5, 0]
    radius: 0.5
```

---

## Parser behaviour

- At most 100 materials and 1000 geometry entries per file.
- Keys are case-sensitive. An unknown material `type` falls back to `lambertian`; an unknown
  geometry type is skipped with a message.
- An OBJ that fails to load prints an error but does not abort the scene.
- With a duplicate material name, the last entry wins.

The bundled scenes are listed in [Getting Started](../getting-started.md#example-scenes).
