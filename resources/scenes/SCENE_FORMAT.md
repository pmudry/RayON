# RayON — Scene File Format Reference

Scene files are written in a subset of YAML and loaded with the built-in lightweight parser
(no external YAML library required).

Run any scene with:

```bash
./rayon --scene resources/scenes/my_scene.yaml -s 128 -r 1080
```

---

## Top-level sections

A scene file contains four optional top-level sections, in any order:

```
camera:        # Viewpoint
settings:      # Render / acceleration options
materials:     # Named material library
geometry:      # Objects that make up the scene
```

Comments start with `#` and run to the end of the line.
Values may optionally be quoted with `"`.

---

## `camera`

All fields are optional; the defaults listed below apply when a field is absent.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `position` | vec3 | `[0, 0, 0]` | Camera origin in world space |
| `look_at` | vec3 | `[0, 0, -1]` | Point the camera looks toward |
| `up` | vec3 | `[0, 1, 0]` | World up vector |
| `fov` | float | `90` | Vertical field of view in degrees |

```yaml
camera:
  position: [0, 2, 8]
  look_at:  [0, 0.5, 0]
  fov: 45
```

---

## `settings`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `background_color` | vec3 | `[0.5, 0.7, 1.0]` | Sky/background colour (linear RGB) |
| `background_intensity` | float | `1.0` | Multiplier applied to the background colour |
| `ambient_light` | float | `0.1` | Constant ambient term added to all surfaces |
| `use_bvh` | bool | `false` | Enable BVH acceleration (recommended for > 50 objects) |
| `adaptive_sampling` | bool | `true` | Let converged pixels stop accumulating early |

```yaml
settings:
  background_color: [0.02, 0.02, 0.05]
  background_intensity: 0.3
  use_bvh: true
  adaptive_sampling: false
```

---

## `materials`

A YAML sequence of named material entries. Each entry requires at minimum `name` and `type`.

### Common fields (apply to every material type)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `name` | string | — | **Required.** Geometry entries reference this name |
| `type` | string | `"lambertian"` | See material types below |
| `albedo` | vec3 | `[0.7, 0.7, 0.7]` | Base/diffuse colour |
| `emission` | vec3 | `[0, 0, 0]` | Emitted radiance (makes the material a light source) |
| `roughness` | float | `0` | Surface roughness [0–1], used by reflective types |
| `metallic` | float | `0` | Metallic factor [0–1] |
| `visible` | — | — | Controlled per geometry item, not per material |

Two alternative ways to set emission:

```yaml
emission: [15, 12, 9]            # Direct linear RGB values
# — or —
color: [1.0, 0.875, 0.75]        # Normalised colour…
emission_intensity: 8            # …multiplied by this scalar
```

---

### Material types

#### `lambertian` — ideal diffuse surface

```yaml
- name: "plaster"
  type: "lambertian"
  albedo: [0.8, 0.78, 0.72]
```

#### `mirror` — perfect specular reflection

```yaml
- name: "chrome"
  type: "mirror"
  albedo: [0.95, 0.95, 0.95]
```

#### `rough_mirror` — microfacet reflection with tunable roughness

Additional fields: `roughness` [0–1].

```yaml
- name: "brushed_gold"
  type: "rough_mirror"
  albedo: [1.0, 0.85, 0.47]
  roughness: 0.08
  metallic: 1.0
```

#### `metal` — alias for `rough_mirror`

#### `glass` / `dielectric` — refractive glass

Additional fields: `refractive_index` (e.g. 1.5 for borosilicate glass), `transmission`.

```yaml
- name: "glass"
  type: "glass"
  albedo: [1.0, 1.0, 1.0]
  refractive_index: 1.52
  transmission: 1.0
```

#### `light` — emissive surface (area light)

Use `emission` or the `color` + `emission_intensity` shorthand. Assign to a rectangle for an area light.

```yaml
- name: "warm_light"
  type: "light"
  color: [1.0, 0.875, 0.75]
  emission_intensity: 12

# — or equivalently —
- name: "warm_light"
  type: "light"
  emission: [12, 10.5, 9]
```

#### `constant` — flat unlit colour (useful for backgrounds / debugging)

```yaml
- name: "debug"
  type: "constant"
  albedo: [1.0, 0.0, 0.5]
```

#### `show_normals` — visualises surface normals as colour

```yaml
- name: "normals"
  type: "show_normals"
```

#### `anisotropic_metal` — physically-based GGX microfacet conductor

Additional fields: `roughness`, `anisotropy` [0–1], and either `preset` **or** explicit `eta`/`k`.

`preset` values: `"gold"`, `"silver"`, `"copper"`, `"aluminum"`.

```yaml
# Named preset
- name: "brushed_aluminum"
  type: "anisotropic_metal"
  roughness: 0.3
  anisotropy: 0.8
  preset: "aluminum"

# Manual complex IOR (measured data, RGB)
- name: "exotic_metal"
  type: "anisotropic_metal"
  roughness: 0.15
  anisotropy: 0.5
  eta: [0.18, 0.42, 1.37]
  k:   [3.42, 2.35, 1.77]
```

#### `thin_film` — iridescent soap-bubble / oil-slick interference

Additional fields: `film_thickness` (nm, e.g. 250–650), `film_ior` (default 1.33), `refractive_index` (exterior medium, default 1.0).

The dominant reflected colour depends on thickness:

| Thickness (nm) | Dominant hue |
|---|---|
| ~250 | violet / UV border |
| ~300 | blue-violet |
| ~400 | blue-green |
| ~500 | green-yellow |
| ~600 | orange-red |

```yaml
- name: "bubble_blue"
  type: "thin_film"
  film_thickness: 400.0
  film_ior: 1.33
  refractive_index: 1.0
```

#### `clear_coat` — lacquered/car-paint: glossy dielectric coat over diffuse base

Additional fields: `roughness` (coat surface, default 0.05), `refractive_index` (coat IOR, default 1.5).

The coat is achromatic; `albedo` controls the diffuse base colour visible through it.

```yaml
- name: "red_car_paint"
  type: "clear_coat"
  albedo: [0.85, 0.10, 0.08]
  roughness: 0.05
  refractive_index: 1.5

- name: "mirror_coat"
  type: "clear_coat"
  albedo: [0.9, 0.9, 0.9]
  roughness: 0.0

- name: "satin_lacquer"
  type: "clear_coat"
  albedo: [0.2, 0.5, 0.8]
  roughness: 0.2
```

---

### Procedural patterns

Any `lambertian` material can carry an optional `pattern` sub-block.

#### `fibonacci_dots`

Regularly-spaced dots distributed with the Fibonacci / Vogel spiral on the sphere.

```yaml
- name: "pokeball_pattern"
  type: "lambertian"
  albedo: [0.9, 0.1, 0.1]
  pattern:
    type: "fibonacci_dots"
    color: [0.02, 0.02, 0.02]   # Dot colour
    dot_count: 12               # Number of dots (integer)
    dot_radius: 0.33            # Angular radius as fraction of hemisphere
```

#### `checkerboard` / `stripes`

```yaml
pattern:
  type: "checkerboard"
  color: [0.05, 0.05, 0.05]
```

---

## `geometry`

A YAML sequence of geometry entries. Each entry requires `type` and `material`.

### `visible` flag

Any geometry entry accepts an optional `visible` flag (default `true`). Setting it to `false` makes the object invisible to primary (camera) rays but still participates in lighting — useful for area lights you want to cast light without appearing in the image.

```yaml
- type: "rectangle"
  material: "hidden_light"
  visible: false
  corner: [...]
  ...
```

---

### Geometry types

#### `sphere`

| Field | Type | Description |
|-------|------|-------------|
| `center` | vec3 | Centre of the sphere |
| `radius` | float | Sphere radius |

```yaml
- type: "sphere"
  material: "glass"
  center: [0, 1, -2]
  radius: 0.8
```

#### `displaced_sphere` — sphere with procedural surface displacement (golf-ball dimples)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `center` | vec3 | — | Centre |
| `radius` | float | `1.0` | Base radius |
| `displacement_scale` | float | `0.2` | Dimple depth |
| `pattern_type` | int | `0` | Pattern variant (`0` = golf-ball dimples) |

```yaml
- type: "displaced_sphere"
  material: "white_ball"
  center: [1.2, 0.0, -2.0]
  radius: 0.5
  displacement_scale: 0.15
  pattern_type: 0
```

#### `rectangle` — flat quadrilateral (also used for area lights)

Defined by `corner` + two edge vectors `u` and `v`. The surface spans `corner + s*u + t*v` for `s,t ∈ [0,1]`.

```yaml
# Overhead area light
- type: "rectangle"
  material: "area_light"
  corner: [-1.5, 4, -1.5]
  u: [3, 0, 0]
  v: [0, 0, 3]

# Ground plane
- type: "rectangle"
  material: "ground"
  corner: [-10, 0, -10]
  u: [20, 0, 0]
  v: [0, 0, 20]
```

#### `triangle` — single triangle

```yaml
- type: "triangle"
  material: "blue_diff"
  v0: [0, 0, 0]
  v1: [1, 0, 0]
  v2: [0.5, 1, 0]

# With per-vertex normals (smooth shading)
- type: "triangle"
  material: "smooth_mat"
  v0: [0, 0, 0]
  v1: [1, 0, 0]
  v2: [0.5, 1, 0]
  n0: [0, 0, 1]
  n1: [0, 0, 1]
  n2: [0, 0, 1]
```

#### `obj` — Wavefront OBJ mesh file

Paths are resolved relative to the scene file's directory.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | string | — | Relative or absolute path to the `.obj` file |
| `position` | vec3 | `[0, 0, 0]` | Translation applied after loading |
| `scale` | vec3 | `[1, 1, 1]` | Per-axis scale factor |

```yaml
- type: "obj"
  material: "silver"
  file: "../models/stanford-bunny.obj"
  position: [0, 0, 0]
  scale: [1, 1, 1]

# Scale down a large mesh (e.g. PokéBall ~80 units → ~1.8 unit diameter)
- type: "obj"
  material: "red_car_paint"
  file: "../models/PokemonBall.obj"
  position: [0, 0, 0]
  scale: [0.0225, 0.0225, 0.0225]
```

Available models in `resources/models/`: `bunny-2.obj`, `bunny-3.obj`, `nefertiti.obj`,
`nefertiti2.obj`, `stanford-bunny.obj`, `suzanne.obj`, `teapot.obj`, `PokemonBall.obj`,
`victory.obj`, `xyzrgb_dragon.obj`, `cube.obj`, `cylinder.obj`,
`tetrahedron/octahedron/dodecahedron/icosahedron.obj`, `isc-text.obj`.

---

## Flow-style (inline) items

Both `materials:` and `geometry:` entries accept compact inline syntax with `{ }`.
This is convenient for simple objects:

```yaml
materials:
  - { name: floor,  type: lambertian, albedo: [0.15, 0.15, 0.18] }
  - { name: light,  type: light,      emission: [18, 18, 18] }
  - { name: bubble, type: thin_film,  film_thickness: 420.0, film_ior: 1.33, refractive_index: 1.0 }

geometry:
  - { type: sphere, material: bubble, center: [-1.0, 1.0, 0.0], radius: 0.7 }
  - { type: rectangle, material: light, corner: [-2, 4, -1], u: [4, 0, 0], v: [0, 0, 4] }
```

---

## Complete minimal example

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
    albedo: [0.95, 0.95, 0.95]

  - name: "glass_ball"
    type: "glass"
    refractive_index: 1.5
    transmission: 1.0

  - name: "area_light"
    type: "light"
    color: [1.0, 0.9, 0.8]
    emission_intensity: 10

geometry:
  - type: "rectangle"
    material: "ground"
    corner: [-10, -0.001, -10]
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

## Parser behaviour and constraints

- Maximum **100 materials** and **1000 geometry** entries per file.
- Keys are case-sensitive.
- Unrecognised material `type` strings fall back to `"lambertian"`.
- Unknown or future geometry types are skipped with an error message to stderr.
- OBJ paths that fail to load print an error but do not abort the scene load.
- Duplicate material names: the second entry overwrites the first in the name→ID map.
- `use_bvh: true` is strongly recommended for scenes with triangle meshes or more than ~50 objects; it can provide 5–50× speedup.
