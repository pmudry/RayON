# YAML Scene Format

All scenes in RayON can be described as plain YAML files and loaded without recompilation.
The format is parsed by a lightweight custom parser (`yaml_scene_loader.cc`) — no external
YAML library is required.

---

## Loading a scene

```bash
./rayon --scene path/to/scene.yaml
```

If the file cannot be found or parsed, the renderer falls back to the built-in default scene
with a warning message. Load time is under 10 ms for any scene.

---

## File structure

A scene file has up to five top-level sections:

```yaml
settings:     # (optional) render parameters and flags
camera:       # (optional) camera position and lens settings
materials:    # list of named materials
geometry:     # list of geometric objects
```

---

## Settings

```yaml
settings:
  use_bvh: true         # enable BVH acceleration (recommended for > 15 objects)
  background_color: [0.05, 0.05, 0.08]   # sky/background RGB (values 0–1)
```

---

## Camera

```yaml
camera:
  look_from: [0.0, 1.5, 6.0]   # camera position
  look_at:   [0.0, 0.5, 0.0]   # point the camera looks at
  up:        [0.0, 1.0, 0.0]   # world up direction
  vfov:      40.0               # vertical field of view (degrees)
  aperture:  0.02               # lens aperture radius (0 = pinhole)
  focus_dist: 6.0               # focal plane distance
```

---

## Materials

Materials are defined in a named list and referenced by name in geometry entries.

### Lambertian (diffuse)

```yaml
materials:
  - name: red_wall
    type: lambertian
    albedo: [0.65, 0.05, 0.05]
```

### Mirror

```yaml
  - name: perfect_mirror
    type: mirror
```

### Rough mirror (metal)

```yaml
  - name: gold
    type: rough_mirror
    albedo: [1.0, 0.85, 0.57]    # tint colour
    roughness: 0.03               # 0.0 = perfect, 1.0 ≈ diffuse
```

### Glass (dielectric)

```yaml
  - name: glass
    type: glass
    ior: 1.5                      # index of refraction
```

### Area light

```yaml
  - name: warm_light
    type: light
    emission: [5.0, 4.5, 3.5]    # RGB emission; values > 1 are valid (HDR)
```

### Anisotropic metal (GGX microfacet)

```yaml
  - name: brushed_gold
    type: anisotropic_metal
    preset: gold            # gold | silver | copper | aluminum
    roughness: 0.08
    anisotropy: 0.7         # 0.0 = isotropic, 1.0 = maximum anisotropy

  - name: custom_conductor
    type: anisotropic_metal
    roughness: 0.05
    anisotropy: 0.5
    eta: [0.18, 0.42, 1.37]   # complex IOR real part (RGB)
    k:   [3.42, 2.35, 1.77]   # complex IOR imaginary part (RGB)
```

### Thin film (iridescent)

```yaml
  - name: soap_bubble
    type: thin_film
    film_thickness: 400    # nanometers — controls highlight colour
    film_ior: 1.33         # refractive index of the film (water ≈ 1.33)
```

### Clear coat

```yaml
  - name: car_paint_red
    type: clear_coat
    albedo: [0.8, 0.05, 0.05]  # diffuse base colour
    roughness: 0.04             # coat roughness (0 = mirror coat)
    refractive_index: 1.5       # coat IOR (typical lacquer/plastic)
```

### Procedural patterns

Procedural patterns can be added to any Lambertian or rough-mirror material via a `pattern`
block. The pattern modulates the base albedo.

```yaml
  - name: checkerboard_floor
    type: lambertian
    albedo: [0.9, 0.9, 0.9]
    pattern:
      type: checkerboard
      color: [0.1, 0.1, 0.1]

  - name: fibonacci_dots
    type: lambertian
    albedo: [0.8, 0.8, 0.8]
    pattern:
      type: fibonacci_dots
      color: [0.1, 0.1, 0.1]
      dot_count: 12
      dot_radius: 0.33

  - name: striped
    type: lambertian
    albedo: [0.9, 0.9, 0.9]
    pattern:
      type: stripes
      color: [0.1, 0.1, 0.1]
```

The texture path is resolved relative to the YAML file's directory. Textures are loaded once
and deduplicated by path.
See [OBJ Loading](obj-loading.md) for applying textures via MTL files.

---

## Geometry

### Sphere

```yaml
geometry:
  - type: sphere
    material: gold
    center: [-2.0, 0.5, 0.0]
    radius: 0.8
```

### Rectangle (area quad)

Defined by a position and two edge vectors:

```yaml
  - type: rectangle
    material: warm_light
    position: [-1.0, 3.0, -2.0]   # one corner of the rectangle
    u_vec:    [ 2.5, 0.0,  0.0]   # first edge vector
    v_vec:    [ 0.0, 0.0,  1.5]   # second edge vector
```

The surface normal is `cross(u_vec, v_vec)`. Flip `u_vec` or `v_vec` to reverse it.

### Displaced sphere (golf ball)

```yaml
  - type: displaced_sphere
    material: golf_white
    center: [1.0, 0.3, 0.0]
    radius: 0.5
    displacement: 0.02    # dimple depth
    frequency: 18.0       # number of dimple rows
```

### OBJ mesh

```yaml
  - type: obj
    material: plastic          # optional — overridden by MTL per-group materials
    file: "../resources/models/bunny.obj"
    scale:    [1.0, 1.0, 1.0]
    position: [0.0, 0.0, 0.0]
```

The `material` key is optional when the OBJ has a `mtllib` — each `usemtl` group in the OBJ
will use the material defined in the `.mtl` file.

The path is relative to the scene YAML file's directory.
See [OBJ Loading](obj-loading.md) for MTL materials, UV textures, and mesh format requirements.

### SDF primitive

```yaml
  - type: sdf_primitive
    material: chrome
    sdf_type: torus          # torus | octahedron | death_star | pyramid
    position: [0.0, 1.0, 0.0]
    scale: 0.6
    rotation: [0.0, 45.0, 0.0]   # Euler angles in degrees (X, Y, Z)
    params: [0.5, 0.2]            # shape-specific: torus = [major_r, minor_r]
```

---

## Complete minimal scene

```yaml
settings:
  use_bvh: false
  background_color: [0.5, 0.7, 1.0]

camera:
  look_from: [0.0, 1.0, 4.0]
  look_at:   [0.0, 0.5, 0.0]
  vfov: 50.0

materials:
  - name: sky_blue
    type: lambertian
    albedo: [0.3, 0.5, 0.8]

  - name: diffuse_red
    type: lambertian
    albedo: [0.8, 0.1, 0.1]

  - name: floor
    type: lambertian
    albedo: [0.8, 0.8, 0.8]

geometry:
  - type: sphere
    material: diffuse_red
    center: [0.0, 0.5, 0.0]
    radius: 0.5

  - type: sphere
    material: floor
    center: [0.0, -100.0, 0.0]   # Earth trick — large sphere as ground plane
    radius: 100.0
```

---

## Available example scenes

| File | What it demonstrates |
|---|---|
| `default_scene.yaml` | Full-featured: metals, glass, area light, SDF shapes |
| `09_color_bleed_box.yaml` | Cornell box with colour bleeding |
| `05_material_laboratory.yaml` | Side-by-side all material types |
| `03_platonic_solids.yaml` | SDF shapes: torus, octahedron, pyramid |
| `04_obj_dragon.yaml` | Stanford dragon OBJ mesh |
| `04_obj_statue.yaml` | Statue OBJ mesh |
| `01_anisotropic_metals_test.yaml` | Anisotropic brushed-metal highlights |
| `02_displacement_garden.yaml` | Displacement-mapped spheres |
| `06_caustics_chapel.yaml` | Refractive caustics from glass |
| `10_isc_neons_bunny.yaml` | Emissive neon lights + bunny mesh |
| `11_soap_bubbles.yaml` | Thin-film iridescence |
| `12_clearcoat_pokemonball.yaml` | Clearcoat over diffuse base |
| `13_texture_test1.yaml` | UV-mapped OBJ with grid texture (MTL-driven) |
| `14_texture_test2.yaml` | Multi-material OBJ scene with Blender-exported MTL |
| `15_mis_furnace_test.yaml` | MIS furnace test — energy conservation validation |
| `bvh_stress_courtyard.yaml` | 300+ objects, BVH stress test |
| `pattern_gallery.yaml` | Fibonacci dots, stripes patterns |
