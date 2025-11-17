# Scene Files

This directory contains YAML scene definitions for the raytracer.

## Available Scenes

### 1. default_scene.yaml
The default scene featuring:
- 11 materials (rough mirrors, glass, Fibonacci dots, area light)
- 11 objects including:
  - Golden rough mirror sphere
  - Blue rough mirror displaced sphere (golf ball)
  - Red sphere with Fibonacci dot pattern
  - Glass sphere
  - 5 ISC logo colored spheres
  - Large ground sphere
  - Rectangular area light

**Recommended settings:** `-s 50 -r 720`

### 2. cornell_box.yaml
Classic Cornell box with:
- Red left wall, green right wall
- White floor, ceiling, and back wall
- Rectangular area light on ceiling
- Two box approximations using spheres
- 5 materials, 11 objects

**Recommended settings:** `-s 100 -r 720`

**Note:** This scene uses rectangular geometry extensively and an area light for realistic indirect lighting.

### 3. simple_scene.yaml
Minimal test scene with:
- 3 spheres: red lambertian, glass, and metallic
- Large ground sphere
- 4 materials, 4 objects

**Recommended settings:** `-s 20 -r 360`

## Usage

Load any scene with the `--scene` flag:

```bash
# Cornell box
./302_raytracer --scene ../resources/cornell_box.yaml -s 100 -r 720

# Default scene
./302_raytracer --scene ../resources/default_scene.yaml -s 50 -r 720

# Simple scene
./302_raytracer --scene ../resources/simple_scene.yaml -s 20 -r 360
```

## Creating Your Own Scenes

See the existing YAML files for examples. Basic structure:

```yaml
scene:
  name: "My Scene"
  description: "Scene description"

materials:
  - name: "material_name"
    type: "lambertian"  # or rough_mirror, glass, light, etc.
    albedo: [r, g, b]   # RGB values 0.0-1.0
    # Additional properties depending on type

geometry:
  - type: "sphere"
    material: "material_name"
    center: [x, y, z]
    radius: r
  
  - type: "rectangle"
    material: "material_name"
    corner: [x, y, z]
    u: [ux, uy, uz]  # First edge vector
    v: [vx, vy, vz]  # Second edge vector
```

## Material Types

- **lambertian**: Diffuse material (matte surfaces)
- **rough_mirror**: Metallic reflection with roughness
- **glass**: Refractive material with IOR
- **light**: Emissive material (area lights)
- **mirror**: Perfect reflection

## Geometry Types

- **sphere**: Basic sphere primitive
- **displaced_sphere**: Sphere with surface displacement (golf ball effect)
- **rectangle**: Quad/rectangle (useful for walls and area lights)

## Output

Rendered images are saved to `resources/output.png` relative to the execution directory.
