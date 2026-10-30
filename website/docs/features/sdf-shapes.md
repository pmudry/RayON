# SDF Shapes

Besides analytic spheres, rectangles and triangles, RayON renders ten shapes defined by
**signed distance functions** (SDFs). They are intersected by sphere tracing in the GPU kernel,
on both back-ends, from one shared source file (`gpu_renderers/sdf_primitives.cuh`). The
distance functions are Íñigo Quilez's.

<img class="render-img" src="../../assets/images/sdf_gallery.png"
     alt="Ten ray-marched SDF primitives in a grey studio room on a lacquered floor">

*`18_sdf_shapes.yaml`: gold mandelbulb, ceramic death star, brass cut hollow sphere, torus,
glass octahedron, pyramid, capsule, box, cylinder and sphere, lit by a hidden soft box and two
bulbs. OptiX, 384 spp, 720p.*

---

## What is an SDF?

A signed distance function returns the distance from a point \(\mathbf{p}\) to the nearest
surface of a shape, negative inside:

\[
f(\mathbf{p}) \begin{cases} < 0 & \text{inside} \\ = 0 & \text{on the surface} \\ > 0 & \text{outside} \end{cases}
\]

For a sphere of radius \(r\): \(f(\mathbf{p}) = |\mathbf{p}| - r\). For a box of half extents
\(\mathbf{b}\), with \(\mathbf{q} = |\mathbf{p}| - \mathbf{b}\):
\(f = |\max(\mathbf{q}, 0)| + \min(\max(q_x, q_y, q_z), 0)\). For a torus of radii \(R, r\):
\(f = \left|\left(\sqrt{p_x^2 + p_z^2} - R,\; p_y\right)\right| - r\).

Shapes compose with unions, intersections and smooth minima, which is how the death star (a
sphere minus a sphere) and the cut hollow sphere are built.

---

## Ray marching

An SDF is intersected by **sphere tracing**: from the ray origin, step forward by the distance
the function returns, which is the largest step that cannot overshoot the surface, until the
distance falls under a threshold.

```
t = t_enter                     # the ray is first clipped to the shape's bounding sphere
repeat up to 320 times:
    d = sdf(origin + t * dir)
    if |d| < eps(t):  hit        # eps = 1e-4 * max(1, 0.1 t): looser far from the camera
    t += d                       # or at least 2 eps once inside the band, to keep moving
    if t > t_exit:   miss
```

Two details matter for a path tracer, where rays also start *on* the shapes:

- A hit only counts after the march has been at least 1.5 ε away from the surface once
  ("arming"), so a refracted ray leaving a glass octahedron is not caught by the surface it
  starts on.
- The normal is the gradient of the SDF by central differences with a step of 2·10⁻⁴.

On OptiX each SDF is a **custom primitive**: its axis-aligned bounding box goes into the
acceleration structure and `__intersection__sdf` runs the marcher when a ray enters the box.
On CUDA the same marcher is called from `intersect_geometry()` through the BVH, which wraps the
shape in its bounding sphere.

---

## YAML syntax

```yaml
  - type: "sdf"
    material: "steel"
    shape: "death_star"
    center: [-3.5, 1.2, -4.5]
    radius: 0.8
    cutout_radius: 0.5
    cutout_distance: 1.0
    rotation: [0, 54, 0]          # degrees, applied in X, Y, Z order
```

| `shape` | Named parameters | Notes |
|---|---|---|
| `sphere` | `radius` | |
| `box` | `half_extents` (vec3) | |
| `torus` | `major_radius`, `minor_radius` | Ring in the XZ plane |
| `capsule` | `radius`, `height` | Along Y |
| `cylinder` | `radius`, `height` | Along Y |
| `mandelbulb` | `power` (8), `iterations` | The fractal; expensive, keep it small |
| `death_star` | `radius`, `cutout_radius`, `cutout_distance` | Crater facing +X before rotation |
| `cut_hollow_sphere` | `radius`, `cut_height`, `thickness` | A bowl |
| `octahedron` | `size` | |
| `pyramid` | `height` | Square base of side 1 at y = 0 |

`params: [a, b, c]` sets the same three values positionally when you prefer. A `plane` shape
also exists for infinite ground planes. Rotations are Euler angles in degrees; the inverse
rotation is applied to the ray, which is equivalent to rotating the shape about its centre.

---

## Materials on SDFs

Any material works: the gallery scene puts glass on the octahedron, a clear-coat lacquer on the
death star and anisotropic gold on the mandelbulb. The `sdf_material` type exists for the
built-in demos but is not required.

---

## Cost

A sphere intersection is one formula; a march is 20 to 80 evaluations of the distance
function, and the mandelbulb evaluates a fractal at each of them. The bounding box or sphere
keeps the marcher from running on rays that miss the shape, so a few SDFs among analytic
geometry cost little; a scene made of mandelbulbs is slow. The gallery scene uses
`settings.samples_per_batch: 2` to stay responsive in interactive mode.
