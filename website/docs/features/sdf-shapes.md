# SDF Shapes

In addition to analytical geometry (spheres, rectangles), RayON supports a set of procedural
shapes defined by **Signed Distance Functions** (SDFs). These are ray-marched in the GPU kernel.

---

## What is an SDF?

A signed distance function returns the signed distance from a point \(\mathbf{p}\) to the
nearest surface of a shape:

\[
d = f(\mathbf{p}) \begin{cases} < 0 & \text{inside the surface} \\ = 0 & \text{on the surface} \\ > 0 & \text{outside the surface} \end{cases}
\]

For a **sphere** of radius \(r\): \(f(\mathbf{p}) = |\mathbf{p}| - r\)

SDFs compose beautifully via smooth-minimum and union operations, enabling complex shapes
that would be impossible to describe analytically.

---

## Ray marching

SDF shapes are intersected by **sphere tracing** — an efficient ray marching algorithm:

```
t = 0
loop:
    p = ray.origin + t * ray.direction
    d = sdf(p)                    # distance to nearest surface
    if d < threshold: HIT at t
    if t > t_max:     MISS
    t += d                        # safe to advance by d — we cannot overshoot the surface
```

Each step advances the ray by exactly the SDF value — the maximum step that cannot overshoot
the surface. Smooth surfaces typically converge in 20–50 steps.

Normals are computed by the **finite difference gradient** of the SDF:

\[
\hat{n} \approx \text{normalize}\!\left(\begin{pmatrix} f(\mathbf{p}+\epsilon\hat{x}) - f(\mathbf{p}-\epsilon\hat{x}) \\ f(\mathbf{p}+\epsilon\hat{y}) - f(\mathbf{p}-\epsilon\hat{y}) \\ f(\mathbf{p}+\epsilon\hat{z}) - f(\mathbf{p}-\epsilon\hat{z}) \end{pmatrix}\right)
\]

---

## Available SDF primitives

### Torus

A donut shape defined by major radius \(R\) (ring radius) and minor radius \(r\) (tube radius):

\[
f_{\text{torus}}(\mathbf{p}) = \left(\sqrt{p_x^2 + p_z^2} - R\right)^2 + p_y^2 - r^2
\]

```yaml
- type: sdf_primitive
  sdf_type: torus
  material: chrome
  position: [0.0, 1.2, 0.0]
  scale: 0.8
  params: [0.5, 0.2]   # [major_radius, minor_radius]
```

### Octahedron

A regular octahedron: 8 triangular faces, 6 vertices:

```yaml
- type: sdf_primitive
  sdf_type: octahedron
  material: gold
  position: [-2.0, 0.8, 0.0]
  scale: 0.6
  rotation: [0.0, 30.0, 0.0]
```

### Death Star

A sphere with a hemispherical bite taken out of it — inspired by Inigo Quilez's SDF library:

```yaml
- type: sdf_primitive
  sdf_type: death_star
  material: rough_steel
  position: [2.0, 1.2, 0.0]
  scale: 0.7
  params: [0.35]   # [bite_radius_fraction]
```

### Pyramid

A 4-sided pyramid. The apex is at `scale` height above `position`:

```yaml
- type: sdf_primitive
  sdf_type: pyramid
  material: sandstone
  position: [0.0, 0.0, -2.0]
  scale: 1.2
  rotation: [0.0, 45.0, 0.0]
```

---

## Rotation support

All SDF primitives support Euler angle rotation. The rotation is applied as an inverse
transformation to the ray — equivalent to rotating the shape around its local origin:

```yaml
rotation: [rx, ry, rz]   # degrees, applied in X → Y → Z order
```

---

## Mixing SDFs with analytical geometry

SDF shapes and analytical geometry (spheres, rectangles) coexist in the same scene.
If BVH is enabled, SDF shapes are included in the hierarchy with a bounding sphere:

```yaml
settings:
  use_bvh: true

geometry:
  - type: sphere
    material: glass
    center: [0.0, 0.5, 0.0]
    radius: 0.5

  - type: sdf_primitive
    sdf_type: torus
    material: gold
    position: [2.0, 0.8, 0.0]
    scale: 0.6
```

<img class="render-img" src="../../assets/images/samples/golf.png"
     alt="Displaced sphere (golf ball) — procedural SDF displacement">

*The golf ball uses procedural displacement mapping — a variant of SDF evaluated on a
sphere surface to create the dimple pattern.*

---

## Performance

SDF ray marching is more expensive than analytical intersection (~20–80 marching steps per ray
hit vs. 1 step for a sphere). For a scene with a few SDF objects among many analytical shapes,
the BVH ensures SDF shapes are only marched when the ray's AABB is hit — keeping the overhead bounded.

| Geometry | Intersections per ray hit | GPU cost (relative) |
|---|---|---|
| Sphere | 1 | 1× |
| Rectangle | 1 | 1× |
| Torus (SDF) | 20–50 marching steps | ~20–30× |
| Octahedron (SDF) | 15–40 marching steps | ~15–25× |

For scenes dominated by SDF shapes, lowering `max_depth` or reducing resolution compensates.
