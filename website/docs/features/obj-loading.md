# OBJ Loading

RayON can import triangle meshes from Wavefront `.obj` files and render them alongside
analytical geometry using physical path tracing — same materials, same BVH, same illumination.
Materials can come from a YAML definition or directly from the `.mtl` file referenced by the
OBJ, and UV-mapped image textures are fully supported on the GPU (CUDA and OptiX backends).

---

## Supported format

| Feature | Supported |
|---|---|
| Vertex positions (`v`) | ✓ |
| Vertex normals (`vn`) | ✓ — smooth per-vertex normals |
| Texture coordinates (`vt`) | ✓ — UV-mapped image textures (GPU backends) |
| Triangular faces (`f`) | ✓ |
| Quad faces | Auto-triangulated into two triangles |
| Multiple material groups (`usemtl`) | ✓ — per-group materials from MTL |
| MTL material files (`mtllib`) | ✓ — Kd, Ke, Ni, Ns, map_Kd supported |

!!! note "Triangles only"
    Faces with more than 4 vertices are not supported. Most 3D modelling tools can export
    triangulated meshes via an export option.

!!! note "GPU only for textures"
    UV texture sampling is implemented in the CUDA and OptiX backends. CPU renders ignore
    the texture and fall back to the material's solid albedo.

---

## Loading a mesh

In a YAML scene file:

```yaml
geometry:
  - type: obj
    material: plastic_white    # optional — YAML material overrides MTL
    file: "../resources/models/bunny.obj"
    scale: [1.0, 1.0, 1.0]    # per-axis scale
    position: [0.0, 0.0, 0.0] # translation after scaling
```

The `file` path is resolved **relative to the YAML scene file's directory**.

### MTL-only mode

When the OBJ file contains a `mtllib` reference, the `material` key in YAML is optional.
Omitting it lets all per-face materials come from the `.mtl` file:

```yaml
geometry:
  - type: obj
    file: "../models/my_scene.obj"
    position: [0.0, 0.0, 0.0]
    scale:    [1.0, 1.0, 1.0]
    # No material: key — each usemtl group uses its MTL-defined material
```

If a YAML `material` is provided it is used as fallback for any face group that has no
`usemtl` directive in the OBJ.

Loading multiple meshes:

```yaml
geometry:
  - type: obj
    material: gold
    file: "../resources/models/dragon.obj"
    scale: [0.5, 0.5, 0.5]
    position: [-1.5, 0.0, 0.0]

  - type: obj
    material: glass
    file: "../resources/models/bunny.obj"
    scale: [1.0, 1.0, 1.0]
    position: [1.5, 0.0, 0.0]
```

---

## MTL material support

When the OBJ references an `.mtl` file, RayON parses it and maps standard MTL properties to
internal materials:

| MTL property | Effect in RayON |
|---|---|
| `Kd r g b` | Diffuse albedo colour |
| `Ke r g b` | Emission — creates a `light` material when non-zero |
| `Ns exponent` | Converted to roughness: `roughness ≈ 1 − √(Ns/1000)` |
| `Ni ior` | Refractive index (used for glass materials) |
| `d opacity` | Opacity < 0.5 selects a glass material |
| `illum 5/7` | Selects mirror/glass material |
| `map_Kd path` | Loads an image texture (see below) |

Texture paths in `.mtl` files are resolved relative to **the `.mtl` file's directory**.

---

## Image texture mapping

When a material's `map_Kd` specifies a texture image, RayON loads it with `stb_image`
and uploads it as a CUDA texture object. UV coordinates from the OBJ are interpolated
across the triangle using barycentric weights:

$$
(u, v)_\text{hit} = (1-u-v)\,(u_0,v_0) + u\,(u_1,v_1) + v\,(u_2,v_2)
$$

The sampled texel then overrides the material's solid albedo at the hit point,
so you can use any Lambertian, rough-mirror, or glass material with a texture.

Textures can also be assigned directly in the YAML material block:

```yaml
materials:
  - name: textured_floor
    type: lambertian
    albedo: [1.0, 1.0, 1.0]    # base tint (overridden by texture)
    texture: "../resources/textures/grid_512.png"
```

<div class="img-grid cols-2">
  <figure>
    <img src="../../assets/images/samples/textures/uv_mapping.png" alt="UV texture mapped cube and sphere">
    <figcaption><strong>UV texture mapping</strong> — a grid texture applied to a UV-mapped cube, sphere, and ground plane via MTL.</figcaption>
  </figure>
  <figure>
    <img src="../../assets/images/samples/textures/texture_orig.png" alt="Textured OBJ scene">
    <figcaption><strong>MTL-driven textures</strong> — per-group materials from an exported Blender scene with a UV grid texture.</figcaption>
  </figure>
</div>

---

## Triangle intersection — Möller–Trumbore

Triangles are intersected using the **Möller–Trumbore algorithm** — numerically stable and fast:

For a ray \(\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}\) and triangle \((\mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2)\):

\[
\begin{pmatrix} t \\ u \\ v \end{pmatrix} =
\frac{1}{\mathbf{p} \cdot \mathbf{e}_1}
\begin{pmatrix}
  \mathbf{q} \cdot \mathbf{e}_2 \\
  \mathbf{p} \cdot \mathbf{t}_\text{vec} \\
  \mathbf{q} \cdot \mathbf{d}
\end{pmatrix}
\]

where \(\mathbf{e}_1 = \mathbf{v}_1 - \mathbf{v}_0\), \(\mathbf{e}_2 = \mathbf{v}_2 - \mathbf{v}_0\),
\(\mathbf{p} = \mathbf{d} \times \mathbf{e}_2\), \(\mathbf{t}_\text{vec} = \mathbf{o} - \mathbf{v}_0\),
\(\mathbf{q} = \mathbf{t}_\text{vec} \times \mathbf{e}_1\).

A hit occurs when \(u \geq 0\), \(v \geq 0\), \(u + v \leq 1\), and \(t > t_\min\).

---

## Smooth normals

When the `.obj` file contains vertex normals (`vn` records), RayON interpolates them across
the triangle face using the barycentric coordinates \((u, v)\):

\[
\hat{n}_\text{interp} = (1 - u - v)\,\hat{n}_0 + u\,\hat{n}_1 + v\,\hat{n}_2
\]

This makes the mesh appear smooth even at low polygon counts:

<div class="img-grid cols-2">
  <figure>
    <img src="../../assets/images/samples/obj_loading.png" alt="Smooth OBJ mesh import">
    <figcaption><strong>Smooth normals</strong> — interpolated per-vertex normals make the low-poly mesh look smooth.</figcaption>
  </figure>
  <figure>
    <img src="../../assets/images/samples/gui_debug/normals.png" alt="Normal visualisation of OBJ mesh">
    <figcaption><strong>Normal visualisation</strong> — the <code>ShowNormals</code> material reveals the gradient of interpolated normals across every face.</figcaption>
  </figure>
</div>

If the `.obj` has no normals, flat face normals are computed from `cross(e1, e2)`.

---

## BVH with triangle meshes

When `use_bvh: true`, the BVH wraps the entire OBJ mesh in a single AABB. Within-mesh
triangle acceleration (a per-mesh BVH) is on the roadmap but not yet implemented — for meshes
with thousands of triangles, the current configuration will test every triangle once the outer
AABB is hit.

For best performance with large meshes, keep the scene BVH enabled and limit meshes to
a few thousand triangles.

---

## Included models

Three `.obj` test models are provided in `resources/models/`:

| File | Triangles | Source |
|---|---|---|
| `bunny.obj` | ~16 k | Stanford Bunny (Stanford Scanning Repository) |
| `dragon.obj` | ~100 k | Stanford Dragon (Stanford Scanning Repository) |
| `statue.obj` | ~30 k | Custom scan |

UV-mapped test models for texture validation (generated by `scripts/generate_uv_models.py`):

| File | Description |
|---|---|
| `plane_uv.obj` | Subdivided ground plane with UV coords |
| `cube_uv.obj` | Cube with per-face UV mapping |
| `sphere_uv.obj` | UV sphere with latitude-longitude UV |

The Stanford models are used under the Stanford 3D Scanning Repository's
[terms of use](https://graphics.stanford.edu/data/3Dscanrep/).
