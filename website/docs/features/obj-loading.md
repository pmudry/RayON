# OBJ Loading

RayON can import triangle meshes from Wavefront `.obj` files and render them alongside
analytical geometry using physical path tracing — same materials, same BVH, same illumination.

---

## Supported format

| Feature | Supported |
|---|---|
| Vertex positions (`v`) | ✓ |
| Vertex normals (`vn`) | ✓ — smooth per-vertex normals |
| Texture coordinates (`vt`) | Parsed, not yet used for textures |
| Triangular faces (`f`) | ✓ |
| Quad faces | Auto-triangulated into two triangles |
| Multiple objects / groups | Merged into one mesh per YAML entry |
| MTL material files | Not used — material set via YAML |

!!! note "Triangles only"
    Faces with more than 4 vertices are not supported. Most 3D modelling tools can export
    triangulated meshes via an export option.

---

## Loading a mesh

In a YAML scene file:

```yaml
geometry:
  - type: obj_mesh
    material: plastic_white
    file: "../resources/models/bunny.obj"
    scale: [2.0, 2.0, 2.0]       # uniform or per-axis scale
    offset: [0.0, 0.0, 0.0]      # translation after scaling
```

The `file` path is resolved **relative to the YAML scene file's directory**.

Loading multiple meshes:

```yaml
geometry:
  - type: obj_mesh
    material: gold
    file: "../resources/models/dragon.obj"
    scale: [0.5, 0.5, 0.5]
    offset: [-1.5, 0.0, 0.0]

  - type: obj_mesh
    material: glass
    file: "../resources/models/bunny.obj"
    scale: [1.0, 1.0, 1.0]
    offset: [1.5, 0.0, 0.0]
```

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
    <img src="../assets/images/samples/obj_loading.png" alt="Smooth OBJ mesh import">
    <figcaption><strong>Smooth normals</strong> — interpolated per-vertex normals make the low-poly mesh look smooth.</figcaption>
  </figure>
  <figure>
    <img src="../assets/images/samples/normals.png" alt="Normal visualisation of OBJ mesh">
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

The Stanford models are used under the Stanford 3D Scanning Repository's
[terms of use](https://graphics.stanford.edu/data/3Dscanrep/).
