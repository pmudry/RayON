# OBJ Loading

RayON imports triangle meshes from Wavefront `.obj` files and renders them with the same
materials, lights and acceleration as the analytic geometry. Materials can come from the YAML
scene or from the `.mtl` file the OBJ references, and UV-mapped image textures are sampled on
the GPU.

---

## Supported format

| Feature | Supported |
|---|---|
| Vertex positions (`v`) | ✓ |
| Vertex normals (`vn`) | ✓ smooth per-vertex normals |
| Texture coordinates (`vt`) | ✓ UV-mapped image textures |
| Triangular faces (`f`) | ✓ |
| Quad faces | triangulated into two triangles |
| Material groups (`usemtl`) | ✓ per-group materials from the MTL |
| MTL files (`mtllib`) | ✓ `Kd`, `Ke`, `Ni`, `Ns`, `d`, `illum`, `map_Kd` |

Faces with more than four vertices are not supported; export triangulated meshes.

---

## Loading a mesh

```yaml
geometry:
  - type: "obj"
    material: "plastic_white"      # optional: fallback for groups without usemtl
    file: "../models/stanford-bunny.obj"
    scale: [1.0, 1.0, 1.0]         # per-axis scale
    position: [0.0, 0.0, 0.0]      # translation applied after scaling
```

The `file` path is resolved relative to the scene file. When the OBJ carries a `mtllib`, the
`material` key may be omitted and every `usemtl` group uses its MTL material; a YAML material
is the fallback for faces without one.

Each triangle becomes one geometry entry of the scene description, so a 250 000-triangle dragon
is 250 000 entries: enable `use_bvh: true` for any mesh.

---

## MTL materials

| MTL property | Effect in RayON |
|---|---|
| `Kd r g b` | Diffuse albedo |
| `Ke r g b` | Emission; a non-zero value makes a `light` material |
| `Ns exponent` | Converted to roughness (`1 − √(Ns / 1000)`) |
| `Ni ior` | Refractive index of glass materials |
| `d opacity` | Below 0.5 selects a glass material |
| `illum 5` / `7` | Mirror / glass |
| `map_Kd path` | Image texture, resolved relative to the `.mtl` file |

---

## Image textures

A `map_Kd` texture, or a `texture:` key on a YAML material, is decoded with `stb_image` and
uploaded as a CUDA texture object. At a hit the UVs are interpolated with the barycentric
weights of the triangle,

$$
(u, v)_\text{hit} = (1-u-v)\,(u_0,v_0) + u\,(u_1,v_1) + v\,(u_2,v_2),
$$

and the texel replaces the material's albedo. Textures are loaded once and shared by path.

```yaml
materials:
  - name: "textured_floor"
    type: "lambertian"
    texture: "../textures/grid_512.png"
```

<div class="img-grid cols-2">
  <figure>
    <img src="../../assets/images/samples/textures/uv_mapping.png" alt="UV texture mapped cube and sphere">
    <figcaption><strong>UV mapping</strong> — a grid texture on a UV-mapped cube, sphere and ground plane (<code>13_texture_test1.yaml</code>).</figcaption>
  </figure>
  <figure>
    <img src="../../assets/images/samples/textures/texture_orig.png" alt="Textured OBJ scene">
    <figcaption><strong>MTL-driven textures</strong> — per-group materials of a Blender export (<code>14_texture_test2.yaml</code>).</figcaption>
  </figure>
</div>

---

## Triangle intersection

The CUDA back-end intersects triangles with the **Möller–Trumbore** algorithm. For a ray
\(\mathbf{o} + t\mathbf{d}\) and a triangle \((\mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2)\), with
\(\mathbf{e}_1 = \mathbf{v}_1 - \mathbf{v}_0\), \(\mathbf{e}_2 = \mathbf{v}_2 - \mathbf{v}_0\),
\(\mathbf{p} = \mathbf{d} \times \mathbf{e}_2\), \(\mathbf{t} = \mathbf{o} - \mathbf{v}_0\) and
\(\mathbf{q} = \mathbf{t} \times \mathbf{e}_1\):

\[
\begin{pmatrix} t \\ u \\ v \end{pmatrix} =
\frac{1}{\mathbf{p} \cdot \mathbf{e}_1}
\begin{pmatrix} \mathbf{q} \cdot \mathbf{e}_2 \\ \mathbf{p} \cdot \mathbf{t} \\ \mathbf{q} \cdot \mathbf{d} \end{pmatrix},
\qquad
\text{hit} \iff u \ge 0,\; v \ge 0,\; u + v \le 1,\; t > t_\min .
\]

The OptiX back-end hands every triangle to the **RT cores**: all triangles of the scene go into
one geometry acceleration structure with the built-in intersector, and a per-triangle table
holds the material, the normals, the UVs and the `visible` flag.

### Smooth normals

With `vn` records the shading normal is interpolated across the face,
\(\hat{n} = (1 - u - v)\,\hat{n}_0 + u\,\hat{n}_1 + v\,\hat{n}_2\), which makes a low-polygon
mesh look smooth. Without them the flat face normal is used.

<div class="img-grid cols-2">
  <figure>
    <img src="../../assets/images/samples/obj_loading.png" alt="Smooth OBJ mesh import">
    <figcaption><strong>Smooth normals</strong> — interpolated per-vertex normals on a low-poly mesh.</figcaption>
  </figure>
  <figure>
    <img src="../../assets/images/samples/gui_debug/normals.png" alt="Normal visualisation of an OBJ mesh">
    <figcaption><strong>Normal visualisation</strong> — the <code>show_normals</code> material (key <b>N</b> in the GUI).</figcaption>
  </figure>
</div>

---

## Included models

`resources/models/` ships a set of test meshes, among them:

| File | Triangles | Used by |
|---|---|---|
| `bunny-3.obj`, `bunny-2.obj`, `stanford-bunny.obj` | 22 k, 69 k, 69 k | `10_isc_neons_bunny`, `17_cloud_bunny` |
| `xyzrgb_dragon.obj` | 250 k | `04_obj_dragon` |
| `dragon.obj` | 871 k | the largest mesh, for BVH tests |
| `victory.obj` | 17 k | `04_obj_statue` |
| `nefertiti.obj`, `nefertiti2.obj` | 100 k, 50 k | |
| `suzanne.obj`, `teapot.obj` | 500, 6 k | `12_clearcoat_pokemonball` |
| `tetrahedron`, `cube`, `octahedron`, `dodecahedron`, `icosahedron` | 4 to 20 | `03_platonic_solids` |
| `cube_uv.obj`, `sphere_uv.obj`, `plane_uv.obj` | | `13_texture_test1` (generated by `scripts/generate_uv_models.py`) |
| `shelby-car.obj`, `sci-fi.obj`, `PokemonBall.obj`, `isc-text.obj` | 291 k, 37 k, 7 k, 15 k | |

The Stanford models are used under the
[Stanford 3D Scanning Repository terms](https://graphics.stanford.edu/data/3Dscanrep/).
