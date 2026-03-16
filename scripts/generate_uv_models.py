#!/usr/bin/env python3
"""Generate UV-mapped OBJ models for the texture test scene.

Outputs:
  resources/models/plane_uv.obj   — a 4x4 ground plane in XZ, 5x5 subdivisions
  resources/models/cube_uv.obj    — a 1x1x1 cube with per-face UV mapping
  resources/models/sphere_uv.obj  — a UV sphere (latitude-longitude) with correct UV
  resources/models/texture_test.mtl — MTL referencing the grid texture

Usage: python3 scripts/generate_uv_models.py
"""

import math
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'resources', 'models')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_obj(path, name, vertices, uvs, normals, faces, mtl_lib, mtl_name):
    """Write an OBJ file.  faces: list of (vi_list, uvi_list, ni_list) — 1-based."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(f"# {name}\n")
        f.write(f"mtllib texture_test.mtl\n\n")
        f.write(f"o {name}\n\n")
        for v in vertices:
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(*v))
        f.write('\n')
        for uv in uvs:
            f.write("vt {:.6f} {:.6f}\n".format(*uv))
        f.write('\n')
        for n in normals:
            f.write("vn {:.6f} {:.6f} {:.6f}\n".format(*n))
        f.write('\n')
        f.write(f"usemtl {mtl_name}\n")
        for (vis, uvis, nis) in faces:
            tokens = []
            for vi, uvi, ni in zip(vis, uvis, nis):
                tokens.append(f"{vi}/{uvi}/{ni}")
            f.write("f " + " ".join(tokens) + "\n")
    print(f"Written: {path}")


# ---------------------------------------------------------------------------
# Plane (4x4 units, lying in XZ plane at y=0, subdivided 5x5)
# ---------------------------------------------------------------------------

def gen_plane(filepath, size=4.0, subdivs=5, mtl_name='textured'):
    verts, uvs, norms, faces = [], [], [], []
    N = normal = (0, 1, 0)

    # Grid of (subdivs+1) x (subdivs+1) vertices
    def idx(ix, iz):
        return ix * (subdivs + 1) + iz + 1  # 1-based

    for ix in range(subdivs + 1):
        for iz in range(subdivs + 1):
            x = -size / 2 + ix * size / subdivs
            z = -size / 2 + iz * size / subdivs
            verts.append((x, 0.0, z))
            uvs.append((ix / subdivs, iz / subdivs))
            norms.append(N)

    # Two triangles per cell
    for ix in range(subdivs):
        for iz in range(subdivs):
            a = idx(ix,     iz)
            b = idx(ix + 1, iz)
            c = idx(ix + 1, iz + 1)
            d = idx(ix,     iz + 1)
            # Triangle 1: a, b, c
            faces.append(([a, b, c], [a, b, c], [a, b, c]))
            # Triangle 2: a, c, d
            faces.append(([a, c, d], [a, c, d], [a, c, d]))

    write_obj(filepath, 'Plane', verts, uvs, norms, faces, 'texture_test.mtl', mtl_name)


# ---------------------------------------------------------------------------
# Cube (1x1x1, centered at origin, each face has its own UV square 0..1)
# ---------------------------------------------------------------------------

def gen_cube(filepath, size=1.0, mtl_name='textured'):
    s = size / 2
    # 6 faces, each face: 4 vertices, normal, UV corners
    # face_def: (normal, corners CCW-from-outside-viewpoint)
    face_defs = [
        # (+Y top)
        {'n': (0,1,0),  'v': [(-s,s,-s),(s,s,-s),(s,s,s),(-s,s,s)]},
        # (-Y bottom)
        {'n': (0,-1,0), 'v': [(-s,-s,s),(s,-s,s),(s,-s,-s),(-s,-s,-s)]},
        # (+X right)
        {'n': (1,0,0),  'v': [(s,-s,-s),(s,-s,s),(s,s,s),(s,s,-s)]},
        # (-X left)
        {'n': (-1,0,0), 'v': [(-s,-s,s),(-s,-s,-s),(-s,s,-s),(-s,s,s)]},
        # (+Z front)
        {'n': (0,0,1),  'v': [(-s,-s,s),(s,-s,s),(s,s,s),(-s,s,s)]},
        # (-Z back)
        {'n': (0,0,-1), 'v': [(s,-s,-s),(-s,-s,-s),(-s,s,-s),(s,s,-s)]},
    ]
    # UV corners for any face (square mapping)
    uv_corners = [(0,0),(1,0),(1,1),(0,1)]

    verts, uvs_list, norms, faces = [], [], [], []

    for fd in face_defs:
        nx, ny, nz = fd['n']
        base_v = len(verts) + 1  # 1-based
        base_uv = len(uvs_list) + 1
        base_n = len(norms) + 1

        for v in fd['v']:
            verts.append(v)
        for uv in uv_corners:
            uvs_list.append(uv)
        norms.append((nx, ny, nz))  # one normal per face

        # Two triangles: 0-1-2 and 0-2-3
        a, b, c, d = base_v, base_v+1, base_v+2, base_v+3
        ua, ub, uc, ud = base_uv, base_uv+1, base_uv+2, base_uv+3
        n = base_n
        faces.append(([a,b,c],[ua,ub,uc],[n,n,n]))
        faces.append(([a,c,d],[ua,uc,ud],[n,n,n]))

    write_obj(filepath, 'Cube', verts, uvs_list, norms, faces, 'texture_test.mtl', mtl_name)


# ---------------------------------------------------------------------------
# UV Sphere (latitude-longitude parametrisation)
# ---------------------------------------------------------------------------

def gen_sphere(filepath, radius=0.5, stacks=32, slices=64, mtl_name='textured'):
    verts, uvs_list, norms, faces = [], [], [], []

    # Build vertex grid: (stacks+1) rows x (slices+1) cols
    # u = longitude [0..1], v = latitude [0..1]
    def vidx(stack, sl):
        return stack * (slices + 1) + sl + 1  # 1-based

    for si in range(stacks + 1):
        v = si / stacks                   # [0..1]
        phi = v * math.pi                  # [0..pi]
        y = radius * math.cos(phi)
        r = radius * math.sin(phi)
        for sl in range(slices + 1):
            u = sl / slices               # [0..1]
            theta = u * 2 * math.pi        # [0..2pi]
            x = r * math.cos(theta)
            z = r * math.sin(theta)
            nx, ny, nz = x / radius, y / radius, z / radius
            verts.append((x, y, z))
            uvs_list.append((u, v))  # standard OBJ: V=0 at bottom; shader handles the flip
            norms.append((nx, ny, nz))

    # Triangles
    for si in range(stacks):
        for sl in range(slices):
            a = vidx(si,     sl)
            b = vidx(si,     sl + 1)
            c = vidx(si + 1, sl + 1)
            d = vidx(si + 1, sl)
            # Skip degenerate triangles at poles
            if si > 0:
                faces.append(([a, b, c], [a, b, c], [a, b, c]))
            if si < stacks - 1:
                faces.append(([a, c, d], [a, c, d], [a, c, d]))

    write_obj(filepath, 'Sphere', verts, uvs_list, norms, faces, 'texture_test.mtl', mtl_name)


# ---------------------------------------------------------------------------
# MTL file
# ---------------------------------------------------------------------------

def gen_mtl(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write("# MTL file for texture test scene\n\n")

        f.write("newmtl textured\n")
        f.write("illum 1\n")          # Diffuse, no specular
        f.write("Kd 1.0 1.0 1.0\n")  # White base (modulated by texture)
        f.write("Ka 0.0 0.0 0.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write("map_Kd ../textures/grid_4k.png\n\n")

        f.write("newmtl light_mat\n")
        f.write("illum 0\n")
        f.write("Ke 4.0 4.0 4.0\n")  # Emissive
        f.write("Kd 0.0 0.0 0.0\n\n")

    print(f"Written: {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    gen_mtl(os.path.join(OUT_DIR, 'texture_test.mtl'))

    gen_plane(os.path.join(OUT_DIR, 'plane_uv.obj'),  size=6.0, subdivs=6)
    gen_cube( os.path.join(OUT_DIR, 'cube_uv.obj'),   size=1.2)
    gen_sphere(os.path.join(OUT_DIR, 'sphere_uv.obj'), radius=0.6, stacks=32, slices=64)
