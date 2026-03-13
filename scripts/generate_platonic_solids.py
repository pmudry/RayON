#!/usr/bin/env python3
"""Generate OBJ files for all five Platonic solids (unit-radius, centered at origin)."""

import math
import os

phi = (1 + math.sqrt(5)) / 2
inv_phi = 1 / phi

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def write_obj(filename, vertices, faces):
    """Write an OBJ file without vertex normals (flat shading from face geometry)."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        f.write(f"# {filename} - auto-generated Platonic solid\n")
        for v in vertices:
            f.write(f"v  {v[0]:.6f}  {v[1]:.6f}  {v[2]:.6f}\n")
        f.write("\n")
        for face in faces:
            parts = " ".join(str(idx) for idx in face)
            f.write(f"f {parts}\n")
    print(f"  {filename}: {len(vertices)} vertices, {len(faces)} faces")


def normalize(verts, radius=1.0):
    """Scale vertices so they lie on a sphere of the given radius."""
    r = max(math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) for v in verts)
    s = radius / r
    return [(v[0] * s, v[1] * s, v[2] * s) for v in verts]


def ensure_outward_winding(verts_raw, face_0indexed):
    """Flip face winding if normal points inward (toward origin)."""
    cx = sum(verts_raw[v][0] for v in face_0indexed) / len(face_0indexed)
    cy = sum(verts_raw[v][1] for v in face_0indexed) / len(face_0indexed)
    cz = sum(verts_raw[v][2] for v in face_0indexed) / len(face_0indexed)
    v0, v1, v2 = face_0indexed[0], face_0indexed[1], face_0indexed[2]
    e1 = [verts_raw[v1][k] - verts_raw[v0][k] for k in range(3)]
    e2 = [verts_raw[v2][k] - verts_raw[v0][k] for k in range(3)]
    nx = e1[1] * e2[2] - e1[2] * e2[1]
    ny = e1[2] * e2[0] - e1[0] * e2[2]
    nz = e1[0] * e2[1] - e1[1] * e2[0]
    if nx * cx + ny * cy + nz * cz < 0:
        face_0indexed = face_0indexed[::-1]
    return [v + 1 for v in face_0indexed]  # convert to 1-indexed


# ========================= TETRAHEDRON =========================
def make_tetrahedron():
    verts = normalize([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)])
    faces_0 = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    faces = [ensure_outward_winding(verts, f) for f in faces_0]
    write_obj("tetrahedron.obj", verts, faces)


# ========================= CUBE =========================
def make_cube():
    s = 1.0 / math.sqrt(3)
    verts = [
        (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
        (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s),
    ]
    faces_0 = [
        [4, 5, 6, 7], [1, 0, 3, 2], [1, 2, 6, 5],
        [0, 4, 7, 3], [3, 7, 6, 2], [0, 1, 5, 4],
    ]
    faces = [ensure_outward_winding(verts, f) for f in faces_0]
    write_obj("cube.obj", verts, faces)


# ========================= OCTAHEDRON =========================
def make_octahedron():
    verts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    faces_0 = [
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ]
    faces = [ensure_outward_winding(verts, f) for f in faces_0]
    write_obj("octahedron.obj", verts, faces)


# ========================= DODECAHEDRON =========================
def make_dodecahedron():
    verts_raw = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
        (0, phi, inv_phi), (0, phi, -inv_phi),
        (0, -phi, inv_phi), (0, -phi, -inv_phi),
        (inv_phi, 0, phi), (-inv_phi, 0, phi),
        (inv_phi, 0, -phi), (-inv_phi, 0, -phi),
        (phi, inv_phi, 0), (phi, -inv_phi, 0),
        (-phi, inv_phi, 0), (-phi, -inv_phi, 0),
    ]
    verts = normalize(verts_raw)
    # 12 pentagonal faces (0-indexed), verified: 30 edges each shared by exactly 2 faces
    faces_0 = [
        [0, 16, 17, 2, 12], [0, 12, 13, 4, 8], [0, 8, 9, 1, 16],
        [1, 9, 5, 15, 14], [1, 14, 3, 17, 16], [2, 17, 3, 11, 10],
        [2, 10, 6, 13, 12], [3, 14, 15, 7, 11], [4, 13, 6, 19, 18],
        [4, 18, 5, 9, 8], [5, 18, 19, 7, 15], [6, 10, 11, 7, 19],
    ]
    faces = [ensure_outward_winding(verts_raw, f) for f in faces_0]
    write_obj("dodecahedron.obj", verts, faces)


# ========================= ICOSAHEDRON =========================
def make_icosahedron():
    verts_raw = [
        (0, 1, phi), (0, 1, -phi), (0, -1, phi), (0, -1, -phi),
        (1, phi, 0), (1, -phi, 0), (-1, phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1),
    ]
    verts = normalize(verts_raw)
    faces_0 = [
        [0, 2, 8], [0, 8, 4], [0, 4, 6], [0, 6, 10], [0, 10, 2],
        [1, 3, 9], [1, 9, 4], [1, 4, 6], [1, 6, 11], [1, 11, 3],
        [2, 8, 5], [2, 5, 7], [2, 7, 10],
        [3, 9, 5], [3, 5, 7], [3, 7, 11],
        [8, 4, 9], [8, 5, 9], [10, 6, 11], [10, 7, 11],
    ]
    faces = [ensure_outward_winding(verts_raw, f) for f in faces_0]
    write_obj("icosahedron.obj", verts, faces)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating Platonic solids in", OUTPUT_DIR)
    make_tetrahedron()
    make_cube()
    make_octahedron()
    make_dodecahedron()
    make_icosahedron()
    print("Done.")
