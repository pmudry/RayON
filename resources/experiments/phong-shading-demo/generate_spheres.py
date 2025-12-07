import math
import os

def write_sphere(filename, mtl_filename, material_name, radius, rings, sectors, smooth_normals=True):
    with open(filename, 'w') as f:
        f.write(f"mtllib {mtl_filename}\n")
        f.write(f"o Sphere\n")
        
        # Vertices
        for r in range(rings + 1):
            for s in range(sectors + 1):
                y = math.cos(math.pi * r / rings)
                x = math.cos(2 * math.pi * s / sectors) * math.sin(math.pi * r / rings)
                z = math.sin(2 * math.pi * s / sectors) * math.sin(math.pi * r / rings)
                
                f.write(f"v {x * radius} {y * radius} {z * radius}\n")
                
        # Normals (only if smooth)
        if smooth_normals:
            for r in range(rings + 1):
                for s in range(sectors + 1):
                    y = math.cos(math.pi * r / rings)
                    x = math.cos(2 * math.pi * s / sectors) * math.sin(math.pi * r / rings)
                    z = math.sin(2 * math.pi * s / sectors) * math.sin(math.pi * r / rings)
                    f.write(f"vn {x} {y} {z}\n")

        # UVs
        for r in range(rings + 1):
            for s in range(sectors + 1):
                u = s / sectors
                v = r / rings
                f.write(f"vt {u} {v}\n")

        f.write(f"usemtl {material_name}\n")
        f.write("s 1\n" if smooth_normals else "s off\n")

        # Faces
        for r in range(rings):
            for s in range(sectors):
                # 1-based indexing
                p1 = r * (sectors + 1) + s + 1
                p2 = p1 + sectors + 1
                p3 = p1 + 1
                p4 = p2 + 1
                
                if smooth_normals:
                    f.write(f"f {p1}/{p1}/{p1} {p2}/{p2}/{p2} {p3}/{p3}/{p3}\n")
                    f.write(f"f {p3}/{p3}/{p3} {p2}/{p2}/{p2} {p4}/{p4}/{p4}\n")
                else:
                    f.write(f"f {p1}/{p1} {p2}/{p2} {p3}/{p3}\n")
                    f.write(f"f {p3}/{p3} {p2}/{p2} {p4}/{p4}\n")

def write_mtl(filename, material_name, color):
    with open(filename, 'w') as f:
        f.write(f"newmtl {material_name}\n")
        f.write("Ns 0.0000\n")
        f.write(f"Kd {color[0]} {color[1]} {color[2]}\n")
        f.write("Ka 0.0000 0.0000 0.0000\n")
        f.write("Ks 0.0000 0.0000 0.0000\n")
        f.write("Ke 0.0000 0.0000 0.0000\n")
        f.write("Ni 1.0000\n")
        f.write("d 1.0000\n")
        f.write("illum 1\n")

# Shared Settings
RADIUS = 1.0
COLOR = (0.8, 0.4, 0.3) # Coral/Clay
base_dir = "resources/experiments/phong-shading-demo"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

mtl_file = "phong_demo.mtl"
write_mtl(os.path.join(base_dir, mtl_file), "demo_matte", COLOR)

# 1. Very High (Existing) - 64x64
RINGS_HIGH = 64
SECTORS_HIGH = 64
write_sphere(os.path.join(base_dir, "sphere_flat_very_high.obj"), mtl_file, "demo_matte", RADIUS, RINGS_HIGH, SECTORS_HIGH, smooth_normals=False)
write_sphere(os.path.join(base_dir, "sphere_smooth_very_high.obj"), mtl_file, "demo_matte", RADIUS, RINGS_HIGH, SECTORS_HIGH, smooth_normals=True)
print(f"Generated High Res: {RINGS_HIGH*SECTORS_HIGH*2} triangles")

# 2. Medium (Less faces) - 24x24
RINGS_MED = 24
SECTORS_MED = 24
write_sphere(os.path.join(base_dir, "sphere_flat_medium.obj"), mtl_file, "demo_matte", RADIUS, RINGS_MED, SECTORS_MED, smooth_normals=False)
write_sphere(os.path.join(base_dir, "sphere_smooth_medium.obj"), mtl_file, "demo_matte", RADIUS, RINGS_MED, SECTORS_MED, smooth_normals=True)
print(f"Generated Medium Res: {RINGS_MED*SECTORS_MED*2} triangles")

# 3. Low (Way more less faces) - 12x12
RINGS_LOW = 12
SECTORS_LOW = 12
write_sphere(os.path.join(base_dir, "sphere_flat_low.obj"), mtl_file, "demo_matte", RADIUS, RINGS_LOW, SECTORS_LOW, smooth_normals=False)
write_sphere(os.path.join(base_dir, "sphere_smooth_low.obj"), mtl_file, "demo_matte", RADIUS, RINGS_LOW, SECTORS_LOW, smooth_normals=True)
print(f"Generated Low Res: {RINGS_LOW*SECTORS_LOW*2} triangles")