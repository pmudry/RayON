import yaml

# Configure YAML to disable aliases
class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

def generate_yaml_individual(obj_filename, yaml_filename, mat_name="benchmark_mat"):
    """
    Reads an OBJ file and writes a YAML scene file with explicit individual triangles.
    """
    vertices = []
    triangles = []
    
    with open(obj_filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.strip().split()
                # OBJ indices are 1-based
                idx0 = int(parts[1]) - 1
                idx1 = int(parts[2]) - 1
                idx2 = int(parts[3]) - 1
                triangles.append((idx0, idx1, idx2))

    scene_data = {
        "scene": {
            "name": "Benchmark Individual Triangles",
            "camera": {
                "position": [0.0, 1.5, 4.0],
                "look_at": [0.0, 1.0, 0.0],
                "up": [0.0, 1.0, 0.0],
                "fov": 40.0
            },
            "background_color": [0.1, 0.1, 0.1],
            "use_bvh": True
        },
        "materials": [
            {
                "name": mat_name,
                "type": "lambertian",
                "albedo": [0.8, 0.2, 0.2], # Reddish
                "pattern": {
                     "type": "checkerboard",
                     "color": [0.2, 0.2, 0.8], # Blueish
                     "dot_count": 10.0 # abused as scale for now
                }
            },
            {
                "name": "floor_mat",
                "type": "lambertian",
                "albedo": [0.5, 0.5, 0.5]
            },
            {
                "name": "light",
                "type": "light",
                "emission": [15.0, 15.0, 15.0],
                "albedo": [1.0, 1.0, 1.0]
            }
        ],
        "geometry": [
            # Floor
            {
                "type": "rectangle",
                "material": "floor_mat",
                "corner": [-2.0, 0.0, -2.0],
                "u": [4.0, 0.0, 0.0],
                "v": [0.0, 0.0, 4.0]
            },
            # Light
            {
                "type": "rectangle",
                "material": "light",
                "corner": [-0.5, 3.5, -0.5],
                "u": [1.0, 0.0, 0.0],
                "v": [0.0, 0.0, 1.0]
            }
        ]
    }
    
    # Add triangles
    for tri in triangles:
        v0 = vertices[tri[0]]
        v1 = vertices[tri[1]]
        v2 = vertices[tri[2]]
        
        scene_data["geometry"].append({
            "type": "triangle",
            "material": mat_name,
            "v0": v0,
            "v1": v1,
            "v2": v2
        })

    with open(yaml_filename, 'w') as f:
        # Use the custom NoAliasDumper
        yaml.dump(scene_data, f, Dumper=NoAliasDumper, default_flow_style=None)

def generate_yaml_mesh(obj_filename, yaml_filename, mat_name="benchmark_mat"):
    """
    Generates a YAML scene file referencing the OBJ file (Mesh mode).
    """
    scene_data = {
        "scene": {
            "name": "Benchmark Mesh",
            "camera": {
                "position": [0.0, 1.5, 4.0],
                "look_at": [0.0, 1.0, 0.0],
                "up": [0.0, 1.0, 0.0],
                "fov": 40.0
            },
            "background_color": [0.1, 0.1, 0.1],
            "use_bvh": True
        },
        "materials": [
            {
                "name": mat_name,
                "type": "lambertian",
                "albedo": [0.2, 0.8, 0.2], # Greenish to distinguish
                "pattern": {
                     "type": "checkerboard",
                     "color": [0.8, 0.8, 0.2], # Yellowish
                }
            },
            {
                "name": "floor_mat",
                "type": "lambertian",
                "albedo": [0.5, 0.5, 0.5]
            },
            {
                "name": "light",
                "type": "light",
                "emission": [15.0, 15.0, 15.0],
                "albedo": [1.0, 1.0, 1.0]
            }
        ],
        "geometry": [
            # Floor
            {
                "type": "rectangle",
                "material": "floor_mat",
                "corner": [-2.0, 0.0, -2.0],
                "u": [4.0, 0.0, 0.0],
                "v": [0.0, 0.0, 4.0]
            },
            # Light
            {
                "type": "rectangle",
                "material": "light",
                "corner": [-0.5, 3.5, -0.5],
                "u": [1.0, 0.0, 0.0],
                "v": [0.0, 0.0, 1.0]
            },
            # The Mesh
            {
                "type": "obj",
                "filename": obj_filename.split('/')[-1], # Relative filename
                "material": mat_name,
                "position": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
                "rotation": [0.0, 0.0, 0.0]
            }
        ]
    }
    
    with open(yaml_filename, 'w') as f:
        # Use the custom NoAliasDumper (though less critical here)
        yaml.dump(scene_data, f, Dumper=NoAliasDumper, default_flow_style=None)

if __name__ == "__main__":
    counts = [100, 250, 500]
    base_path = "resources/scenes/triangle-demo/"
    
    for c in counts:
        obj = f"{base_path}pyramid_{c}.obj"
        
        # Generate Individual Triangle Scene (Baseline)
        generate_yaml_individual(obj, f"{base_path}scene_individual_{c}.yaml")
        
        # Generate Mesh Scene (Optimized)
        generate_yaml_mesh(obj, f"{base_path}scene_mesh_{c}.yaml")
        
    print("Generated YAML scene files.")
