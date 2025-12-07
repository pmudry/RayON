import math

def generate_pyramid(num_triangles, radius=1.0, height=2.0, filename="pyramid.obj"):
    """
    Generates a cone-like pyramid approximating the requested number of triangles.
    The base is a regular polygon.
    num_triangles: Total triangles desired.
      - The base takes N triangles (fan).
      - The sides take N triangles.
      - Total = 2 * N.
      - So N = num_triangles / 2.
    """
    
    # We need an even number of triangles roughly
    n_sides = max(3, int(num_triangles / 2))
    
    vertices = []
    # Apex
    vertices.append((0.0, height, 0.0)) # v1
    # Base Center
    vertices.append((0.0, 0.0, 0.0))    # v2
    
    # Base perimeter vertices
    for i in range(n_sides):
        angle = 2.0 * math.pi * i / n_sides
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        vertices.append((x, 0.0, z))
        
    with open(filename, 'w') as f:
        f.write("# Generated Pyramid\n")
        f.write(f"# Target triangles: {num_triangles}, Actual: {2 * n_sides}\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
        # Write faces
        # Base center is index 2
        # Apex is index 1
        # Perimeter vertices start at index 3
        
        # Side faces (Apex -> current -> next)
        for i in range(n_sides):
            current_v = 3 + i
            next_v = 3 + (i + 1) % n_sides
            if (i + 1) % n_sides == 0: # Wrap around logic correction
                 next_v = 3
            
            f.write(f"f 1 {current_v} {next_v}\n")
            
        # Base faces (Center -> next -> current) (to face down)
        for i in range(n_sides):
            current_v = 3 + i
            next_v = 3 + (i + 1) % n_sides
            if (i + 1) % n_sides == 0:
                 next_v = 3
                 
            f.write(f"f 2 {next_v} {current_v}\n")

if __name__ == "__main__":
    generate_pyramid(100, filename="resources/experiments/benchmark/pyramid_100.obj")
    generate_pyramid(250, filename="resources/experiments/benchmark/pyramid_250.obj")
    generate_pyramid(500, filename="resources/experiments/benchmark/pyramid_500.obj")
    print("Generated pyramid OBJs.")
