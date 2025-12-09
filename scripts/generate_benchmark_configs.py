import os

scenes = {
    "erato": "resources/experiments/benchmark/erato_in_box.yaml",
    "cornell": "resources/scenes/cornell_box.yaml",
    "mesh_500": "resources/experiments/triangle-demo/scene_mesh_500.yaml"
}

variations = [
    {"suffix": "quick", "width": 1280, "height": 720, "samples": 32, "time": 30},
    {"suffix": "hd_std", "width": 1920, "height": 1080, "samples": 128, "time": 60},
    {"suffix": "hd_high", "width": 1920, "height": 1080, "samples": 512, "time": 180},
    {"suffix": "4k_fill", "width": 3840, "height": 2160, "samples": 64, "time": 120},
    {"suffix": "converge", "width": 1280, "height": 720, "samples": 1024, "time": 300}
]

output_dir = "benchmark_configs"
os.makedirs(output_dir, exist_ok=True)

for scene_name, scene_path in scenes.items():
    for var in variations:
        filename = f"{scene_name}_{var['suffix']}.yaml"
        
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write("benchmark:\n")
            f.write(f"  scene_file: \"{scene_path}\"\n")
            f.write(f"  output_name: \"{scene_name}_{var['suffix']}\"\n")
            f.write(f"  target_samples: {var['samples']}\n")
            f.write(f"  max_time_seconds: {float(var['time'])}\n")
            f.write(f"  resolution_width: {var['width']}\n")
            f.write(f"  resolution_height: {var['height']}\n")

print(f"Generated {len(scenes) * len(variations)} config files in {output_dir}")