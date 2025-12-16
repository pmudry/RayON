import os

scenes = {
    "erato": "resources/scenes/benchmark/erato_in_box.yaml",
    "conf_room_2": "resources/scenes/benchmark/conference_room_2.yaml",
    "conf_room_1": "resources/scenes/benchmark/conference_room_1.yaml"
}

variations = [
    {"suffix": "hd_std", "width": 1920, "height": 1080, "samples": 2048, "time": 60},
    {"suffix": "hd_high", "width": 1920, "height": 1080, "samples": 4096, "time": 180},
    {"suffix": "4k_fill", "width": 3840, "height": 2160, "samples": 8192, "time": 300},
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