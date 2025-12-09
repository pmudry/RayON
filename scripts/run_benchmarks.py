#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import json
import glob
import time
from pathlib import Path

# ANSI colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def find_executable(build_dir="build"):
    """Find the rayon executable in the build directory."""
    exe_name = "rayon"
    if os.name == 'nt':
        exe_name += ".exe"
    
    # Try common paths
    paths = [
        os.path.join(build_dir, exe_name),
        os.path.join(build_dir, "Release", exe_name),
        os.path.join(build_dir, "Debug", exe_name),
        os.path.join(".", exe_name)
    ]
    
    for p in paths:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return os.path.abspath(p)
            
    return None

def run_benchmark(executable, config_file, output_dir=None, profile_nsys=False, profile_ncu=False):
    """Run a single benchmark configuration."""
    print(f"{Colors.OKBLUE}Running benchmark: {config_file}{Colors.ENDC}")
    
    cmd = [executable, "--benchmark", config_file]
    if output_dir:
        cmd.extend(["--benchmark-out", output_dir])
    
    # Profiling integration
    if profile_nsys or profile_ncu:
        prof_dir = os.path.join(output_dir if output_dir else "benchmark_results", "profiling")
        os.makedirs(prof_dir, exist_ok=True)
        
        config_name = Path(config_file).stem
        report_path = os.path.join(prof_dir, f"{config_name}")

        import shutil
        
        if profile_nsys:
            nsys_exe = shutil.which("nsys") or "/usr/local/cuda/bin/nsys"
            print(f"  {Colors.HEADER}[Profiling] Nsight Systems: {report_path}.nsys-rep{Colors.ENDC}")
            # -t cuda,osrt,nvtx: Trace CUDA, OS runtime, and NVTX
            profiler_cmd = [nsys_exe, "profile", "-t", "cuda,osrt,nvtx", "-o", report_path, "--force-overwrite", "true"]
            cmd = profiler_cmd + cmd
        elif profile_ncu:
            ncu_exe = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
            print(f"  {Colors.HEADER}[Profiling] Nsight Compute: {report_path}.ncu-rep{Colors.ENDC}")
            # --set full: All metrics
            # --launch-count 1: Only profile 1 kernel launch (fixed for older ncu versions)
            profiler_cmd = [ncu_exe, "--set", "full", "-c", "1", "-o", report_path, "--force-overwrite"]
            cmd = profiler_cmd + cmd

    print(f"DEBUG: Executing command: {' '.join(cmd)}")
    try:
        # Run the command
        start_time = time.time()
        
        if profile_nsys or profile_ncu:
            # When profiling, let output stream to console so user can see progress/errors
            # check=False allows us to continue even if profiler fails (e.g. permission error)
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                 print(f"  {Colors.WARNING}Warning: Profiler/Benchmark exited with code {result.returncode}. Attempting to collect results anyway.{Colors.ENDC}")
            stdout_content = "" # We can't parse stdout if we didn't capture it
        else:
            # Capture output for normal runs to keep it clean and parse filename
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            stdout_content = result.stdout

        end_time = time.time()
        
        print(f"  {Colors.OKGREEN}Success ({end_time - start_time:.2f}s){Colors.ENDC}")
        
        output_json = None
        
        if stdout_content:
            # Find the output JSON path from stdout
            for line in stdout_content.splitlines():
                if "Saving benchmark results to:" in line:
                    path_str = line.split("Saving benchmark results to:")[1].strip()
                    path = Path(path_str)
                    json_path = path.with_suffix('.json')
                    if json_path.exists():
                        output_json = json_path
                        break
        
        # Fallback: if we didn't capture stdout (profiling), look for the newest JSON in the output dir
        if not output_json and output_dir:
             search_pattern = os.path.join(output_dir, "*.json")
             candidates = glob.glob(search_pattern)
             
             if candidates:
                 # Get the one with the latest modification time
                 newest = max(candidates, key=os.path.getmtime)
                 mtime = os.path.getmtime(newest)
                 
                 # Check if it was created during our run window (approx)
                 # Allow a small buffer before start_time just in case of clock skew, though unlikely
                 if mtime >= start_time - 10: 
                     output_json = newest
                 else:
                     print(f"  {Colors.WARNING}Warning: Found JSON {newest} but it seems too old (diff: {mtime - start_time:.1f}s).{Colors.ENDC}")
             else:
                 print(f"  {Colors.WARNING}Warning: No JSON files found in {output_dir}.{Colors.ENDC}")

        if output_json:
            with open(output_json, 'r') as f:
                data = json.load(f)
            return data
        else:
            print(f"  {Colors.WARNING}Warning: Could not locate output JSON file from logs.{Colors.ENDC}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"  {Colors.FAIL}Failed!{Colors.ENDC}")
        print(e.stdout)
        print(e.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description="RayON Batch Benchmark Runner")
    parser.add_argument("configs", nargs='+', help="Benchmark configuration YAML files or directories")
    parser.add_argument("--exec", help="Path to rayon executable", default=None)
    parser.add_argument("--output", help="Output summary CSV file", default="benchmark_results/summary.csv")
    parser.add_argument("--name", help="Name for this benchmark run (creates subfolder)", default=None)
    parser.add_argument("--profile-nsys", action="store_true", help="Enable Nsight Systems profiling")
    parser.add_argument("--profile-ncu", action="store_true", help="Enable Nsight Compute profiling (single kernel)")
    args = parser.parse_args()

    # Find executable
    executable = args.exec
    if not executable:
        executable = find_executable()
    
    if not executable:
        print(f"{Colors.FAIL}Error: Could not find 'rayon' executable. Build the project first or specify --exec.{Colors.ENDC}")
        sys.exit(1)
        
    print(f"{Colors.HEADER}Using executable: {executable}{Colors.ENDC}")

    # Setup output directory
    base_output_dir = "benchmark_results"
    if args.name:
        run_output_dir = os.path.join(base_output_dir, args.name)
    else:
        run_output_dir = base_output_dir
        
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"{Colors.HEADER}Output directory: {run_output_dir}{Colors.ENDC}")

    # Update summary path if using default and name is provided
    if args.output == "benchmark_results/summary.csv" and args.name:
        args.output = os.path.join(run_output_dir, "summary.csv")

    # Gather config files
    config_files = []
    for path in args.configs:
        if os.path.isdir(path):
            config_files.extend(glob.glob(os.path.join(path, "*.yaml")))
            config_files.extend(glob.glob(os.path.join(path, "*.yml")))
        elif os.path.isfile(path):
            config_files.append(path)
        else:
            print(f"{Colors.WARNING}Warning: {path} not found.{Colors.ENDC}")

    if not config_files:
        print(f"{Colors.FAIL}No configuration files found.{Colors.ENDC}")
        sys.exit(1)

    print(f"Found {len(config_files)} benchmark configurations.")
    print("-" * 60)

    results = []

    # Run benchmarks
    for config in config_files:
        res = run_benchmark(executable, config, run_output_dir, args.profile_nsys, args.profile_ncu)
        if res:
            results.append(res)
    
    if not results:
        print("No results collected.")
        sys.exit(1)

    # Generate Summary
    print("\n" + "=" * 80)
    print(f"{Colors.HEADER}BENCHMARK SUMMARY{Colors.ENDC}")
    print("=" * 80)
    
    # Define columns
    headers = [
        "Device", "Resolution", "Samples", 
        "Rays/Sec", "Time", "VRAM (MB)", "Scene"
    ]
    
    rows = []
    for r in results:
        device = r.get("hardware", {}).get("device_name", "Unknown")
        res_w = r.get("resolution", {}).get("width", 0)
        res_h = r.get("resolution", {}).get("height", 0)
        resolution = f"{res_w}x{res_h}"
        samples = r.get("samples_per_pixel", 0)
        
        rays_sec = r.get("performance", {}).get("rays_per_second", 0)
        rays_sec_fmt = f"{rays_sec / 1_000_000:.2f} M"
        
        time_pretty = r.get("render_time_pretty", "N/A")
        
        vram_bytes = r.get("hardware", {}).get("vram_usage_bytes", 0)
        vram_mb = f"{vram_bytes / (1024*1024):.1f}"
        
        # Extract scene name from image filename or config
        image_name = r.get("image", "")
        # Try to clean up name
        scene_name = image_name.replace(".png", "")
        # Remove timestamp prefix if present
        if len(scene_name) > 20 and scene_name[4] == '-' and scene_name[7] == '-':
             # Heuristic: 2025-12-09_12-41-07_name
             parts = scene_name.split('_', 2)
             if len(parts) > 2:
                 scene_name = parts[2]
        
        rows.append([device, resolution, str(samples), rays_sec_fmt, time_pretty, vram_mb, scene_name])

    # Print Table
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))
            
    # Print Header
    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_str)
    print("-" * len(header_str))
    
    # Print Rows
    for row in rows:
        print(" | ".join(f"{val:<{w}}" for val, w in zip(row, col_widths)))

    # Save to CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")
            
    print(f"\n{Colors.OKGREEN}Summary saved to: {args.output}{Colors.ENDC}")

if __name__ == "__main__":
    main()