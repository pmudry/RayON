# RayON Benchmarking Guide

This guide explains how to run benchmarks for the RayON raytracer, both for single scene configurations and automated batch runs.

## Prerequisites

Before running any benchmarks, ensure you have built the RayON project:

1.  Navigate to the project root directory.
2.  Run CMake to configure and build:
    ```bash
    cmake -B build -S .
    cmake --build build
    ```
    This will create the `rayon` executable in your `build/` directory (e.g., `build/rayon`).

## Single Benchmark Run

To run a single benchmark with a specific configuration, use the `--benchmark` command-line argument with the `rayon` executable.

**Command Syntax:**

```bash
./build/rayon --benchmark <path/to/config_file.yaml>
```

**Example:**

To run the benchmark defined in `my_benchmark_config.yaml`:

```bash
./build/rayon --benchmark my_benchmark_config.yaml
```

**Output:**

After the run, you will find:
*   A rendered image (`.png`) in `rendered_images/latest.png` and `benchmark_results/` (with a timestamped name).
*   A JSON file (`.json`) containing detailed benchmark statistics and scene settings in `benchmark_results/` (with a timestamped name).
*   A plain text file (`.txt`) with a summary of the stats in `benchmark_results/` (with a timestamped name).

## Batch Benchmark Runs (Automated Script)

For running multiple benchmarks with different scenes or parameters, you can use the provided Python automation script: `scripts/run_benchmarks.py`.

### 1. Generating Benchmark Configurations

You can generate a set of diverse benchmark configurations using the `scripts/generate_benchmark_configs.py` script. This script will create a variety of `.yaml` files in the `benchmark_configs/` directory.

**Command:**

```bash
python3 scripts/generate_benchmark_configs.py
```

This will generate 15 YAML files (3 scenes x 5 variations) in the `benchmark_configs/` directory, each defining a specific test case (e.g., `erato_quick.yaml`, `cornell_hd_high.yaml`).

### 2. Running the Batch Benchmarks

The `scripts/run_benchmarks.py` script executes the `rayon` executable for each specified configuration, collects the results, and generates a summary.

**Command Syntax:**

```bash
./scripts/run_benchmarks.py <config_path_1> [<config_path_2> ...] [OPTIONS]
```

`<config_path>` can be:
*   A single YAML configuration file.
*   A directory containing multiple YAML configuration files (the script will find all `.yaml` and `.yml` files within it).

**Examples:**

*   **Run all generated configs:**
    ```bash
    ./scripts/run_benchmarks.py benchmark_configs/
    ```

*   **Run a specific set of configs:**
    ```bash
    ./scripts/run_benchmarks.py benchmark_configs/erato_quick.yaml benchmark_configs/cornell_hd_high.yaml
    ```

*   **Run your custom config:**
    ```bash
    ./scripts/run_benchmarks.py my_benchmark_config.yaml
    ```

**Script Options:**

*   `--exec <path/to/rayon>`: Manually specify the path to the `rayon` executable if it's not found automatically (e.g., `./build/rayon`).
*   `--output <path/to/summary.csv>`: Specify the output path for the summary CSV file (default: `benchmark_results/summary.csv`).
*   `--name <folder_name>`: Create a specific subdirectory in `benchmark_results/` to store all outputs (images, JSONs, summary) for this run.

**Batch Run Output:**

The script will:
*   Print a summary table to the console, showing key metrics (Rays/Sec, Render Time, VRAM usage) for each run.
*   Save a comprehensive CSV file (`summary.csv`) in the `benchmark_results/` directory, containing all the collected metrics.

## Profiling Benchmarks (Nsight Tools)

To gain deeper insights into the performance bottlenecks of your CUDA code, you can leverage NVIDIA's Nsight profiling tools: Nsight Systems (`nsys`) for system-wide tracing, and Nsight Compute (`ncu`) for detailed kernel analysis.

These tools are *complementary* to the standard benchmarks. They produce rich graphical reports that can be viewed with the respective Nsight GUIs (available as part of the CUDA Toolkit).

**Important:** You cannot run Nsight Systems and Nsight Compute simultaneously on the same benchmark execution. They require exclusive access to hardware counters. If both flags are provided, `--profile-nsys` will take precedence.

### Using Nsight Systems (`--profile-nsys`)

Nsight Systems provides a timeline view of your application, showing CPU and GPU activity, CUDA API calls, memory transfers, and kernel execution. Use it to identify high-level bottlenecks, such as CPU-GPU synchronization issues, data transfer overheads, or overall GPU utilization.

**Command Syntax:**

```bash
./scripts/run_benchmarks.py <config_path> --profile-nsys [OPTIONS]
```

**Example:**

```bash
./scripts/run_benchmarks.py benchmark_configs/erato_hd_high.yaml --profile-nsys
```

**Output:**
An `.nsys-rep` report file will be generated in `benchmark_results/profiling/` (or `benchmark_results/<your_name>/profiling/` if `--name` is used). The filename will be based on the benchmark configuration. Open this file with the Nsight Systems GUI for visualization.

### Using Nsight Compute (`--profile-ncu`)

Nsight Compute provides detailed, low-level metrics for individual CUDA kernels. It helps you understand why a specific kernel might be performing poorly by showing warp divergence, memory access patterns, cache hit rates, instruction throughput, and more. When `--profile-ncu` is used, the script will profile only the *first* kernel launched.

**Command Syntax:**

```bash
./scripts/run_benchmarks.py <config_path> --profile-ncu [OPTIONS]
```

**Example:**

```bash
./scripts/run_benchmarks.py benchmark_configs/conf_room_2_hd_high.yaml --profile-ncu
```

**Output:**
An `.ncu-rep` report file will be generated in `benchmark_results/profiling/` (or `benchmark_results/<your_name>/profiling/` if `--name` is used). The filename will be based on the benchmark configuration.

### Analyzing Results

#### 1. Using the Nsight Compute GUI (Recommended)
The most effective way to analyze results is to open the `.ncu-rep` files in the Nsight Compute GUI (`ncu-ui`).
*   **Launch:** Run `ncu-ui` (or open "Nsight Compute" from your applications menu).
*   **Open:** Go to `File > Open File` and select your generated `.ncu-rep` file.
*   **Inspect:** Use the "Details" page to see "Speed of Light" throughput, warp stall reasons, and memory access analysis. The "Source" tab allows you to correlate performance metrics directly with your CUDA code lines.

#### 2. Using the Command Line (CLI)
You can print a text summary of the report directly in the terminal using the `--import` flag. This is useful for quick checks on headless servers.

**Command:**
```bash
ncu --import benchmark_results/profiling/<file>.ncu-rep
```

### Troubleshooting: Permission Denied (`ERR_NVGPUCTRPERM`)

If you encounter the error `ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters`, it means your user account is restricted from profiling.

**Permanent Fix (Recommended):**
Allow non-root users to access performance counters by creating a modprobe configuration file.

1.  Run the following command:
    ```bash
    echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee -a /etc/modprobe.d/nvidia-profiler.conf
    ```
2.  **Reboot your system** for the changes to take effect.

**Temporary Workaround:**
Run the benchmark script with `sudo` (root privileges). Note that output files will be owned by root.

```bash
sudo python3 ./scripts/run_benchmarks.py ...
```

---

## Creating Custom Benchmark Configurations

You can create your own `.yaml` files to define custom benchmark scenarios.

**Example `my_custom_benchmark.yaml`:**

```yaml
benchmark:
  scene_file: "resources/scenes/default_scene.yaml" # Path to the scene file to render
  output_name: "my_custom_benchmark_test"         # Name used for output files (e.g., PNG, JSON)
  target_samples: 256                             # Samples per pixel
  max_time_seconds: 60.0                          # Maximum render time in seconds (0.0 for no limit)
  resolution_width: 1280
  resolution_height: 720
```

Once created, you can run it using either the single launch command or include it in a batch run.