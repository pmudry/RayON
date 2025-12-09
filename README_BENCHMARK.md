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
