# RayON CLI Arguments Reference

This document provides a detailed reference for all command-line arguments available in the RayON renderer.

## Usage
```bash
./rayon [options]
```

## General Options

| Option | Description | Default |
| :--- | :--- | :--- |
| `-h`, `--help`, `/?` | Show the help message and exit. | - |
| `-m <mode>` | Set the rendering method. | `2` (CUDA GPU) |
| `-s <samples>` | Set the number of samples per pixel. | `64` (or `2000` in interactive) |
| `-r <height>` | Set vertical resolution (180, 360, 720, 1080, 2160). | `720` |
| `--scene <file>` | Load a scene from a YAML file. | Built-in default scene |

### Rendering Modes (`-m`)
*   `0`: CPU Sequential (Reference)
*   `1`: CPU Parallel (OpenMP)
*   `2`: CUDA GPU (Fast, Non-interactive)
*   `3`: CUDA GPU Interactive (Requires SDL2)

## Interactive Mode Options
These options apply when using `-m 3`.

| Option | Description | Default |
| :--- | :--- | :--- |
| `--start-samples <n>` | Initial samples per frame when camera moves. | `32` |
| `--target-fps <fps>` | Target frame rate for the interactive loop. | `60` |
| `--adaptive-depth` | Enable adaptive ray depth (increases depth as image converges). | `Off` |
| `--no-auto-accumulate` | Disable automatic sample accumulation when stationary. | `Off` |
| `--debug` | Enable debug overlay. | `Off` |

## Benchmark Options

| Option | Description |
| :--- | :--- |
| `--benchmark <file>` | Activate benchmark mode using the specified config YAML. |
| `--benchmark-out <dir>` | Specify the output directory for benchmark results. |

## Examples

**Standard GPU Render:**
```bash
./rayon --scene resources/scenes/benchmark/sponza.yaml -r 1080 -s 500
```

**Interactive Exploration:**
```bash
./rayon -m 3 --scene resources/scenes/benchmark/fireplace_room.yaml --target-fps 144
```

**Run a Benchmark:**
```bash
./rayon --benchmark benchmark_configs/cornell_hd.yaml
```
