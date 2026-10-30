# Performance

Benchmark results and speedup analysis for RayON on the primary development
machine (NVIDIA DGX Spark, 2026).

---

## Test configuration

| Parameter | Value |
|---|---|
| GPU | NVIDIA GB10 (DGX Spark) |
| Resolution | 1280 × 720 (720p) |
| Samples per pixel | 1024 |
| Runs per benchmark | 3 (averaged) |
| CUDA block size | 32 × 8 (256 threads) |
| BVH | Enabled (SAH) |

---

## Latest benchmark results

Render times (seconds) for 720p at 1024 SPP across recent commits on `main`:

| Commit | Average (s) | Min (s) | Max (s) | Notes |
|---|---|---|---|---|
| `433000f` | 1.82 | 1.77 | 1.93 | — |
| `8381ca6` | 1.99 | 1.92 | 2.05 | — |
| `155b8cd` | 4.52 | 4.31 | 4.66 | Regression — BVH traversal bug, since fixed |
| `ce8002d` | 1.72 | 1.65 | 1.82 | BVH fix landed; best recorded time |

!!! info "Regression and recovery"
    Commit `155b8cd` shows a ~2.5× slowdown caused by a BVH traversal regression.
    This was caught by the automated benchmark in `bench.sh` and fixed in `ce8002d`.
    The benchmark is designed exactly for this — catching unexpected performance drops.

---

## Renderer comparison

Measured on a single test scene (default scene, 720p, 256 SPP):

| Renderer | Time (s) | Speedup vs CPU |
|---|---|---|
| CPU single-thread *(archived)* | ~1 800 | 1× (baseline) |
| CPU multi-thread *(archived)* | ~120 | ~15× |
| CUDA GPU — no BVH | ~4.5 | ~400× |
| CUDA GPU — with BVH | ~1.7 | ~1 060× |

!!! info "CPU renderers archived"
    The CPU rendering backends (sequential and multi-threaded) have been moved to the
    [`legacy/cpu-renderer`](https://github.com/pmudry/RayON/tree/legacy/cpu-renderer) branch.
    The main branch now supports GPU rendering only. CPU times above are kept for historical
    reference as the baseline for speedup calculations.

---

## BVH acceleration

The BVH (Bounding Volume Hierarchy) provides the largest single speedup for scenes with many objects.
Performance measured across three scene sizes at 128 SPP, 720p:

| Scene | Objects | No BVH (ms) | With BVH (ms) | Speedup |
|---|---|---|---|---|
| `simple_scene.yaml` (3 spheres) | 3 | 110 | 120 | 0.9× (overhead) |
| `default_scene.yaml` | 20 | 380 | 280 | 1.4× |
| `09_color_bleed_box.yaml` | 50 | 1 200 | 450 | 2.7× |
| `bvh_stress_courtyard.yaml` | 300+ | 12 000 | 820 | **14.6×** |

!!! tip "When to enable BVH"
    BVH adds a small constant overhead so it is not worth it for scenes with fewer than ~10 objects.
    Enable it automatically in YAML: `use_bvh: true`.

---

## Sampling efficiency

The graph below shows how image quality (measured as RMS noise relative to the 4096 SPP reference)
improves with sample count. The \(1/\sqrt{N}\) curve is the theoretical Monte Carlo convergence rate.

| Samples | Relative noise | Render time (720p CUDA) |
|---|---|---|
| 1 | 100 % | 0.002 s |
| 8 | 35 % | 0.015 s |
| 32 | 18 % | 0.060 s |
| 128 | 9 % | 0.24 s |
| 256 | 6 % | 0.48 s |
| 512 | 4 % | 0.96 s |
| 1 024 | 3 % | 1.72 s |
| 2 048 | 2 % | 3.44 s |

The \(\mathcal{O}(1/\sqrt{N})\) convergence is a fundamental property of Monte Carlo integration.
Doubling sample count reduces noise by a factor of \(\sqrt{2} \approx 1.41\), meaning halving visible
noise requires **quadrupling** the samples.

---

## Reference timings, October 2026

Offline OptiX renders at 720p on the DGX Spark, measured during the development of the volume
material (single run each, the GPU otherwise idle):

| Scene | Samples | Time | Notes |
|---|---|---|---|
| `17_cloud_bunny.yaml` | 512 | 11 s | procedural volume, baked field, empty-space skipping |
| `19_vdb_bunny.yaml` | 512 | 40 s | density grid, no empty-space skipping yet |
| `18_sdf_shapes.yaml` | 384 | 23 s | ten ray-marched shapes incl. a mandelbulb |
| `16_nee_small_light.yaml` (360p) | 64 | 0.17 s | five matte spheres, one bulb |

---

## Running the benchmark yourself

```bash
cd /path/to/RayON
bash scripts/benchmark.sh --method cuda --sampler pcg
# Results appended to bench_results.csv
```

The script launches three sequential renders of the default scene, records GPU, resolution, samples,
renderer method, sampler, commit hash, branch, and wall-clock render time to `bench_results.csv`.
Supported methods are `cuda` (`-m 2`) and `optix` (`-m 4`), and the default benchmark sampler is `pcg`.
You can override sampler with `--sampler sobol`.

```csv
timestamp,commit,branch,gpu,resolution,samples,method,sampler,run,time_s
2026-03-12T21:39:52+01:00,ce8002d-dirty,main,NVIDIA GB10,720,1024,cuda(2),pcg,1,1.651
2026-03-12T21:39:52+01:00,ce8002d-dirty,main,NVIDIA GB10,720,1024,cuda(2),pcg,2,1.787
2026-03-12T21:39:52+01:00,ce8002d-dirty,main,NVIDIA GB10,720,1024,cuda(2),pcg,3,1.652
```

---

## Tuning tips

- **Enable BVH** for any scene with more than ~15 objects: `use_bvh: true` in YAML.
- **Raise the Target FPS** in the GUI's "Camera Motion" section for a very responsive orbit/pan
  experience — the batch size and the render resolution auto-scale down to keep up.
- **Block size** — the default 32×8 is tuned for warp alignment; register-cap and block-shape
  experiments (see `ROADMAP.md`) gave no consistent gain.
- **Fast math** — `--use_fast_math` is on in Release builds and off in Debug builds.
- **Prefer OptiX** (`-m 4`, `-m 5`) on RTX-class hardware: triangles and shadow rays run on the
  RT cores, and the interactive mode has the motion regulator.
- **Keep an eye on the first run after a build**: OptiX compiles its module on the first launch
  (a few seconds, then cached on disk), so time the second run.
