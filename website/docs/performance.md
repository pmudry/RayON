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
| CUDA block size | 32 × 4 (128 threads) |
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
| CPU single-thread | ~1 800 | 1× (baseline) |
| CPU multi-thread (16 cores) | ~120 | ~15× |
| CUDA GPU — no BVH | ~4.5 | ~400× |
| CUDA GPU — with BVH | ~1.7 | ~1 060× |

!!! note "CPU times above are estimates"
    The CPU renderer was not benchmarked for 256 SPP at 720p in the automated suite
    (it would take 30+ minutes). Values above extrapolate from lower-resolution timing
    runs scaled linearly with pixel count.

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

## Running the benchmark yourself

```bash
cd /path/to/RayON
bash bench.sh
# Results appended to bench_results.csv
```

The script launches three sequential renders of the default scene, records GPU, resolution, samples,
commit hash, branch, and wall-clock render time to `bench_results.csv`:

```csv
timestamp,commit,branch,gpu,resolution,samples,run,time_s
2026-03-12T21:39:52+01:00,ce8002d-dirty,main,NVIDIA GB10,720,1024,1,1.651
2026-03-12T21:39:52+01:00,ce8002d-dirty,main,NVIDIA GB10,720,1024,2,1.787
2026-03-12T21:39:52+01:00,ce8002d-dirty,main,NVIDIA GB10,720,1024,3,1.652
```

---

## Tuning tips

- **Enable BVH** for any scene with more than ~15 objects: `use_bvh: true` in YAML.
- **Start with low SPP** in interactive mode (`--start-samples 8`) for responsive orbit/pan.
- **Adaptive depth** (`--adaptive-depth`) starts at 4 bounces and increments after each sample
  stage — gives a good balance between responsiveness and physically accurate caustics.
- **Block size** — the default 32×4 is tuned for warp alignment. Changing it often regresses performance.
- **Fast math** — `--use_fast_math` is currently disabled by default because a small number of
  scenes show artefacts in glass materials. Enable only if you do not have refractive surfaces.
