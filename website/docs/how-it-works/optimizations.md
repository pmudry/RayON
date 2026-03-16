# Performance Optimisations

This page documents every significant optimisation that was applied to RayON, roughly in the
order they were introduced, together with the measured or estimated impact of each one.
The cumulative effect is a **~1 060× speedup** over the single-threaded CPU baseline for a
typical 720p scene with 1 024 SPP.

---

## 1 — CPU multi-threading

**What it is:** tile-based work dispatch using `std::async`. The image is divided into blocks;
each block is submitted as an independent `std::future` and picked up by a thread-pool of
`N−2` hardware threads.

**Why it helps:** path tracing is embarrassingly parallel — each pixel is independent. Saturating
all cores gives a near-linear speedup.

**Measured impact:** ~15× on a 16-core machine.

```cpp
// renderer_cpu_parallel.hpp — simplified
for (int ty = 0; ty < num_tiles_y; ++ty) {
    for (int tx = 0; tx < num_tiles_x; ++tx) {
        futures.push_back(std::async(std::launch::async, [=] {
            renderTile(tx, ty, tile_w, tile_h);
        }));
    }
}
for (auto& f : futures) f.get();
```

---

## 2 — CUDA GPU rendering

**What it is:** a CUDA kernel that assigns one thread per pixel. Each thread independently traces
rays, with no communication between threads.

**Why it helps:** a modern GPU has thousands of streaming processors. The GPU also hides memory
latency through warp switching — while one warp waits for a memory transaction, another warp
executes.

**Measured impact:** ~400× vs. single-threaded CPU on a 720p scene (without BVH).

---

## 3 — Thread-block shape: 32 × 4

**What it is:** the kernel is launched with 2-D thread blocks of 32 columns × 4 rows = 128 threads.

**Why it helps:** 32 threads exactly fill one CUDA **warp** — the unit of SIMD execution. With
32 threads per row, adjacent threads access adjacent pixel addresses, which coalesces into a
single memory transaction. A 16×8 block would split one row across two warps and introduce
cross-warp divergence on conditional branches.

**Measured impact:** ~5–10% throughput gain over 16×8 at the same occupancy.

```cpp
dim3 block_size(32, 4);   // 128 threads, one warp per row
dim3 grid_size(
    (width  + block_size.x - 1) / block_size.x,
    (height + block_size.y - 1) / block_size.y
);
renderPixelsKernel<<<grid_size, block_size>>>(...);
```

---

## 4 — Cosine-weighted hemisphere sampling

**What it is:** diffuse surfaces scatter rays with probability proportional to
\(\cos\theta / \pi\) rather than \(1 / 2\pi\) (uniform).

**Why it helps:** the Monte Carlo integrand for a Lambertian surface contains a \(\cos\theta\)
factor. When the PDF *matches* that factor, the weights become constant:

\[
\frac{(\rho/\pi) \cdot L_i \cdot \cos\theta}{\cos\theta / \pi} = \rho \cdot L_i
\]

Every sample contributes equally — variance drops dramatically near shadow boundaries and at
grazing angles.

**Measured impact:** 4–8× fewer samples needed for equivalent convergence vs. uniform hemisphere
sampling (scene-dependent).

---

## 5 — Russian roulette path termination

**What it is:** after each bounce, a path is terminated randomly with probability proportional to
\(1 - \max(\text{throughput})\). Surviving paths are compensated to maintain an unbiased estimate.

**Why it helps:** carrying a path to `MAX_DEPTH` bounces even when throughput is near zero (e.g.
after five rough-mirror reflections) wastes GPU cycles. Russian roulette cuts those paths early
while redistributing the saved compute to paths that still carry energy.

**Measured impact:** ~15–20% throughput improvement on typical scenes with `MAX_DEPTH=16`.

```cpp
// After each bounce in the GPU kernel
float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
if (curand_uniform(&rng) > p) break;  // terminate — unbiased
throughput /= p;                       // compensate
```

---

## 6 — Persistent curand RNG states

**What it is:** one `curandState` per pixel is allocated in device memory at startup and
**reused across all frames** in progressive mode.

**Why it helps:** calling `curand_init()` is expensive (~50 ns per thread). Re-initialising 720p
= 921 600 states every frame would cost ~46 ms — longer than the render itself at 60 fps.
Persistent states also avoid repeating the same random sequence, which would cause visible
banding in accumulated renders.

**Measured impact:** eliminates a ~46 ms per-frame overhead in interactive mode at 720p.

---

## 7 — Accumulation on GPU + uint8 D2H transfer

**What it is:** sample results are accumulated in a `float` buffer that stays on the GPU.
After each batch, a lightweight gamma-correction kernel converts the float sums to `uint8`.
Only the 3-byte-per-pixel `uint8` result is copied host ← device.

**Why it helps:** the previous design transferred the full 3×float accumulation buffer each
frame (12 bytes/pixel). The new design reduces D2H bandwidth by **4×**. At 1920×1080, that
drops the PCIe transfer from ~24 MB/frame to ~6 MB/frame — comfortably below 60 fps budget.

```cpp
// GPU kernel: accumulate in float4 (one atomic per pixel)
atomicAdd(&d_accum[pixel_idx * 3 + 0], pixel_color.x);

// GPU gamma kernel: convert + pack (no CPU involvement)
display[idx*3+0] = (uint8_t)(clamp(sqrtf(accum[idx*3+0]/spp), 0.f, 1.f) * 255.f);
```

---

## 8 — BVH acceleration (SAH)

**What it is:** a **Bounding Volume Hierarchy** built on the CPU with **Surface Area Heuristic**
(SAH) splitting. The flat node array is uploaded to the GPU once and traversed iteratively by
every kernel thread.

**Why it helps:** without BVH, every ray tests all \(N\) objects — \(O(N)\). With a SAH-BVH of
depth \(\log_2 N\), average traversal cost drops to \(O(\log N)\) with tight bounds.

**Measured impact:** up to **14.6×** on a 300-object scene. See [Performance](../performance.md)
for the full table.

Key implementation details:

- **8 split candidates per axis** (3 axes × 8 = 24 candidates per node)
- **64-byte aligned `BVHNode`** — one complete node fits in one L2 cache line
- **Iterative stack traversal** on the GPU (depth-32 local stack, no recursion)
- **Nearer child pushed last** — the near child is at the top of the stack and processed first,
  allowing the "farther than current best" early-out to skip more nodes

```cpp
struct alignas(64) BVHNode {
    float3 aabb_min, aabb_max;  // 24 bytes
    int    left_child;           //  4 bytes
    int    right_child;          //  4 bytes
    int    prim_start;           //  4 bytes
    int    prim_count;           //  4 bytes  (> 0 → leaf)
    // padding to 64 bytes
};
```

---

## 9 — Inlined material dispatch

**What it is:** GPU material evaluation uses a `switch` statement over a `MaterialType` enum
rather than virtual functions. The compiler inlines every case at build time.

**Why it helps:** virtual function calls on the GPU require two indirect memory accesses (vtable
pointer + vtable entry) and break warp coherence when threads in the same warp hit different
materials. An inlined `switch` eliminates both costs.

**Measured impact:** ~5–10% throughput improvement on mixed-material scenes.

---

## 10 — Adaptive sampling (converged-pixel skipping)

**What it is:** each pixel tracks a running sample count. After ≥ 32 accumulated samples, the
renderer checks whether the relative change in luminance between the previous batch and the new
batch is below a threshold (~10⁻⁴·⁵). Converged pixels are flagged (negative sample count) and
skipped in all subsequent batches.

**Why it helps:** in most scenes, large uniform regions (sky, flat walls) converge quickly while
complex areas (shadow boundaries, caustics) need many more samples. Skipping converged pixels
redirects the GPU to the pixels that still need work.

A **heatmap visualisation** (purple = few samples, yellow = many) can be toggled in the ImGui
panel to show where samples are being spent.

**Measured impact:** 20–50% effective speedup in mixed-complexity scenes; less useful in
uniformly complex scenes.

Disable with `--no-adaptive-sampling`.

---

## 11 — Adaptive depth

**What it is:** `MAX_DEPTH` (maximum ray-bounce count) starts at 4 in interactive mode and
increments by 1 after each completed sample stage (when `--adaptive-depth` is passed).

**Why it helps:** high bounce counts are needed for accurate caustics and multiple
inter-reflections, but they are expensive. Starting low keeps the first frames fast and
responsive; increasing depth only after the image has begun to converge avoids wasting GPU
cycles on deep paths before coarser lighting is established.

```
Stage 1 (first batch):  MAX_DEPTH = 4   ← fast, direct lighting
Stage 2:                MAX_DEPTH = 5
Stage 3:                MAX_DEPTH = 6   ← first-order caustics
Stage 4+:               MAX_DEPTH = 7–8 ← full quality
```

Enable with `--adaptive-depth`.

---

## Summary

| Optimisation | Measured gain | Renderer |
|---|---|---|
| CPU multi-threading | ~15× | CPU |
| CUDA GPU kernels | ~400× (vs CPU ST) | CUDA |
| 32×4 thread blocks | ~5–10% | CUDA |
| Cosine-weighted sampling | 4–8× fewer SPP | All |
| Russian roulette | ~15–20% throughput | All |
| Persistent curand states | −46 ms/frame overhead | CUDA progressive |
| GPU accum + uint8 D2H | 4× lower PCIe bandwidth | CUDA progressive |
| BVH (SAH) | up to 14.6× on 300+ objects | All |
| Inlined material dispatch | ~5–10% throughput | CUDA |
| Adaptive sampling | 20–50% on mixed scenes | CUDA progressive |
| Adaptive depth | Subjective responsiveness | CUDA progressive |

The **combined CUDA + BVH** speedup reaches **~1 060×** over single-threaded CPU on the
default scene at 720p, 1 024 SPP — measured on an NVIDIA DGX Spark (GB10 GPU).
