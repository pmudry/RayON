# Performance Optimisations

This page documents every significant optimisation that was applied to RayON, roughly in the
order they were introduced, together with the measured or estimated impact of each one.
The cumulative effect is a **~1 060× speedup** over the single-threaded CPU baseline for a
typical 720p scene with 1 024 SPP.

---

## 1 — CPU multi-threading

!!! info "CPU renderers archived"
    The CPU rendering backends (sequential and multi-threaded) have been moved to the
    [`legacy/cpu-renderer`](https://github.com/pmudry/RayON/tree/legacy/cpu-renderer) branch.
    The main branch now supports GPU rendering only.

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

## 11 — Non-blocking CUDA stream + pinned-memory D2H pipeline

**What it is:** the display path (gamma-correction kernel + device→host copy) runs on a
dedicated **non-blocking CUDA stream** (`cudaStreamNonBlocking`). The host memory target is
**page-locked (pinned)**, allocated with `cudaMallocHost`.

**Why it helps:** the old design used the default CUDA stream and `cudaDeviceSynchronize()`.
`cudaDeviceSynchronize()` is a *global* barrier — it drains every outstanding GPU operation
before returning. In interactive mode this means the next render batch cannot start until the
display pipeline has completely finished, stalling the CPU and the GPU at the same time.

The new design:

1. Creates a **separate, non-blocking stream** (`s_display_stream`) for display work. Non-blocking
   means it will never implicitly synchronize with the default stream used by the render kernel.
2. Queues both the gamma-correction kernel and the `cudaMemcpyAsync` on that stream so the GPU
   processes them in order with no CPU involvement between the two.
3. Uses a **pinned host buffer** (`cudaMallocHost`) as the DMA target. Pinned memory has a fixed
   physical address the GPU's DMA engine can write to directly over PCIe without an extra kernel-
   initiated copy — DMA throughput is typically 2–4× higher than to pageable memory.
4. Synchronizes *only* the display stream (`cudaStreamSynchronize(s_display_stream)`) rather than
   every GPU activity.

```cpp
// renderer_cuda_device.cu — display stream setup
cudaStreamCreateWithFlags(&s_display_stream, cudaStreamNonBlocking);
cudaMallocHost(&s_pinned_display, display_size);   // pinned staging buffer

// Per-frame display update — kernel + async DMA on same stream
cudaStream_t stream = s_display_stream;
gammaCorrectKernel<<<blocks, threads, 0, stream>>>(...);
cudaMemcpyAsync(s_pinned_display, d_display, display_size,
                cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);  // wait only for this stream
memcpy(display_image, s_pinned_display, display_size);  // fast pinned→pageable
```

**Measured impact:** removes the `cudaDeviceSynchronize()` bubble between consecutive render
batches in interactive mode. On the DGX Spark the display path dropped from ~3 ms (blocked) to
~0.8 ms (async), allowing the render kernel to start sooner each frame.

---

## 12 — Adaptive depth

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

## GPU Implementation Techniques

The sections above cover algorithmic and system-level decisions. This section documents the
lower-level CUDA and OptiX implementation details that address four recurring hardware
bottlenecks:

| Bottleneck | Where it hurts |
|---|---|
| Redundant arithmetic in hot loops | BVH traversal, ray–AABB intersection |
| Full-device barriers (`cudaDeviceSynchronize`) | Stalls CPU + all GPU streams |
| Large D2H transfers of per-pixel buffers | PCIe bandwidth waste |
| Default-stream race conditions | Correctness issues with non-blocking streams |

---

## 13 — Precomputed inverse ray direction for BVH traversal

**What it is:** the slab-method AABB test computes `1/dir.x`, `1/dir.y`, `1/dir.z` for every
bounding-box test during BVH traversal. Since the ray direction is constant across the entire
traversal, these three reciprocal divisions are redundant. The inverse is precomputed once
per ray and passed as a parameter to `hit_aabb()`.

```cuda
// hit_scene() — computed once per ray
const f3 inv_dir(1.0f / r.dir.x, 1.0f / r.dir.y, 1.0f / r.dir.z);

// hit_aabb() — uses precomputed inverse, no divisions
__device__ __forceinline__ bool hit_aabb(
    const ray_simple &r, const f3 &inv_dir,
    const f3 &box_min, const f3 &box_max,
    float t_min, float t_max)
{
    float t0_x = (box_min.x - r.orig.x) * inv_dir.x;  // multiply, not divide
    // ...
}
```

**Measured impact:** eliminates 3 `fdiv` instructions per AABB test. For a BVH of depth 12
with 300+ objects, each ray saves ~36 divisions per bounce.

**Files:** `cuda_raytracer.cuh` — `hit_aabb()`, `hit_scene()`

---

## 14 — `__launch_bounds__` on the path-tracing kernel

**What it is:** the `__launch_bounds__(256)` annotation tells the CUDA compiler that the
path-tracing kernel is always launched with at most 256 threads per block (our 32 × 8
configuration). Without it the compiler must assume a generic thread count and may
over-allocate registers or spill to slow local memory.

```cuda
__global__ void __launch_bounds__(256)
renderAccKernel(float4 *accum_buffer, ...)
{
    // ... path tracing logic ...
}
```

**Why 256?** The kernel is register-heavy (ray state, hit records, material data, RNG state).
With 256 threads per block, the compiler can allocate up to 256 registers per thread on modern
GPUs without spilling — giving better occupancy than if it had to assume a higher thread count.

**Measured impact:** ~5–10% throughput improvement from better register allocation; avoids
spills to slow local memory.

**Files:** `shaders/render_acc_kernel.cu`, `shaders/render_acc_kernel.cuh`

---

## 15 — GPU-side converged pixel counting (warp-shuffle reduction)

**What it is:** adaptive sampling tracks per-pixel convergence via a device-side `int` array
(negative values mark converged pixels). The original code copied the entire array (~3.5 MB
at 720p) to the host, then counted on the CPU. A replacement single-pass GPU reduction kernel
uses warp-shuffle instructions; only one `int` (4 bytes) is transferred back to the host.

```cuda
__global__ void countConvergedKernel(
    const int *pixel_sample_counts, int num_pixels, int *d_converged_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int converged = (idx < num_pixels && pixel_sample_counts[idx] < 0) ? 1 : 0;

    // Warp-level reduction — no shared memory needed
    for (int offset = 16; offset > 0; offset >>= 1)
        converged += __shfl_down_sync(0xFFFFFFFF, converged, offset);

    if ((threadIdx.x & 31) == 0)
        atomicAdd(d_converged_count, converged);
}
```

**Measured impact:** eliminates the per-frame 3.5 MB D2H transfer. The GPU kernel runs in
< 0.1 ms; the old copy + CPU loop took ~1–2 ms per frame.

**Files:** `shaders/render_acc_kernel.cu`, `renderer_cuda_device.cu`

---

## 16 — Accumulation buffer reset ordering

**What it is:** when a camera move invalidates accumulated samples, the accumulation buffer
must be zeroed before the next render kernel reads it. Using `cudaMemset()` (which enqueues on
the **default stream 0**) while the render kernel runs on a **non-blocking custom stream**
creates a race condition: the kernel can start reading the buffer while the memset is still
running.

The fix is to use `cudaMemsetAsync` on the *same* non-blocking stream as the kernel. This
race affected both backends:

```cuda
// WRONG — stream 0 races with s_compute_stream / render_stream
cudaMemset(d_accum_buffer, 0, size);

// CORRECT — guaranteed to complete before the next kernel launch on the same stream
cudaMemsetAsync(d_accum_buffer, 0, size, s_compute_stream);   // CUDA
cudaMemsetAsync(g_state.d_accum_buffer, 0, size, getOptiXStream());  // OptiX
```

**Measured impact:** eliminates a white-frame artifact in the CUDA renderer (visible on every
camera move when adaptive sampling was enabled) and black-streak artifacts in the OptiX
renderer. No throughput change — this is a correctness fix.

**Files:** `renderer_cuda_device.cu`, `optix/optix_renderer.cu`

---

## 17 — OptiX: GPU-side gamma correction with pinned memory

**What it is:** the original OptiX pipeline downloaded the full `float4` accumulation buffer
to the host (~14 MB at 720p) before performing gamma correction and format conversion on the
CPU. A GPU gamma-correction kernel now converts `float4 → uint8` directly on the device;
only the compact display buffer (~2.7 MB) is transferred via an async copy to pinned host
memory.

```
GPU (float4 accum) → gammaCorrectKernel → uint8 d_display
                                            ↓
                                    cudaMemcpyAsync (2.7 MB, pinned)
                                            ↓
                                        Host display buffer
```

**Measured impact:**

- **5× smaller D2H transfer:** 2.7 MB (uint8 RGB) vs. 14 MB (float4 RGBA)
- **GPU-parallel gamma correction:** no CPU involvement between render and display
- **Async DMA transfer:** pinned memory allows the GPU's DMA engine to write directly over PCIe

**Files:** `optix/optix_renderer.cu`, `renderer_optix_host.hpp`, `renderer_optix_progressive_host.hpp`

---

## 18 — Firefly rejection (per-sample luminance clamp)

**What it is:** HDR environment map texels (e.g. the sun disk in an outdoor sky image) can
have linear luminance > 50 000. A single such sample early in accumulation snaps the pixel to
white and takes many subsequent samples to average down. A luminance-preserving clamp caps
each sample's contribution before it is added to the accumulation buffer; hue is preserved by
scaling all three channels uniformly.

```cuda
// In renderAccKernel (CUDA) and __raygen__rg (OptiX):
constexpr float FIREFLY_CLAMP = 20.0f;
float sample_lum = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
if (sample_lum > FIREFLY_CLAMP)
    color = color * (FIREFLY_CLAMP / sample_lum);  // scale, don't clip per-channel
```

The threshold of 20.0 (linear) covers the full visible sky (3–15) while rejecting only the
extreme sun-disk texels. Per-channel clamping (`fminf(r, C)`) is avoided because it shifts
hue — a luminance scale keeps the colour balanced.

**Measured impact:** eliminates white-dot flickering during camera motion with HDR environment
maps. Introduces a slight bias in extremely bright regions — the standard trade-off in
production renderers (Blender Cycles exposes equivalent "Clamp Direct / Indirect" settings).

**Files:** `shaders/render_acc_kernel.cu`, `optix/optix_programs.cu`

---

## Summary

| # | Optimisation | Impact | Renderer |
|---|---|---|---|
| 1 | CPU multi-threading | ~15× | CPU *(archived — see `legacy/cpu-renderer` branch)* |
| 2 | CUDA GPU kernels | ~400× vs. CPU ST | CUDA |
| 3 | 32×4 thread blocks | ~5–10% throughput | CUDA |
| 4 | Cosine-weighted sampling | 4–8× fewer SPP | All |
| 5 | Russian roulette termination | ~15–20% throughput | All |
| 6 | Persistent curand states | −46 ms/frame overhead | CUDA |
| 7 | GPU accumulation + uint8 D2H | 4× lower PCIe bandwidth | CUDA |
| 8 | BVH with SAH | up to 14.6× on 300+ objects | All |
| 9 | Inlined material dispatch | ~5–10% throughput | CUDA |
| 10 | Adaptive sampling | 20–50% on mixed scenes | CUDA |
| 11 | Non-blocking stream + pinned memory | −2 ms/frame display latency | CUDA |
| 12 | Adaptive depth | Subjective responsiveness | CUDA |
| 13 | Precomputed inverse ray direction | 5–15% for BVH scenes | CUDA |
| 14 | `__launch_bounds__(256)` | ~5–10% throughput | CUDA |
| 15 | Warp-shuffle converged counting | <0.1 ms vs. ~1–2 ms D2H | CUDA |
| 16 | Accumulation reset stream ordering | Eliminates white-frame/black-streak artifacts | CUDA + OptiX |
| 17 | OptiX GPU gamma + pinned memory | 5× bandwidth reduction (14 MB → 2.7 MB) | OptiX |
| 18 | Firefly rejection | Eliminates HDR white-dot flickering | CUDA + OptiX |

The **combined CUDA + BVH** speedup reaches **~1 660×** over single-threaded CPU on the
default scene at 720p, 1 024 SPP — measured on an NVIDIA DGX Spark (GB10 GPU).

Techniques 1–16 are **backward-compatible** — they do not change the rendered output. Technique
17 and 18 introduce minor biases (OptiX gamma rounding; HDR luminance clamping) that are
invisible at normal viewing conditions but eliminate distracting artifacts.
