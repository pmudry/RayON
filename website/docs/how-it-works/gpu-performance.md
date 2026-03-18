# GPU Performance Techniques

This page documents the GPU-specific performance optimisation techniques implemented in
both the **CUDA** and **OptiX** renderers. These complement the general optimisations
described in [Performance Optimisations](optimizations.md) and focus on reducing latency,
maximizing GPU throughput, and minimizing unnecessary host↔device data transfer.

---

## Overview

The two GPU rendering backends in RayON — **custom CUDA** and **OptiX** (hardware RT cores) —
share many of the same performance bottlenecks:

| Bottleneck | Where it hurts |
|---|---|
| Redundant computation inside hot loops | BVH traversal, ray–AABB intersection |
| Full-device synchronization (`cudaDeviceSynchronize`) | Blocks CPU + all GPU streams |
| Host-side pixel counting and conversion | D2H transfer of full-resolution buffers |
| Synchronous memory copies | Prevents overlap of GPU compute and data transfer |

The techniques below address each of these.

---

## 1 — Precomputed inverse ray direction for BVH traversal

**Problem:** During BVH traversal, every ray is tested against dozens of axis-aligned bounding
boxes (AABBs). The standard slab-method test computes `1/dir.x`, `1/dir.y`, `1/dir.z` — three
reciprocal divisions — for *every* AABB test. Since the ray direction doesn't change during
traversal, these divisions are redundant.

**Solution:** Precompute the inverse ray direction once per ray at the start of `hit_scene()` and
pass it as a parameter to `hit_aabb()`:

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

**Impact:** Eliminates 3 `fdiv` instructions per AABB test. For a BVH with depth *d*, each ray
saves up to *3d* divisions. On a scene with 300+ objects and BVH depth ≈ 12, this saves ~36
divisions per ray, per bounce.

**Files:** `cuda_raytracer.cuh` — `hit_aabb()`, `hit_scene()`

---

## 2 — `__launch_bounds__` on the path tracing kernel

**Problem:** The CUDA compiler (`nvcc`) must decide how many registers to allocate per thread.
Without explicit guidance, it optimises for a generic thread count, which may result in register
spilling (using slow local memory) or poor occupancy.

**Solution:** The `__launch_bounds__(256)` annotation tells the compiler that this kernel will
always be launched with at most 256 threads per block (our 32 × 8 configuration):

```cuda
__global__ void __launch_bounds__(256)
renderAccKernel(float4 *accum_buffer, ...)
{
   // ... path tracing logic ...
}
```

**Why 256?** The path tracing kernel is register-heavy (ray state, hit records, material data,
RNG state). With 256 threads, the compiler can allocate up to 64 registers per thread on modern
GPUs (65,536 registers per SM ÷ 256 threads = 256 max, but the compiler can keep more resident
warps). This gives better occupancy than if the compiler had to assume a higher thread count.

**Impact:** ~5–10% throughput improvement from better register allocation decisions. The compiler
avoids unnecessary register spills and can keep more state in fast register file.

**Files:** `shaders/render_acc_kernel.cu`, `shaders/render_acc_kernel.cuh`

---

## 3 — GPU-side converged pixel counting (warp-shuffle reduction)

**Problem:** Adaptive sampling tracks per-pixel convergence by storing a negative sample count
for converged pixels. To display the convergence percentage, the original code copied the entire
pixel-count buffer (4 bytes × width × height ≈ 3.5 MB at 720p) from device to host, then
iterated over every pixel on the CPU:

```cpp
// OLD: ~3.5 MB D2H transfer + O(n) CPU loop per frame
std::vector<int> host_counts(num_pixels);
cudaMemcpy(host_counts.data(), d_pixel_sample_counts, ...);
for (int i = 0; i < num_pixels; ++i)
   if (host_counts[i] < 0) ++converged;
```

**Solution:** A single-pass GPU reduction kernel using warp-shuffle instructions. Each thread
checks one pixel, then a warp-level reduction combines 32 results without shared memory.
Lane 0 of each warp atomically adds to a global counter:

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

**Impact:** Eliminates the per-frame 3.5 MB D2H transfer. The GPU kernel runs in < 0.1 ms,
and only a single `int` (4 bytes) is copied back to the host.

**Files:** `shaders/render_acc_kernel.cu`, `renderer_cuda_device.cu`

---

## 4 — Dedicated CUDA streams for render and display pipelines

**Problem:** Using `cudaDeviceSynchronize()` after the render kernel forces the GPU to complete
*all* work before the CPU regains control. This prevents overlapping the render kernel with
the display conversion pipeline (gamma correction + D2H copy).

**Solution:** Two dedicated non-blocking CUDA streams:

| Stream | Purpose | Work items |
|---|---|---|
| `s_compute_stream` | Path tracing | `renderAccKernel` launch + sync |
| `s_display_stream` | Display pipeline | `gammaCorrectKernel` → async D2H copy → `memcpy` |

```cuda
// Render kernel on compute stream
renderAccKernel<<<blocks, threads, 0, s_compute_stream>>>(...);
cudaStreamSynchronize(s_compute_stream);  // Only waits for this stream

// Display kernel on separate stream (can overlap with next render batch)
gammaCorrectKernel<<<blocks, threads, 0, s_display_stream>>>(...);
cudaMemcpyAsync(pinned_buf, d_display, size, cudaMemcpyDeviceToHost, s_display_stream);
cudaStreamSynchronize(s_display_stream);
```

**Critical requirement — buffer resets must use the same stream:**
Because `s_compute_stream` is `cudaStreamNonBlocking`, it has *no implicit ordering relationship*
with the default stream (stream 0). Calling `cudaMemset(...)` (which uses stream 0) to zero the
accumulation buffer or the adaptive sample-count buffer before the next render kernel therefore
creates a race condition: the kernel can start reading the buffer while the memset is still
running. The fix is to use `cudaMemsetAsync` on the same non-blocking stream:

```cuda
// WRONG: cudaMemset uses stream 0 — races with compute_stream kernel
cudaMemset(d_accum_buffer, 0, size);

// CORRECT: ordered before the next renderAccKernel on the same stream
cudaMemsetAsync(d_accum_buffer, 0, size, s_compute_stream);
```

This applies to both `resetDeviceAccumBuffer` and `resetAdaptiveBuffer`. Without it, stale
accumulation values combined with a zeroed-but-unread sample-count buffer caused pixels to be
divided by 1 instead of the true sample count, producing a white frame on the first rendered
batch after a camera move (visible whenever adaptive sampling was enabled).

**Impact:** In the interactive progressive renderer, the display pipeline (~0.5–2 ms) can
overlap with the start of the next render batch, reducing per-frame latency. The stream-ordering
fix also eliminates the white-frame artifact on camera movement with adaptive sampling.

**Files:** `renderer_cuda_device.cu`

---

## 5 — OptiX: Dedicated render stream

**Problem:** The OptiX renderer used `optixLaunch(..., 0, ...)` (default stream) followed by
`cudaDeviceSynchronize()`. This blocked all GPU work including any concurrent display operations.

**Solution:** A dedicated `render_stream` in the `OptixState` structure. The OptiX launch and
parameter upload use this stream, and synchronization is stream-specific:

```cuda
// Async param upload + launch on dedicated stream
cudaMemcpyAsync(d_launch_params, &params, sizeof(params), H2D, render_stream);
optixLaunch(pipeline, render_stream, d_launch_params, sizeof(params), &sbt, w, h, 1);
cudaStreamSynchronize(render_stream);  // Only waits for OptiX work
```

**Critical requirement — accumulation reset must use the same stream:**
The `render_stream` is `cudaStreamNonBlocking`, so it has no ordering relationship with stream 0.
If `optixRendererResetAccum` calls `cudaMemset(...)` (stream 0) to zero the accumulation buffer,
the next `optixLaunch` on `render_stream` can run *concurrently* with the memset — meaning the
launch can write pixels that the late-arriving memset then overwrites with zeros, producing black
streaks on camera movement. The fix:

```cuda
// WRONG: cudaMemset uses stream 0 — races with optixLaunch on render_stream
cudaMemset(g_state.d_accum_buffer, 0, size);

// CORRECT: ordered before the next optixLaunch on the same stream
cudaMemsetAsync(g_state.d_accum_buffer, 0, size, getOptiXStream());
```

**Impact:** Enables future overlap between OptiX rendering and display conversion, avoids
blocking unrelated GPU work on other streams, and eliminates black-streak artifacts on camera
movement caused by the stream 0 vs. non-blocking stream race.

**Files:** `optix/optix_renderer.cu`

---

## 6 — OptiX: GPU-side gamma correction with pinned memory

**Problem:** The original OptiX pipeline downloaded the full float4 accumulation buffer to the
host (4 × 4 bytes × pixels ≈ 14 MB at 720p), then performed gamma correction and format
conversion on the CPU:

```
GPU (float4 accum) → cudaMemcpy D2H (14 MB) → CPU gamma + float→uint8 → display
```

**Solution:** A GPU gamma correction kernel that converts float4 → uint8 directly on the device,
followed by an async D2H copy of only the small display buffer (3 bytes × pixels ≈ 2.7 MB at
720p) via pinned host memory:

```
GPU (float4 accum) → gammaCorrectKernel → uint8 d_display
                                            ↓
                                    cudaMemcpyAsync (2.7 MB, pinned)
                                            ↓
                                        Host display buffer
```

This is the same pipeline architecture used by the CUDA progressive renderer, now ported to
OptiX. The persistent pinned memory and device display buffers are managed as part of
`OptixState` and survive across frames.

**Impact:**

- **5× smaller D2H transfer:** 2.7 MB (uint8 RGB) vs. 14 MB (float4 RGBA)
- **Eliminates CPU gamma correction:** the GPU does it in parallel across all pixels
- **Async transfer:** pinned memory enables DMA-based copy without CPU involvement

**Files:** `optix/optix_renderer.cu`, `renderer_optix_host.hpp`, `renderer_optix_progressive_host.hpp`

---

## 7 — Firefly rejection (per-sample luminance clamp)

**Problem:** HDR environment maps contain extreme luminance values — the sun disk in an outdoor
sky image (`sunflowers_puresky_4k`, `rosendal_plains_2_4k`, …) can reach 50,000+ in linear
light. When a path happens to hit one of those texels (especially in the first few samples after
a camera move or scene reset), the single-sample contribution dominates the accumulation buffer.
With only 1–4 accumulated samples the gamma kernel divides by a small N, leaving the pixel
white. The artifact resolves as more samples accumulate, but it is very visible as flickering
white dots during motion.

**Solution:** Apply a **luminance-preserving clamp** to each sample's contribution immediately
before it is added to the accumulation buffer. The hue of the sample is preserved by scaling
all three channels by the same factor:

```cuda
// In renderAccKernel (CUDA) and __raygen__rg (OptiX):
constexpr float FIREFLY_CLAMP = 20.0f;
float sample_lum = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
if (sample_lum > FIREFLY_CLAMP)
    color = color * (FIREFLY_CLAMP / sample_lum);  // scale, don't clip per-channel
```

The threshold of **20.0** (linear) covers the full visible sky (clear blue ≈ 3–8, bright clouds
≈ 10–15) while rejecting only the extreme sun disk. Production renderers such as Blender Cycles
expose equivalent settings ("Clamp Direct / Indirect").

**Why luminance-preserving rather than per-channel min?** Per-channel clamping (`fminf(r, C)`)
shifts hue — a nearly-white sun pixel that is already balanced stays balanced with a
luminance scale, whereas clamping R/G/B independently can introduce a color cast.

**Files:** `shaders/render_acc_kernel.cu`, `optix/optix_programs.cu`

---

## Summary of techniques

| # | Technique | Renderer | Bottleneck addressed | Estimated impact |
|---|---|---|---|---|
| 1 | Precomputed inverse ray direction | CUDA | Redundant arithmetic in BVH traversal | 5–15% for BVH scenes |
| 2 | `__launch_bounds__(256)` | CUDA | Register allocation / occupancy | ~5–10% |
| 3 | Warp-shuffle converged counting | CUDA | 3.5 MB D2H per frame | < 0.1 ms vs. ~1–2 ms |
| 4 | Dual CUDA streams | CUDA | Full-device synchronization | ~0.5–2 ms/frame latency reduction |
| 4a | `cudaMemsetAsync` on compute stream | CUDA | Stream 0 race with non-blocking stream | Fixes white-frame on camera move (adaptive sampling) |
| 5 | OptiX render stream | OptiX | Full-device synchronization | Enables async overlap |
| 5a | `cudaMemsetAsync` on render stream | OptiX | Stream 0 race with non-blocking stream | Fixes black-streak artifacts on camera move |
| 6 | GPU gamma + pinned memory | OptiX | 14 MB D2H + CPU conversion | 5× bandwidth reduction |
| 7 | Per-sample firefly clamp | CUDA + OptiX | Extreme HDR texel contributions | Eliminates white-dot artifacts with HDR env maps |

Techniques 1–6 are **backward-compatible** — they do not change the rendered output, only the
speed at which it is produced. Technique 7 introduces a slight bias (under-representing
extreme-luminance regions such as the sun disk) in exchange for artifact-free interactive
rendering.
