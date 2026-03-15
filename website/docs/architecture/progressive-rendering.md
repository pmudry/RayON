# Progressive Rendering

Interactive mode (renderer mode 3) is where RayON feels most alive. This page explains exactly
how the progressive accumulation pipeline keeps the display responsive while steadily improving
image quality.

---

## Overview

The progressive renderer operates in a loop:

1. User moves the camera → **motion detected** → reset accumulation, restart at low SPP.
2. Camera is still for 500 ms → **accumulation phase** → add more samples per stage.
3. Each stage: one GPU kernel pass, gamma-correct on GPU, copy uint8 to host, blit to SDL texture.

```mermaid
sequenceDiagram
    actor User
    participant SDL
    participant Host (C++)
    participant GPU Kernel

    User->>SDL: move mouse
    SDL-->>Host (C++): SDL_MOUSEMOTION event
    Host (C++)->>GPU Kernel: reset accum buffer (clear float array)
    loop accumulation stages
        Host (C++)->>GPU Kernel: renderPixelsKernelAccum(spp=8)
        GPU Kernel-->>Host (C++): uint8 display buffer (D2H)
        Host (C++)->>SDL: SDL_UpdateTexture + SDL_RenderCopy
        SDL-->>User: frame displayed
        Note over Host (C++): wait 500 ms if stationary
        Host (C++)->>GPU Kernel: renderPixelsKernelAccum(spp=+8)
    end
```

---

## Sample stages

The accumulative renderer progresses through predefined SPP levels:

```
8 → 16 → 32 → 64 → 128 → 256 → 512 → 1024 → 2048 → max_samples
```

After each stage completes, the renderer pauses **500 ms** before starting the next, checking
for user input. This keeps the UI responsive — a mouse drag will immediately interrupt the
accumulation and reset to 8 SPP.

During camera motion, only the **lowest SPP** (e.g., `--start-samples 8`) is used to maintain
frame rate. Once the camera is stationary, each subsequent frame adds another stage.

---

## Tiled rendering for responsiveness

The image is divided into an **8×8 grid of tiles** (64 tiles). Each tile is rendered
independently, so progress is visible as tiles complete:

```cpp
// Each tile is 1/8 of image width × 1/8 of image height
int tile_w = (image_width  + 7) / 8;
int tile_h = (image_height + 7) / 8;

for (int ty = 0; ty < 8 && !user_moved; ++ty) {
    for (int tx = 0; tx < 8 && !user_moved; ++tx) {
        renderTile(tx * tile_w, ty * tile_h, tile_w, tile_h, spp);
        displayIntermediate(); // show partial result
    }
}
```

If the user moves the camera while tiles are rendering, the inner loops break immediately
and the accumulation resets. This prevents the "frozen UI" problem of waiting for a
full-resolution render to complete.

---

## Float accumulation on GPU

The accumulation buffer is a **float array on the GPU** of size `width × height × 3`.
Each new sample pass adds to this buffer with atomic additions:

```cpp
// In the accumulative kernel
atomicAdd(&d_accum[pixel_idx * 3 + 0], pixel_color.x);
atomicAdd(&d_accum[pixel_idx * 3 + 1], pixel_color.y);
atomicAdd(&d_accum[pixel_idx * 3 + 2], pixel_color.z);
```

Atomics are needed because multiple samples per kernel launch write to the same pixel.

After each pass, a lightweight gamma kernel normalises and converts to uint8:

```cpp
float r = sqrtf(d_accum[i*3+0] / total_samples); // gamma 2.0 = sqrt
```

Only the 3-byte-per-pixel uint8 result is copied host ← device. The float buffer stays
on device throughout the session.

---

## Adaptive depth

With `--adaptive-depth`, `MAX_DEPTH` (maximum bounce count) starts at 4 and increases
by 1 after each completed sample stage. This reduces the initial compute cost and progressively
enables more complex light paths (caustics, multiple inter-reflections) as the image converges.

```
Stage 1 (8 SPP):   MAX_DEPTH = 4   ← fast, direct lighting only
Stage 2 (16 SPP):  MAX_DEPTH = 5
Stage 3 (32 SPP):  MAX_DEPTH = 6   ← first-order caustics readable
Stage 4 (64 SPP):  MAX_DEPTH = 7
Stage 5+:          MAX_DEPTH = 8   ← full-quality bouncing
```

---

## Camera interaction

The camera is controlled via SDL mouse events:

| Input | Action |
|---|---|
| Left mouse button + drag | **Orbit** — rotate around look-at point |
| Right mouse button + drag | **Pan** — translate look-at point laterally |
| Scroll wheel | **Zoom** — adjust distance from look-at point |
| `Space` | Force re-render from current accumulation state |
| `ESC` | Quit |

**Orbit implementation:**

```cpp
// sdl_gui_controls.hpp — simplified
void handle_orbit(float dx, float dy) {
    spherical_theta += dy * sensitivity;
    spherical_phi   += dx * sensitivity;
    spherical_theta = clamp(spherical_theta, 0.01f, PI - 0.01f);

    // Convert spherical → Cartesian offset from look_at
    float r = distance;
    look_from.x = look_at.x + r * sin(spherical_theta) * cos(spherical_phi);
    look_from.y = look_at.y + r * cos(spherical_theta);
    look_from.z = look_at.z + r * sin(spherical_theta) * sin(spherical_phi);
}
```

---

## Dear ImGui controls

The interactive window overlays a Dear ImGui panel with live controls:

| Slider | Affects |
|---|---|
| Samples per pixel | Base SPP for current accumulation pass |
| Max samples | Upper bound for auto-accumulate |
| Light intensity | Scales area light emission (via `cudaMemcpyToSymbol`) |
| Roughness | Material roughness of selected sphere |
| Aperture / Focus | DOF lens model parameters |
| Max depth | Ray bounce limit |

Changes to any slider trigger `needs_rerender = true`, which resets the accumulation buffer and
restarts from 8 SPP.

<img class="render-img" src="../../assets/images/real_time_raytrace.png"
     alt="Interactive SDL2 window with Dear ImGui overlay showing real-time path traced scene">

*The interactive window at ~100 Hz on the DGX Spark. The ImGui panel is overlaid on the right.*

---

## Live parameter updates via `cudaMemcpyToSymbol`

Some parameters (like light intensity) are stored as __device__ constants and updated
without re-creating the scene:

```cpp
// Host side
float h_light_intensity = 5.0f;
cudaMemcpyToSymbol(d_light_intensity, &h_light_intensity, sizeof(float));

// Device side  (__device__ in shader_common.cuh)
__device__ float d_light_intensity;
// ... used directly in the emission evaluation
```

This avoids a full GPU scene rebuild on every slider change.
