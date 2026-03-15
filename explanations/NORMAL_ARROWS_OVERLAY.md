# CPU Normal Arrows Overlay

This document explains how the normal arrows overlay is implemented in interactive mode.

## Goal

Display surface normal directions as arrows on top of the interactive image, without changing the main CUDA renderer.

The overlay is:
- Computed on CPU (ray-hit + normal extraction)
- Drawn on CPU (line rasterization into the display image buffer)
- Available in both standard shading and show-normals visualization mode

## Where The Code Lives

- Main implementation: `src/rayon/gpu_renderers/renderer_cuda_progressive_host.hpp`
- UI controls: `src/rayon/camera/sdl/sdl_gui_handler.hpp`

## High-Level Pipeline

Each interactive frame:

1. CUDA renders/updates `display_image` (RGB8) from the accumulation buffer.
2. CPU overlay pass optionally draws arrows into the same `display_image`.
3. SDL/ImGui displays the final composited image.

This means arrows are a post-process overlay and do not affect path tracing itself.

## Core Data Used For Arrows

The overlay reuses camera frame data and a CPU scene:

- Camera basis and pixel rays from `CameraFrame`:
  - `frame.pixel00_loc`
  - `frame.pixel_delta_u`
  - `frame.pixel_delta_v`
  - `frame.u`, `frame.v`
- CPU hit geometry from:
  - `Hittable_list cpu_scene_for_arrows = Scene::CPUSceneBuilder::buildCPUScene(original_scene);`

Why a CPU scene?
- We need normals even in standard shading mode.
- Reading per-pixel normals back from CUDA would require extra GPU buffers + transfer.
- CPU hit testing on sparse samples is simpler and fast enough for a debug overlay.

## Arrow Sampling Strategy

The UI exposes a target count (`normal_arrow_count`).

At runtime the code computes a grid step:

- `target_density = (image_width * image_height) / normal_arrow_count`
- `step = max(6, sqrt(target_density))`

Then it samples one arrow seed per grid cell at `(x, y)`.

This keeps arrow count roughly stable across resolutions.

## Computing Arrow Direction

For each sampled pixel:

1. Build a camera ray toward pixel center:
   - `pixel_center = frame.pixel00_loc + x * frame.pixel_delta_u + y * frame.pixel_delta_v`
   - `Ray r(frame.camera_center, pixel_center - frame.camera_center)`
2. Intersect CPU scene (`cpu_scene_for_arrows.hit(...)`).
3. Get normal from `Hit_record rec.normal`.
4. Project normal to screen axes:
   - `sx_n = dot(rec.normal, frame.u)`
   - `sy_n = -dot(rec.normal, frame.v)`
5. Normalize `(sx_n, sy_n)` and draw an arrow body + two head lines.

If no hit or projected magnitude is too small, that sample is skipped.

## Arrow Color

Arrow color is derived from the hit normal (same convention as show-normals):

- `R = 127.5 * (nx + 1)`
- `G = 127.5 * (ny + 1)`
- `B = 127.5 * (nz + 1)`

So arrows visually match the normal encoding.

## Line Rasterization + Thickness

A Bresenham-style line loop writes pixels to `display_image`.

Thickness is controlled by `normal_arrow_thickness` and implemented with soft coverage blending around each line point:

- `radius_f = max(0, thickness - 1.0)`
- `radius = ceil(radius_f + 1.0)`
- per pixel around the line point:
  - `dist = sqrt(ox*ox + oy*oy)`
  - `base = radius_f + 1.0 - dist`
  - `coverage = clamp(pow(max(0, base), 0.7) * 1.8, 0, 1)`
  - blend destination color with arrow color using `coverage`

The nonlinear gain makes small values above `1.0` visibly different (`1.0`, `1.1`, `1.2`), while still avoiding hard threshold jumps.

UI range is 1.0 to 2.5 (continuous slider, displayed with 0.1 precision).

Note on SDL thickness:
- SDL2 `SDL_RenderDrawLine` is 1-pixel only.
- There is no built-in variable-thickness line API in core SDL2.
- For this reason, thickness is implemented manually in the CPU image buffer.

## UI Controls

In `SDLGuiHandler::updateDisplay(...)` under Environment & Materials:

- `Show Normals` (checkbox) controls visualization mode
- `Normal Arrows (CPU)` toggles overlay
- `Arrow Count`
- `Arrow Scale`
- `Arrow Thickness`

The visualization mode was changed from dropdown to checkbox for faster toggling.

## Important Bug Fix: Arrows Persisting After Toggle Off

Problem:
- Arrows were composited into `display_image`.
- If accumulation had stopped, no new render pass refreshed the base image.
- Unticking arrows did not remove already drawn lines.

Fix:
- Track previous arrow settings each frame.
- If any arrow setting changes, force `needs_rerender = true`.
- This triggers a fresh `convertAccumToDisplayCUDA(...)` before overlay, clearing stale arrows.

Changed settings watched:
- `show_normal_arrows`
- `normal_arrow_count`
- `normal_arrow_scale`
- `normal_arrow_thickness`

## Scene Switching Behavior

When scene changes, both GPU and CPU representations are rebuilt:

- GPU: `buildGPUScene(active_scene)`
- CPU overlay scene: `cpu_scene_for_arrows = Scene::CPUSceneBuilder::buildCPUScene(original_scene)`

This keeps overlay normals consistent with the currently active scene.

## Performance Notes

- Cost scales with sampled arrows, not full resolution.
- Most expensive part is CPU ray-hit per arrow sample.
- `Arrow Count` is the main performance knob.

For heavy scenes/resolutions:
- Reduce `Arrow Count`
- Keep `Arrow Thickness` low

## Current Limitations

- Overlay is 2D (screen-space arrows), not 3D geometry arrows.
- Normal source for overlay uses CPU scene support; unsupported CPU geometry types will not contribute arrows.
- Overlay is debug-oriented and intentionally approximate.
