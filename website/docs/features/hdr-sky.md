# HDR Sky Dome

RayON supports two sky models: a built-in procedural gradient (zero setup required) and a
full HDR environment map loaded from an `.hdr` file and projected as an equirectangular sky dome.

---

## Built-in gradient sky

By default the sky is a vertical gradient from white at the horizon to `background_color` at
the zenith, scaled by `background_intensity` (the GUI's *Ambient Light*):

```yaml
settings:
  background_color: [0.45, 0.65, 0.95]   # zenith colour (default: [0.5, 0.7, 1.0])
  background_intensity: 0.25             # 0 = no sky light at all
```

This sky needs no image file and costs nothing.

---

## HDR environment sky dome

An equirectangular `.hdr` file (Radiance RGBE format) is mapped onto an infinite sphere
around the scene. Any ray that misses all geometry samples the sky dome in its direction.

The sky is chosen in the interactive modes, not in the scene file: at start-up the renderer
scans `resources/hdri/` (looked up as `../resources/hdri` from `build/`) and lists every
`.hdr` it finds in the **Sky** combo of the *Environment & Materials* section, after the
built-in gradient. **Numpad + / −** cycle through them. The offline modes render with the
gradient sky.

The mapping is derived from the ray's unit direction vector:

$$\theta = \arccos(d_y), \quad \phi = \text{atan2}(-d_z,\; d_x)$$
$$u = \frac{\phi + \pi}{2\pi}, \quad v = \frac{\theta}{\pi}$$

The resulting $ (u, v) \in [0,1]^2 $ coordinates index into the equirectangular texture,
with $v = 0$ at the north pole and $v = 1$ at the south pole.

### Switching sky in interactive mode

| Input | Effect |
|---|---|
| **Numpad +** or **=** | Next HDRI of the directory |
| **Numpad −** or **−** | Previous HDRI |
| **Sky** combo | Pick any file by name; the first entry is the built-in gradient |
| `--sky <name>` | Command line: start with the first file whose name contains `<name>` (`--sky sunflowers`) |
| `settings.hdr_sky` | Scene file: the sky to show when this scene is loaded (`hdr_sky: "venice_sunset"`), the gradient without it; `--sky` wins at start-up |

Switching takes the decode time of the file the first time (seconds for an 8K map), then the
`.hdrcache` sidecar makes it near instant.

---

## Downloading the HDRI files

The bundled HDRIs are sourced from [Poly Haven](https://polyhaven.com) under the
[CC0 licence](https://creativecommons.org/publicdomain/zero/1.0/).
Because each file is 25–250 MB, they are **not included in the repository**.
A download script is provided:

```bash
# Download 8K HDRIs (recommended quality)
cd resources/hdri
bash download_hdri.sh 8k

# Or lower resolutions
bash download_hdri.sh 2k
bash download_hdri.sh 4k
```

The six bundled environments:

| Environment | Source |
|---|---|
| `venice_sunset` | [polyhaven.com/a/venice_sunset](https://polyhaven.com/a/venice_sunset) |
| `kloppenheim_06` | [polyhaven.com/a/kloppenheim_06](https://polyhaven.com/a/kloppenheim_06) |
| `autumn_crossing` | [polyhaven.com/a/autumn_crossing](https://polyhaven.com/a/autumn_crossing) |
| `studio_small_03` | [polyhaven.com/a/studio_small_03](https://polyhaven.com/a/studio_small_03) |
| `sunflowers_puresky` | [polyhaven.com/a/sunflowers_puresky](https://polyhaven.com/a/sunflowers_puresky) |
| `rosendal_plains_2` | [polyhaven.com/a/rosendal_plains_2](https://polyhaven.com/a/rosendal_plains_2) |

!!! note "Adding your own HDRIs"
    Any equirectangular `.hdr` file works: put it in `resources/hdri/` and it appears in the
    Sky combo at the next start.

A very bright sun disk in an HDR map is the classic firefly source: in `sunflowers_puresky`
the sun covers 0.08 millisteradian, about 180 texels of the 8K map, yet carries 45 % of the
sky's energy. The OptiX interactive mode samples the sky directly (see
[Sky importance sampling](#sky-importance-sampling) below), so the sun lights diffuse surfaces
through shadow rays and casts proper shadows instead of being found by chance. The CUDA
interactive mode has no sky sampling: a diffuse bounce hits the sun once in about 40 000
samples, each hit worth tens of thousands, and only the luminance clamp (a fixed ceiling of 20
while the camera moves, a growing one when still, see [Lights & Shadows](lighting.md#fireflies))
keeps the speckle in check. It never converges there: the clamp grows with the square root of
the sample count and the hits arrive in proportion to it, so the speckle keeps a constant
amplitude.

---

## Gallery

<div class="img-grid cols-3">
  <img src="../../assets/images/samples/hdri/hdri_0.png" alt="HDR sky render 0">
  <img src="../../assets/images/samples/hdri/hdri_1.png" alt="HDR sky render 1">
  <img src="../../assets/images/samples/hdri/hdri_3.png" alt="HDR sky render 3">
  <img src="../../assets/images/samples/hdri/hdri_4.png" alt="HDR sky render 4">
  <img src="../../assets/images/samples/hdri/hdri_5.png" alt="HDR sky render 5">
  <img src="../../assets/images/samples/hdri/hdri_6.png" alt="HDR sky render 6">
  <img src="../../assets/images/samples/hdri/hdri_7.png" alt="HDR sky render 7">
  <img src="../../assets/images/samples/hdri/hdri_8.png" alt="HDR sky render 8">
  <img src="../../assets/images/samples/hdri/hdri_9.png" alt="HDR sky render 9">
</div>

---

## Technical implementation

### Sky importance sampling

*OptiX interactive mode only.* When an HDR map is uploaded, `buildEnvSamplingTables()`
(`optix_renderer.cu`) reduces it to a grid of at most 2048 × 1024 cells (a 4 × 4 texel box
on an 8K map) weighted by luminance × sin θ, the solid angle of a lat-long cell, and builds a
2D distribution over it: a marginal CDF over the rows and one conditional CDF per row. The
tables are accumulated in double precision and stored as float, so the probability of a cell
is exactly the difference of two consecutive CDF entries, on the sampling side as on the pdf
side.

The sky then takes part in next-event estimation like a light: `sample_env_optix()` picks a
row from the marginal CDF, a cell from that row, a uniform direction inside the cell and
returns its radiance with the solid-angle pdf

$$p_\Omega(u, v) = \frac{p_\text{row} \, p_\text{cell} \, W H}{2\pi^2 \sin\theta}$$

where \(2\pi^2 \sin\theta\) is the Jacobian of the lat-long mapping. When the scene also has
emitters, half of the NEE samples go to the sky and half to the light list, and every pdf
carries that selection probability. A BSDF-sampled ray that escapes to the sky is weighted by
the power heuristic against `env_pdf_optix()` of its direction, exactly as an emitter hit is
weighted against the light pdf, so the sun is never counted twice. Samples drawn below the
surface skip the shadow ray. Everything is tied to the **Enable MIS** toggle: with MIS off the
sky is left to BSDF sampling, as before.

Measured on the default scene under `sunflowers_puresky` after 30 s at 720p: the speckle count
drops by a factor of 30 and the image is 20 % brighter, the sun's share that the luminance
clamp was holding back. The remaining dots are sun caustics: the sun reaching a diffuse surface
through the glass sphere or off the metal sphere, paths that no light sampling can reach (the
same scene with those two spheres made diffuse shows none). They are held down by the
[indirect clamp](lighting.md#indirect-clamp-optix). The cost is the shadow ray of each diffuse bounce (about a quarter
of the frame rate on that scene, which had no emitter and therefore no shadow rays before).

### float16 GPU texture

The `.hdr` file is decoded from Radiance RGBE (four bytes per pixel) to 32-bit float triplets by
`stb_image`, then each channel is converted to **IEEE 754 float16** (half precision).
The resulting four-channel half-float image is uploaded to a `cudaTextureObject_t` with format:

```cpp
cudaChannelFormatDesc{16, 16, 16, 16, cudaChannelFormatKindFloat}
```

CUDA hardware automatically promotes the fp16 values to fp32 on `tex2D<float4>` fetch —
no shader-side conversion is needed.

**Memory impact:**

| Resolution | Uncompressed fp32 (VRAM) | fp16 VRAM | Saving |
|---|---|---|---|
| 2K (2048 × 1024) | 32 MB | **16 MB** | 50 % |
| 4K (4096 × 2048) | 128 MB | **64 MB** | 50 % |
| 8K (8192 × 4096) | 512 MB | **256 MB** | 50 % |

### Binary disk cache (`.hdrcache`)

Loading and decoding an 8K `.hdr` file (≈ 200 MB on disk) takes several seconds.
On subsequent launches, the renderer writes a binary sidecar next to the `.hdr` file
(e.g. `venice_sunset_8k.hdr.hdrcache`) and reads it on the next run.

A `.hdrcache` file is a compact binary blob with a 24-byte header followed by the raw fp16
pixel data — no decompression or format conversion needed.
Typical time savings: **5–10× faster** loads compared to re-decoding the RGBE source.

```
┌─────────────────────────────────┐
│  magic  │  version  │  w  │  h  │  (16 bytes)
│            source_size          │  (8 bytes, staleness check)
├─────────────────────────────────┤
│  width × height × 4 × uint16_t │  (raw fp16 pixels, RGBA order)
└─────────────────────────────────┘
```

The source file's byte-size is stored in the header.
If the `.hdr` file is replaced or modified (size changes), the cache is automatically
invalidated and regenerated.

To skip the cache entirely (e.g. for diagnostics):

```bash
./rayon --no-hdr-cache --scene ../resources/scenes/default_scene.yaml
```

### fp16 clamping

HDR scene radiance values can exceed the fp16 finite range (65 504).
Values above this threshold are clamped to `±65504` before storage — ensuring the GPU texture
never receives ±∞, which would propagate as NaN during path-tracing arithmetic and produce
black "firefly" pixels.

!!! warning "Stale cache files"
    If you have `.hdrcache` files created by an older version of RayON (before the clamping fix),
    delete them and let the renderer regenerate them.
    Newly generated caches are identified by a version field in the header and will be
    automatically regenerated when the version does not match.
