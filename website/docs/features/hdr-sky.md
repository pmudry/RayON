# HDR Sky Dome

RayON supports two sky models: a built-in procedural gradient (zero setup required) and a
full HDR environment map loaded from an `.hdr` file and projected as an equirectangular sky dome.

---

## Built-in gradient sky

When no YAML sky key is specified, the renderer uses a simple linear gradient between a horizon
colour and a zenith colour:

```yaml
settings:
  background_color: [0.05, 0.05, 0.08]  # overrides gradient with a solid colour
```

Omit `background_color` entirely to get the default blue-to-white gradient.
This sky model requires no image files and renders at zero memory cost.

---

## HDR environment sky dome

An equirectangular `.hdr` file (Radiance RGBE format) is mapped onto an infinite sphere
around the scene.
Any ray that misses all geometry samples the sky dome at the corresponding direction.

```yaml
settings:
  hdr_sky: "../resources/hdri/venice_sunset_8k.hdr"
```

The mapping is derived from the ray's unit direction vector:

$$\theta = \arccos(d_y), \quad \phi = \text{atan2}(-d_z,\; d_x)$$
$$u = \frac{\phi + \pi}{2\pi}, \quad v = \frac{\theta}{\pi}$$

The resulting $ (u, v) \in [0,1]^2 $ coordinates index into the equirectangular texture,
with $v = 0$ at the north pole and $v = 1$ at the south pole.

### Switching sky in interactive mode

In interactive mode the sky can be cycled at runtime **without restarting**:

| Input | Effect |
|---|---|
| **Numpad +** or **=** | Next HDRI in directory |
| **Numpad −** or **−** | Previous HDRI in directory |

A **Sky** combo-box in the *Environment* section of the ImGui panel also lets you pick any
loaded `.hdr` file by name.

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
    Any equirectangular `.hdr` file works — just place it in `resources/hdri/` and reference it
    from your YAML scene file or cycle to it with Numpad +/-.

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
