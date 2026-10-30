# Lights & Shadows

Lights in RayON are ordinary geometry carrying a `light` material. Rectangles and spheres are
sampled explicitly by next event estimation, which is what turns a tiny bright bulb from a
firefly generator into clean, soft shadows at a handful of samples per pixel.

---

## Emitters

```yaml
materials:
  - name: "sun"
    type: "light"
    emission: [300.0, 285.0, 255.0]        # linear radiance, any magnitude

  - name: "warm_light"
    type: "light"
    color: [1.0, 0.875, 0.75]              # or a colour …
    emission_intensity: 12                 # … times a scalar

geometry:
  - type: "rectangle"                      # area light
    material: "warm_light"
    corner: [-1.5, 4, -1.5]
    u: [3, 0, 0]
    v: [0, 0, 3]

  - type: "sphere"                         # bulb
    material: "sun"
    center: [-42, 26, 14]
    radius: 4
    visible: false                         # lights the scene, not seen by camera rays
```

The emission is radiance: a small light needs a large value to deliver the same power as a
large panel. The comment in `16_nee_small_light.yaml` shows the arithmetic for swapping a
12 cm square panel for a 6 cm bulb.

### Hidden emitters

`visible: false` hides an object from camera rays only. It still casts light, shadows and shows
in reflections. Both back-ends honour the flag. The cloud scenes use it for the sun, the SDF
gallery for a large soft box above the room.

---

## Next event estimation

Every non-specular hit traces one **shadow ray** towards a point sampled on a light: a light is
picked with probability proportional to its area, then a point on it (uniform on rectangles,
inside the visible cone for spheres). The emitter's contribution is weighted against the chance
that the material's own scattering would have found it, so that nothing is counted twice. This is
multiple importance sampling; the math is on the [MIS page](../how-it-works/mis.md).

On OptiX, Lambertian, clear-coat and SDF-material surfaces take shadow rays; mirrors, glass,
anisotropic metals and thin films are treated as specular and skip them. The CUDA back-end
samples lights for Lambertian surfaces. Volumes trace shadow rays from every scattering event,
through the medium itself ([Volumes](volumes.md)).

| Option | Where | Default |
|---|---|---|
| Enable MIS | GUI *MIS / NEE*, `--no-mis` | on |
| NEE stride | GUI, `--nee-stride <N>` | 1 (every sample; the first bounce always traces its shadow ray) |
| Disable while the camera moves | GUI *Camera Motion*, `--motion-gate-mis` | off |

### What it buys

`16_nee_small_light.yaml`: five matte spheres in a dark room under a 6 cm bulb. Render it at 4
samples per pixel with and without MIS:

```bash
./rayon -m 4 -s 4 --scene ../resources/scenes/16_nee_small_light.yaml
./rayon -m 4 -s 4 --scene ../resources/scenes/16_nee_small_light.yaml --no-mis
```

Without explicit light sampling, almost no random bounce hits a 6 cm sphere, so the image is
black with sparse white speckles. With it, the shadows are already readable. Both converge to
the same image at high sample counts. The [Visual Comparisons](../how-it-works/visual-comparisons.md)
page shows the effect on four scenes with a slider.

---

## Light intensity and falloff

Two per-scene settings adjust all the lights at once, live from the GUI, saved with the camera.

| Setting | GUI | Default | Effect |
|---|---|---|---|
| `light_intensity` | *Lights inten.* | 1 | Multiplier on every emission |
| `light_falloff` | *Lights falloff* | 2 | Distance exponent of the lights' reach |

The falloff is an artistic control layered over the physics. Sphere and rectangle lights
already follow the inverse square law through the geometry; the setting multiplies every light
contribution at distance \(d\) by

\[
d^{\,2 - \text{falloff}}
\]

so nothing changes at one scene unit from the light. At 1 the light carries farther and a
dark room opens up; at 3 or 4 it dies out around the source and the scene becomes a bulb in
the dark. The factor is applied to the shadow rays and to the emitter hits of scattered paths
alike, which keeps MIS consistent; camera rays see the emitters unchanged. It is not energy
conserving, and it applies to the volumetric shadow rays too, so a sun 50 units away reacts
strongly to it.

```yaml
settings:
  light_intensity: 1.0
  light_falloff: 2.0
```

---

## Ambient light

Rays that leave the scene return the sky: the built-in gradient (`background_color` at the top,
white at the horizon) scaled by `background_intensity`, or an HDR environment
([HDR Sky Dome](hdr-sky.md)). Set `background_intensity: 0` for a scene lit by its emitters
alone, as in the Cornell box and the NEE scene.

---

## Fireflies

A single very bright sample, typically a path that found a small light through a specular
bounce, can leave a white dot that takes thousands of samples to fade. Two mechanisms keep them
in check:

- the still image clamps each sample's luminance to a ceiling that grows with the sample
  count, so the estimate stays unbiased in the limit;
- while the camera moves, isolated bright pixels are scaled down at display time (the
  **Despeckle motion frames** option), which touches the preview only.
