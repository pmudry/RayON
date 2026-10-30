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
picked with probability proportional to its power (area × luminance of its emission) on OptiX, to
its area on CUDA, then a point on it (uniform on rectangles,
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

## Caustics (OptiX)

A shadow ray cannot pass through glass or bounce off a mirror, so the light focused by a glass
sphere onto the floor (a caustic) is only found when a diffuse bounce happens to go through the
glass and hit the lamp: speckle that the indirect clamp below then dims. OptiX adds a **caustic
pass** after each batch: light paths leave the lamps towards the mirrors and glass (aimed at their
bounding spheres), follow their perfectly specular surfaces, and at the first diffuse surface are
joined to the camera and added to the pixel they land in. The camera paths leave exactly those
paths (lamp → mirrors/glass → diffuse surface seen directly → camera) to the pass, so nothing is
counted twice and the result is unbiased: the caustic converges smoothly instead of as speckle.

- Covered: smooth mirrors and metals, glass, thin films. Glossy surfaces (roughness > 0), clear
  coat and media stay with the camera paths, as do caustics seen through glass or in a mirror.
- Lamps only: caustics of the HDR sky stay with the camera paths.
- Off while depth of field is on (the pass assumes a pinhole camera), and with `--no-caustics`.
- Cost: a quarter of a light path per camera sample, a few percent of the frame time.

| Option | Where | Default |
|---|---|---|
| Caustic pass | GUI *MIS / NEE* (OptiX interactive), `--no-caustics` | on (OptiX modes) |

`21_caustics.yaml` shows it best: render it with and without `--no-caustics`.

## Light intensity and falloff

Two per-scene settings adjust all the lights at once, live from the GUI, saved with the camera.

| Setting | GUI | Default | Effect |
|---|---|---|---|
| `light_intensity` | *Lights inten.* | 1 | Multiplier on every emission |
| `light_falloff` | *Lights falloff* | 2 | Distance exponent of the lights' reach |
| `indirect_clamp` | *Indirect clamp* | 50 | OptiX: per-sample ceiling on indirect light, 0 = off (see [Fireflies](#fireflies)) |

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
  **Despeckle motion frames** option), which touches the preview only;
- the sun of an HDR sky, the brightest firefly source of all, is sampled directly by the
  OptiX interactive mode (sky importance sampling with MIS, see
  [HDR Sky Dome](hdr-sky.md#sky-importance-sampling)), so it reaches diffuse surfaces through
  shadow rays rather than by chance;
- what light sampling cannot reach, the **indirect clamp** (OptiX) holds down: a fixed
  per-sample ceiling on the light gathered after the first diffuse bounce.

### Indirect clamp (OptiX)

Caustic paths, a light or the sun reaching a diffuse surface through glass or off a mirror,
cannot be targeted by shadow rays. Under the growing clamp above they still converge, but
slowly: the ceiling grows as \(\sqrt{N}\) and the hits arrive in proportion to \(N\), so an
isolated dot keeps an amplitude of \(20/\sqrt{N}\), still visible after tens of thousands of
samples. The indirect clamp is the same answer as Cycles' *Clamp Indirect*: everything a sample
gathers after it leaves its first diffuse surface (or its first scattering event in a medium)
is capped at `indirect_clamp` luminance per sample, 50 by default, 0 to disable. What the
camera sees directly, through mirrors and glass included, and the shadow rays of that first
surface are untouched, so lights, highlights and shadows keep their full value. The price is a
bias on the caustics themselves: a caustic brighter than the ceiling per path (the sun focused
by a glass sphere, tens of thousands per hit) is dimmed, a caustic from a lamp of emission 25
is not. The setting is per scene, live in the GUI (*MIS / NEE* section) and saved with the
camera.
