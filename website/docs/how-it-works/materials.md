# Materials

Materials define how surfaces interact with light. In RayON, every material implements two
functions: `emitted()` (returns radiance emitted by the surface) and `scatter()` (returns a
scattered ray direction and an attenuation factor).

---

## Material overview

| Type | Enum | Scatters | Emits | Key parameter |
|---|---|---|---|---|
| **Lambertian** | `LAMBERTIAN` | ✓ | — | `albedo` (RGB) |
| **Metal** | `METAL` | ✓ | — | `albedo`, `roughness` [0–1] |
| **Mirror** | `MIRROR` | ✓ | — | — |
| **Rough Mirror** | `ROUGH_MIRROR` | ✓ | — | `roughness` [0–1], `albedo` (tint RGB) |
| **Glass** | `GLASS` | ✓ | — | `ior` (index of refraction) |
| **Dielectric** | `DIELECTRIC` | ✓ | — | `refractive_index` |
| **Anisotropic Metal** | `ANISOTROPIC_METAL` | ✓ | — | `roughness`, `anisotropy`, `eta`/`k` (complex IOR) |
| **Thin Film** | `THIN_FILM` | ✓ | — | `film_thickness` (nm), `film_ior` |
| **Clear Coat** | `CLEAR_COAT` | ✓ | — | `albedo`, `roughness`, `refractive_index` |
| **Area Light** | `LIGHT` | — | ✓ | `emission` (RGB) |
| **SDF Material** | `SDF_MATERIAL` | ✓ | — | For ray-marched SDF objects |
| **Show Normals** | `SHOW_NORMALS` | — | — | debug: normal → RGB |
| **Constant** | `CONSTANT` | — | — | debug: solid colour |

---

## Lambertian (diffuse)

A perfectly diffuse surface scatters light equally in all directions weighted by the cosine of the
angle from the surface normal (Lambert's cosine law):

\[
f_r^{\text{Lambert}} = \frac{\rho}{\pi}
\]

where \(\rho\) is the **albedo** — the fraction of light the surface reflects.

Scatter direction: a random point on a unit sphere offset along the surface normal
(cosine-weighted hemisphere — see [Sampling](sampling.md)).

```yaml
materials:
  - name: diffuse_red
    type: lambertian
    albedo: [0.8, 0.1, 0.1]
```

<div class="img-grid cols-2">
  <figure>
    <img src="../../assets/images/dev/old_lambertian.png" alt="Old uniform hemisphere sampling">
    <figcaption>Old: uniform hemisphere sampling — too much noise at grazing angles.</figcaption>
  </figure>
  <figure>
    <img src="../../assets/images/dev/new_lambertian.png" alt="Improved cosine-weighted sampling">
    <figcaption>New: cosine-weighted sampling — smoother gradients, same mean.</figcaption>
  </figure>
</div>

---

## Mirror (perfect specular)

A perfect mirror reflects rays using the reflection formula:

\[
\mathbf{r} = \mathbf{d} - 2(\mathbf{d} \cdot \hat{n})\hat{n}
\]

where \(\mathbf{d}\) is the incoming direction and \(\hat{n}\) is the surface normal.
Attenuation is always 1.0 (no energy loss).

```yaml
materials:
  - name: perfect_mirror
    type: mirror
```

---

## Rough Mirror (microfacet)

Extends the perfect mirror with a **roughness** parameter that adds a random perturbation to the
reflected direction:

\[
\mathbf{r}_\text{fuzzy} = \mathbf{r}_\text{perfect} + \text{roughness} \cdot \xi
\]

where \(\xi\) is a random vector in the unit sphere. Rays that scatter back below the surface are
absorbed (energy conservation).

A **tint** (RGB colour) modulates the reflection to produce metallic effects:

\[
\text{attenuation} = \text{tint} \times 0.7
\]

Preset tints:

| Metal | Tint RGB |
|---|---|
| Gold | (1.00, 0.85, 0.57) |
| Copper | (0.95, 0.50, 0.30) |
| Steel | (0.80, 0.85, 0.90) |
| Iron | (0.56, 0.57, 0.58) |
| Bronze | (0.80, 0.50, 0.20) |

```yaml
materials:
  - name: gold_sphere
    type: rough_mirror
    albedo: [1.0, 0.85, 0.57]   # tint = gold
    roughness: 0.03              # near-perfect reflection
```

<div class="img-grid cols-2">
  <figure>
    <img src="../../assets/images/samples/metals shine 2.png" alt="Tinted rough mirrors — gold, copper, steel">
    <figcaption>Gold, copper, and steel spheres at <code>roughness=0.05</code>. The warm tint is visible in highlights.</figcaption>
  </figure>
  <figure>
    <img src="../../assets/images/dev/output_rough_mirror.png" alt="Roughness gradient from 0 to 0.5">
    <figcaption>Left to right: roughness 0.0 → 0.5. The reflected image progressively blurs.</figcaption>
  </figure>
</div>

---

## Anisotropic Metal (GGX microfacet)

A physically-based conductor material using the **GGX microfacet distribution** with a full
complex index of refraction (η + ik) for accurate spectral Fresnel. Unlike rough mirror, it models
the anisotropic highlight patterns seen on brushed metal, vinyl records, and machined surfaces.

\[
f_r = \frac{D_\text{GGX}(\mathbf{m})\, G(\mathbf{l},\mathbf{v})\, F(\mathbf{v},\mathbf{m})}
           {4\,(\mathbf{n}\cdot\mathbf{l})\,(\mathbf{n}\cdot\mathbf{v})}
\]

The `anisotropy` parameter [0–1] stretches the highlight tangentially (0 = isotropic sphere,
1 = line highlight). Named presets use measured IOR values:

| Preset | η (RGB) | k (RGB) |
|---|---|---|
| `gold` | (0.18, 0.42, 1.37) | (3.42, 2.35, 1.77) |
| `silver` | (0.05, 0.06, 0.05) | (4.18, 3.35, 2.58) |
| `copper` | (0.27, 0.68, 1.22) | (3.60, 2.63, 2.29) |
| `aluminum` | (1.35, 0.97, 0.53) | (7.47, 6.40, 5.28) |

```yaml
materials:
  - name: brushed_gold
    type: anisotropic_metal
    preset: gold             # uses measured IOR values
    roughness: 0.08
    anisotropy: 0.7

  - name: custom_metal
    type: anisotropic_metal
    roughness: 0.1
    anisotropy: 0.3
    eta: [0.18, 0.42, 1.37]
    k:   [3.42, 2.35, 1.77]
```

<img class="render-img" src="../../assets/images/samples/dielectric metsals.png"
     alt="Anisotropic metal spheres with brushed highlight patterns">

*Anisotropic metal spheres. The stretched highlight pattern is controlled by the `anisotropy` parameter.*

---

## Thin Film (interference)

Simulates the iridescent colour shifts seen on soap bubbles, oil slicks, and butterfly wings.
Thin-film interference arises when the optical path difference between reflections from the
top and bottom of a thin layer produces wavelength-dependent constructive/destructive interference:

\[
\Delta\phi = \frac{4\pi\, n_f\, d\, \cos\theta_t}{\lambda}
\]

The `film_thickness` parameter controls which wavelengths are amplified — smaller values produce
blue/violet; larger values shift toward red.

```yaml
materials:
  - name: soap_bubble
    type: thin_film
    film_thickness: 400      # nanometers — shifts highlight colour
    film_ior: 1.33           # refractive index of the film (water/soap ≈ 1.33)
```

<img class="render-img" src="../../assets/images/samples/thin_film_shader.png"
     alt="Thin-film iridescent soap bubbles">

*Thin-film material at various film thicknesses. Varying `film_thickness` sweeps through the visible spectrum.*

---

## Clear Coat

A two-layer material combining a smooth **dielectric coat** (Fresnel specular) over a **diffuse
base**. Physically it models car paint, lacquered wood, and hard-plastic objects.

At each ray hit, a Fresnel-weighted coin flip decides:

- **Coat layer** (probability proportional to Fresnel reflectance) → sharp specular reflection.
- **Base layer** (remaining probability) → Lambertian diffuse scatter using `albedo`.

```yaml
materials:
  - name: red_car_paint
    type: clear_coat
    albedo: [0.8, 0.05, 0.05]   # diffuse base colour
    roughness: 0.04              # coat roughness (0 = mirror coat)
    refractive_index: 1.5        # coat IOR (1.5 = typical lacquer/plastic)
```

<img class="render-img" src="../../assets/images/samples/plastic_shading.png"
     alt="Clear-coat material on a Stanford dragon">

*The Stanford dragon with a red clear-coat material. The highlight is the dielectric coat; the
colour is the diffuse base beneath.*

---

## Glass (dielectric)

Glass both reflects and refracts light. The ratio of reflected to transmitted energy is governed
by the **Fresnel equations**, approximated efficiently using **Schlick's formula**:

\[
R(\theta) = R_0 + (1 - R_0)(1 - \cos\theta)^5, \quad R_0 = \left(\frac{n_1 - n_2}{n_1 + n_2}\right)^2
\]

Refraction direction follows **Snell's law**:

\[
n_1 \sin\theta_1 = n_2 \sin\theta_2
\]

When \(n_1 > n_2\) and \(\theta_1\) exceeds the critical angle, **total internal reflection** occurs
and the surface acts as a perfect mirror.

```yaml
materials:
  - name: glass
    type: glass
    ior: 1.5      # crown glass (water ≈ 1.33, diamond ≈ 2.42)
```

<img class="render-img" src="../../assets/images/samples/dielectric metsals.png"
     alt="Glass spheres alongside metallic surfaces">

*Glass spheres refracting the background. The caustic ring under the sphere is not explicit — it
emerges from the path tracing integral.*

---

## Area Light

Light sources are geometry (rectangles or spheres) with the `LIGHT` material. The `scatter()`
function returns `false` (no further bouncing) and `emitted()` returns the emission colour.

Soft shadows emerge naturally — a larger light source means more rays from the scene "see" the
light directly, producing smooth penumbrae.

```yaml
materials:
  - name: warm_white_light
    type: light
    emission: [5.0, 4.5, 3.5]     # HDR — values > 1 are physically valid

geometry:
  - type: rectangle
    material: warm_white_light
    position: [-1.0, 3.0, -2.0]
    u_vec: [2.5, 0.0, 0.0]
    v_vec: [0.0, 0.0, 1.5]
```

<img class="render-img" src="../../assets/images/samples/cornell.png" alt="Cornell box with area light">

*Cornell box lit by a single rectangular area light on the ceiling.
The coloured walls produce visible colour bleeding on the floor and white sphere.*

---

## Debug materials

Two materials are available for development and debugging:

**ShowNormals** — maps the surface normal to RGB:
\[
\text{colour} = 0.5(\hat{n} + \mathbf{1})
\]
Red = +X axis, Green = +Y axis, Blue = +Z axis.

**Constant** — emits a fixed solid colour regardless of lighting. Useful for environment backgrounds.

<img class="render-img" src="../../assets/images/samples/gui_debug/normals.png" alt="Show normals visualisation">

*All surfaces shown with the `ShowNormals` material. Note how the smooth normal interpolation on
the imported mesh makes the face boundaries invisible.*

---

## The material dispatch system

Both CPU and GPU renderers share the same material logic. On the CPU, standard
C++ virtual functions are used. On the GPU, virtual dispatch is not allowed, so RayON stores all
material parameters in a **flat POD union** and selects the correct code path at runtime via a
`switch` statement:

```cpp
// MaterialParamsUnion — POD struct, GPU safe
union MaterialParamsUnion {
    LambertianParams     lambertian;
    MirrorParams         mirror;
    RoughMirrorParams    rough_mirror;
    GlassParams          glass;
    LightParams          light;
    AnisotropicParams    anisotropic_metal;
    ThinFilmParams       thin_film;
    ClearCoatParams      clear_coat;
    // ...
};
```

At runtime the `MaterialType` enum tag selects the correct branch:

```cpp
__device__ void scatter_material(const Material& mat, ...) {
    switch (mat.type) {
        case MaterialType::LAMBERTIAN:       scatter_lambertian(...);       break;
        case MaterialType::ROUGH_MIRROR:     scatter_rough_mirror(...);     break;
        case MaterialType::GLASS:            scatter_glass(...);            break;
        case MaterialType::ANISOTROPIC_METAL: scatter_anisotropic(...);    break;
        // ...
    }
}
```

The compiler inlines each case completely — zero virtual-function overhead on the GPU.
See [Architecture → Scene System](../architecture/scene-system.md) for details about how
materials are transferred to the GPU.
