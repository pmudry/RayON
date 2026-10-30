# Multiple Importance Sampling (MIS)

RayON's path tracer originally used **pure BSDF sampling**: at each bounce the material chose a
scattered direction and the path continued, hoping to eventually hit a light source by chance.
For scenes with small, bright area lights this produces extreme variance — most paths never reach
the light, and the rare paths that do are enormously bright, creating *fireflies*.

**Multiple Importance Sampling** solves this by combining two complementary sampling strategies
and weighting each one by how well-suited it is to the situation.

---

## Why two strategies?

| Strategy | What it samples | When it wins |
|---|---|---|
| **BSDF sampling** | Direction drawn from the material scatter distribution | Specular / mirror surfaces, smooth environment transitions |
| **NEE (Next Event Estimation)** | Direction toward a randomly chosen area light | Diffuse surfaces illuminated by small, bright lights |

Neither strategy dominates in all cases. MIS blends them so the final estimator has **lower
variance than either strategy alone**, with no double-counting and no energy loss.

---

## The rendering equation (brief recap)

At each surface point the outgoing radiance is:

\[
L(x, \omega_o) = L_e(x, \omega_o) + \int_\Omega f(x, \omega_i, \omega_o)\, L_i(x, \omega_i)\, |\cos\theta_i|\, \mathrm{d}\omega_i
\]

A path tracer approximates this integral by Monte Carlo sampling. With pure BSDF sampling the
single-bounce estimator is:

\[
\hat{L} = \frac{f(\omega_o, \omega_i) \cdot L_i \cdot |\cos\theta_i|}{p_\text{BSDF}(\omega_i)}
\]

With MIS we instead compute **two independent estimates** — one via BSDF, one via NEE — and
combine them with weights \(w_a\) and \(w_b\):

\[
\hat{L}_\text{MIS} =
  \frac{f \cdot L_e^{(\text{nee})} \cdot |\cos\theta|}{p_\text{light}} \cdot w_\text{light}
  \;+\;
  \frac{f \cdot L_e^{(\text{bsdf})} \cdot |\cos\theta|}{p_\text{BSDF}} \cdot w_\text{BSDF}
\]

Both terms are unbiased; combining them reduces variance.

---

## The power heuristic

The **power heuristic** (Veach & Guibas, 1995) is the standard MIS weighting function.
With \(\beta = 2\) (the most common choice) it reduces to:

\[
w_a(p_a, p_b) = \frac{p_a^2}{p_a^2 + p_b^2}
\]

No `powf()` call is needed — squaring is sufficient and the formula is numerically stable even
when one PDF approaches zero.

```cpp
// as written in both kernels
float a = pdf_a * pdf_a;
float b = pdf_b * pdf_b;
float w = a / (a + b);
```

**Intuition**: if the light PDF is much larger than the BSDF PDF for a given direction, NEE was
the strategy that generated that sample most easily, so it gets most of the credit. If the BSDF
PDF is much larger (e.g. a specular spike), the BSDF sample gets most of the credit.

---

## Algorithm: iterative path tracer with MIS

Both GPU kernels (`cuda_raytracer.cuh` and the ray-generation program of `optix_programs.cu`)
implement the same loop. Below is a commented walkthrough of one path:

```
for each bounce:

  1. Trace current_ray → hit record (or sky)

  2. EMISSIVE HIT — apply MIS weight
     ─────────────────────────────────
     Le = rec.material.emitted()

     if bounce == 0 or previous bounce was specular:
         w = 1.0        ← full contribution (no prior NEE could have sampled this)
     else:
         light_pdf = lights.pdf_value(prev_origin, current_direction)
         w = power_heuristic(prev_bsdf_pdf, light_pdf)

     accumulated += throughput * Le * w

  3. BSDF SCATTER — sample next direction
     ──────────────────────────────────────
     srec = material.scatter_mis(ray, rec)
     if !srec: break     ← absorbed / terminal

  4. NEE — direct light sampling (skip for delta BSDFs)
     ───────────────────────────────────────────────────
     if not srec.is_specular and lights not empty:
         direction = lights.random_direction(rec.p)
         light_pdf = lights.pdf_value(rec.p, direction)

         shadow_ray = Ray(rec.p + ε·n, direction)
         if world.hit(shadow_ray) hits a LIGHT:
             f       = material.eval_bsdf(ray, rec, direction)
             cos_θ   = dot(direction, rec.normal)
             p_mat   = material.scatter_pdf(ray, rec, direction)
             w_nee   = power_heuristic(light_pdf, p_mat)

             accumulated += throughput * f * Le_nee * cos_θ * w_nee / light_pdf

  5. ADVANCE PATH
     ─────────────
     if srec.is_specular:
         throughput    *= srec.bsdf_value
         prev_bsdf_pdf  = 1.0          ← delta: infinite PDF → weight = 1
         prev_specular  = true
     else:
         cos_θ          = dot(scatter_dir, rec.normal)
         throughput    *= srec.bsdf_value * cos_θ / srec.pdf
         prev_bsdf_pdf  = srec.pdf
         prev_specular  = false

     current_ray = srec.scattered
```

!!! info "Key invariant"
    `prev_bsdf_pdf` tracks the PDF of the BSDF sampling decision that produced `current_ray`.
    When that ray hits an emissive surface at the **next** bounce, this stored PDF is used to
    compute the MIS weight (step 2), preventing double-counting with NEE.

---

## What the materials provide

For MIS a material has to answer two questions beyond "give me a scattered direction": what is
the BSDF value for an arbitrary direction, and what is the probability that the material's own
sampler would have chosen it. The scattering code therefore returns the BSDF value and the PDF
separately instead of the pre-divided attenuation; the PDF is kept for the MIS weight at the
next emissive hit.

### Lambertian — full MIS support

\[
f(\omega_o, \omega_i) = \frac{\rho}{\pi}
\qquad
p(\omega_i) = \frac{\max(0, \cos\theta)}{\pi}
\qquad
\texttt{is\_specular} = \text{false}
\]

Sampling uses **Malley's method** (uniform disk → hemisphere projection).
Because \(f \cdot \cos\theta / p = (\rho/\pi) \cdot \cos\theta / (\cos\theta/\pi) = \rho\),
the throughput simplifies to the albedo in the pure-BSDF path — exactly as before. The explicit
form is only needed for NEE evaluation and MIS weights.

### Other materials

Mirrors, glass, anisotropic GGX metals and thin films are treated as delta (or near-delta)
BSDFs: they skip NEE and get a weight of 1 when they hit a light, which is correct for a perfect
mirror and a reasonable approximation for the glossy ones. On the OptiX back-end the clear coat
and the SDF material also take shadow rays (the diffuse base of the clear coat benefits most;
treating it as delta produced a visible bias that was fixed in October 2026). Volumes sample
the lights from every scattering event with the phase function as the BSDF.

---

## Light sampling (NEE)

### Area PDF → solid-angle PDF

NEE samples a point uniformly on the surface of a light and converts the area-measure PDF to a
solid-angle-measure PDF:

\[
p_\Omega = p_A \cdot \frac{d^2}{|\cos\theta_L|}
\]

where \(d\) is the distance from the shading point to the sampled point and \(\theta_L\) is the
angle between the light normal and the direction toward the shading point.

### Rectangle lights

1. Sample \((u_1, u_2) \sim U[0,1]^2\) → \(P = \text{corner} + u_1\,\mathbf{u} + u_2\,\mathbf{v}\)
2. Area \(= |\mathbf{u} \times \mathbf{v}|\)
3. \(\displaystyle p_\Omega = \frac{d^2}{|\cos\theta_L| \cdot A}\)

### Sphere lights (cone sampling)

Sphere lights subtend a cone of half-angle
\(\theta_\text{max} = \arcsin(R / d)\) from the shading point. Sampling uniformly within this
cone is more efficient than uniform surface sampling:

1. \(\cos\theta_\text{max} = \sqrt{1 - R^2/d^2}\)
2. \(\cos\theta \sim U[\cos\theta_\text{max},\, 1]\)
3. Solid angle \(= 2\pi(1 - \cos\theta_\text{max})\)
4. \(\displaystyle p_\Omega = \frac{1}{2\pi(1 - \cos\theta_\text{max})}\)

### Mixture distribution over multiple lights

When a scene has \(N\) lights, RayON builds a **cumulative area CDF** over them. At each shading
point one light is selected proportionally to its area, then a point is sampled on it.
The combined solid-angle PDF is:

\[
p_\text{total}(\omega) = \sum_{i=1}^{N} p_\text{select}(i) \cdot p_{\Omega,i}(\omega)
\]

The area CDF is stored in `scene.light_cdfs[]` (CUDA) and `params.light_cdfs[]` (OptiX)
and gives area-proportional selection; the solid-angle PDF of a direction towards a given
light is evaluated the same way for the MIS weight of an emitter hit.

### The HDR sky as a light (OptiX)

In the OptiX interactive mode an HDR sky is one more NEE target: a 2D distribution
proportional to luminance × sin θ over a downsampled lat-long grid (marginal CDF over the rows,
conditional CDF per row) picks a direction, and a ray that escapes to the sky after a diffuse
bounce is MIS-weighted against that pdf. With emitters in the scene, the sky and the light list
each receive half of the NEE samples. See
[HDR Sky Dome — Sky importance sampling](../features/hdr-sky.md#sky-importance-sampling).

---

## GPU implementation

### Light tracking in `CudaScene::Scene`

```cpp
struct Scene {
    // … existing fields …
    int   *light_indices; // indices into geometries[] for LIGHT materials
    int    num_lights;
    float *light_cdfs;   // cumulative area CDF [num_lights + 1]
};
```

These arrays are populated at scene build time in `scene_builder_cuda.cu` by scanning all geometry
for `MaterialType::LIGHT`.

### Shadow rays

CUDA: `hit_scene_shadow()` is an early-exit BVH traversal that returns as soon as **any**
geometry occludes the segment, with a shallow stack, about half the cost of a closest-hit
query. OptiX: shadow rays use the `OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT` path with a slim
payload, and triangles are tested by the RT cores. Through a volume, the shadow ray's
transmittance is estimated by ratio tracking instead of a binary hit.

### BSDF evaluation and PDF

On CUDA the explicit BSDF is implemented for **Lambertian** (every other material is delta via
`is_delta_material()`); OptiX adds the SDF material and the clear coat:

```cpp
__device__ f3 eval_bsdf_gpu(const hit_record_simple &rec, const f3 &wi)
{
    if (rec.material == LAMBERTIAN)
        return rec.color / M_PI;
    return f3(0);
}

__device__ float scatter_pdf_gpu(const hit_record_simple &rec, const f3 &wi)
{
    if (rec.material == LAMBERTIAN)
        return max(0.f, dot(wi, rec.normal)) / M_PI;
    return 0.0f;
}
```

### MIS weight for emissive hits

When the BSDF-sampled path hits a `LIGHT` surface at bounce \(b > 0\), the MIS weight requires
the solid-angle PDF that NEE would have assigned to the same direction from the previous hit
point. This is computed by `light_dir_pdf_gpu()`, which looks up the geometry in the light list
and evaluates the appropriate solid-angle PDF (rectangle or sphere).

---

## Expected variance reduction

| Scene type | Typical variance reduction | Equivalent sample reduction |
|---|---|---|
| Interior, small area lights | Very high | 10–50× |
| Mixed diffuse + specular | High | 5–20× |
| Outdoor with HDR (diffuse) | Moderate | 2–3× |
| Pure mirror / glass | None (MIS skipped) | 0× |

### Performance cost

NEE adds one shadow ray per non-specular bounce. Shadow rays cost roughly 40–60 % of a full
`hit_scene()` call (early-exit BVH). For a typical 8-bounce path with 2 diffuse bounces, the
total ray count increases by ~25–40 %.

!!! tip "Break-even point"
    At 10× variance reduction, the same noise level is achieved with 10× fewer samples. Paying
    40 % more per sample still yields a **~7× wall-clock speedup**.

---

## Verification

1. **Furnace test** — `15_mis_furnace_test.yaml`: diffuse spheres in a uniform radiance
   environment. A sphere with albedo 1 must vanish against the background; any bias indicates
   an energy leak in the MIS weights.

2. **Light count at startup** — the OptiX backend prints the lights it uploads:
   ```
   OptiX NEE: 1 lights uploaded (total area 0.05)
   ```

3. **CUDA / OptiX consistency** — render a scene with both backends at high sample counts. Mean
   pixel values should agree to within statistical noise.

4. **Variance comparison** — render a Cornell-box-style scene at 64 spp with and without MIS. The
   MIS image should show dramatically lower noise around shadow edges and on diffuse surfaces
   facing the light.

---

## Visual comparison

The renders below isolate the effect of enabling MIS while keeping the Sobol' sampler fixed —
so all noise differences come from Next-Event Estimation alone.
Drag the slider to compare.

### Caustics chapel (512 SPP)

<div class="comparison-wrap comparison-block">
  <span class="label-left">No MIS</span>
  <span class="label-right">MIS on</span>
  <img-comparison-slider>
    <img slot="first"  src="../../assets/images/comparisons/06_caustics_sobol_nomis.png" alt="Caustics — Sobol', no MIS">
    <img slot="second" src="../../assets/images/comparisons/06_caustics_sobol_mis.png"   alt="Caustics — Sobol', MIS on">
  </img-comparison-slider>
  <p class="comparison-caption">
    <strong>Left: MIS off &nbsp;|&nbsp; Right: MIS on</strong> — Sobol' sampler, 512 SPP.
    MIS eliminates the firefly hot-spots and resolves the diffuse wall illumination correctly.
  </p>
</div>

### Default scene — area light only (64 SPP)

<div class="comparison-wrap comparison-block">
  <span class="label-left">No MIS</span>
  <span class="label-right">MIS on</span>
  <img-comparison-slider>
    <img slot="first"  src="../../assets/images/comparisons/default_sobol_nomis.png" alt="Default scene — Sobol', no MIS">
    <img slot="second" src="../../assets/images/comparisons/default_sobol_mis.png"   alt="Default scene — Sobol', MIS on">
  </img-comparison-slider>
  <p class="comparison-caption">
    <strong>Left: MIS off &nbsp;|&nbsp; Right: MIS on</strong> — Sobol' sampler, 64 SPP.
    Shadow edges, the lit side of the coloured spheres, and the rough-mirror highlight all
    resolve cleanly with NEE.
  </p>
</div>

!!! info "See the full comparison"
    Four scenes across all configurations are on the
    [Visual Comparisons](visual-comparisons.md) page.

---

## References

- Veach, E. & Guibas, L. (1995). *Optimally Combining Sampling Techniques for Monte Carlo Rendering*. SIGGRAPH '95.
- Pharr, M., Jakob, W. & Humphreys, G. *Physically Based Rendering: From Theory to Implementation*, 4th ed. §13.4 (MIS), §12.2 (area lights).
- Shirley, P. *Ray Tracing: The Rest of Your Life*. §7 (mixture PDFs), §8 (light sampling).
