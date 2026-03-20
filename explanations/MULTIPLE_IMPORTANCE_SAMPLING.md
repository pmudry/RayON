# Multiple Importance Sampling (MIS) in RayON

## Overview

RayON's path tracer previously used **pure BSDF sampling**: at each bounce, a new direction was drawn
from the material's scatter distribution and the path continued only if it eventually happened to
hit an emissive surface by chance.  For scenes with small, bright area lights this produces
extreme variance — most paths never reach the light, and the rare paths that do are enormously
bright, creating fireflies.

MIS solves this by combining **two complementary sampling strategies**:

| Strategy | What it samples | When it wins |
|---|---|---|
| **BSDF sampling** | Direction drawn from material scatter distribution | Specular/mirror surfaces, smooth transitions to environment |
| **NEE (Next Event Estimation)** | Direction toward a randomly chosen area light | Diffuse surfaces with small bright lights |

The **power heuristic** (β = 2) blends both contributions so that each strategy receives weight
proportional to how well-suited it is, with no double-counting and no energy loss.

---

## The Rendering Equation (brief recap)

$$L(x, \omega_o) = L_e(x, \omega_o) + \int_\Omega f(x, \omega_i, \omega_o)\, L_i(x, \omega_i)\, |\cos\theta_i|\, d\omega_i$$

A path tracer approximates this integral by Monte Carlo sampling.  With pure BSDF sampling the
estimator for one bounce is:

$$\hat{L} = \frac{f(\omega_o, \omega_i)\cdot L_i \cdot |\cos\theta_i|}{p_\text{BSDF}(\omega_i)}$$

With MIS we instead compute two independent estimates — one via BSDF sampling, one via NEE — and
combine them with weights $w_a$ and $w_b$ that sum to 1:

$$\hat{L}_\text{MIS} = \frac{f \cdot L_e^{(\text{nee})} \cdot |\cos\theta|}{p_\text{light}} \cdot w_\text{light}
                     + \frac{f \cdot L_e^{(\text{bsdf})} \cdot |\cos\theta|}{p_\text{BSDF}} \cdot w_\text{BSDF}$$

Both terms are unbiased; combining them reduces variance.

---

## The Power Heuristic

The **power heuristic** (Veach & Guibas, 1995) is the standard MIS weighting function.  With
β = 2 (the most common choice) it reduces to:

$$w_a(p_a, p_b) = \frac{p_a^2}{p_a^2 + p_b^2}$$

No `powf()` is needed — squaring is sufficient and the formula is numerically stable even when
one PDF approaches zero.

```cpp
// mis_utils.hpp
inline double power_heuristic(double pdf_a, double pdf_b)
{
    double a = pdf_a * pdf_a;
    double b = pdf_b * pdf_b;
    return (a + b > 0.0) ? (a / (a + b)) : 0.0;
}
```

**Intuition**: if the light PDF is much larger than the BSDF PDF for a given direction, the NEE
estimate for that direction was taken with high probability and gets most of the credit.  If the
BSDF PDF is large (e.g. a specular spike), the BSDF sample gets most of the credit.

---

## Algorithm: Iterative Path Tracer with MIS

The CPU `ray_color()` loop in `cpu_ray_tracer.hpp` implements the full algorithm.  Below is a
commented walkthrough.

```
for each bounce:

  1. Trace current_ray → hit record (or sky)

  2. EMISSIVE HIT — apply MIS weight
     ─────────────────────────────────
     Le = rec.material.emitted()

     if bounce == 0 or previous bounce was specular:
         w = 1.0   ← full contribution (no prior NEE could have sampled this)
     else:
         light_pdf = lights.pdf_value(prev_origin, current_direction)
         w = power_heuristic(prev_bsdf_pdf, light_pdf)

     accumulated += throughput * Le * w

  3. BSDF SCATTER — sample next direction
     ──────────────────────────────────────
     srec = material.scatter_mis(ray, rec)
     if !srec: break  ← absorbed / terminal

  4. NEE — direct light sampling (skip for delta BSDFs)
     ───────────────────────────────────────────────────
     if not srec.is_specular and lights not empty:
         direction = lights.random_direction(rec.p)   ← mixture over light list
         light_pdf = lights.pdf_value(rec.p, direction)

         shadow_ray = Ray(rec.p + ε·n, direction)
         if world.hit(shadow_ray) hits a LIGHT:
             f       = material.eval_bsdf(ray, rec, direction)   // f(wo, wi)
             cos_θ   = dot(direction, rec.normal)
             p_mat   = material.scatter_pdf(ray, rec, direction)
             w_nee   = power_heuristic(light_pdf, p_mat)

             accumulated += throughput * f * Le_nee * cos_θ * w_nee / light_pdf

  5. ADVANCE PATH
     ─────────────
     if srec.is_specular:
         throughput    *= srec.bsdf_value          // attenuation handles everything
         prev_bsdf_pdf  = 1.0                       // delta: infinite PDF → weight = 1
         prev_specular  = true
     else:
         cos_θ          = dot(scatter_dir, rec.normal)
         throughput    *= srec.bsdf_value * cos_θ / srec.pdf
         prev_bsdf_pdf  = srec.pdf
         prev_specular  = false

     current_ray = srec.scattered
```

### Key invariant

`prev_bsdf_pdf` tracks the PDF of the BSDF sampling decision that produced `current_ray`.  When
that ray hits an emissive surface at the next bounce, this stored PDF is used to compute the MIS
weight (step 2 above), preventing double-counting with NEE.

---

## Material Interface Changes

### ScatterRecord

The new `scatter_mis()` method returns a `ScatterRecord` instead of an `(attenuation, ray)` pair:

```cpp
struct ScatterRecord
{
    Ray    scattered;    // sampled outgoing ray
    Color  bsdf_value;  // f(wo, wi) — NOT pre-divided by PDF
    double pdf;          // PDF of the sampled direction
    bool   is_specular;  // true → delta BSDF, skip NEE entirely
};
```

The separate `pdf` field is essential for MIS: the throughput update `bsdf_value * cos_θ / pdf`
can now be computed explicitly, and `pdf` can be re-used for the MIS weight at the next emissive
hit.

### Lambertian — full MIS support

```
f(wo, wi)     = albedo / π           (isotropic diffuse BRDF)
pdf(wi)       = max(0, cos θ) / π    (cosine-weighted hemisphere)
is_specular   = false
```

Sampling uses **Malley's method** (map uniform disk → hemisphere):

```cpp
double r   = sqrt(u1);
double phi = 2π * u2;
// local direction:  (r·cos φ, r·sin φ, sqrt(1 - u1))
// transform to world using ONB from surface normal
```

Because `f * cos_θ / pdf = (albedo/π) * cos_θ / (cos_θ/π) = albedo`, the throughput simplifies
to `albedo` in the pure-BSDF path — exactly as before.  The explicit form is only needed for the
NEE evaluation and MIS weights.

### Light material

```cpp
class Light : public Material {
    Color emitted(const Hit_record &) const override { return emission_color; }
    bool  scatter_mis(…) const override { return false; } // no scattering
};
```

Previously `LIGHT` materials fell back to `Lambertian` in the CPU scene builder — they emitted
nothing.  The fix ensures CPU and GPU are now consistent.

### Other materials (Mirror, Glass, GGX, ThinFilm, ClearCoat)

These use the default `scatter_mis()` fallback which wraps the legacy `scatter()` and sets
`is_specular = true`.  They skip NEE and MIS weighting entirely — correct behaviour for delta or
near-delta BSDFs.

---

## Light Sampling (NEE)

### Area PDF → Solid-angle PDF

NEE samples a point uniformly on the surface of a light and converts the area-measure PDF to a
solid-angle-measure PDF using:

$$p_\Omega = p_A \cdot \frac{d^2}{|\cos\theta_L|}$$

where $d$ is the distance from the shading point to the sampled point and $\theta_L$ is the angle
between the light normal and the direction toward the shading point.

### Rectangle lights

1. Sample $(u_1, u_2) \sim U[0,1]^2$ → $P = \text{corner} + u_1 \mathbf{u} + u_2 \mathbf{v}$
2. Area = $|\mathbf{u} \times \mathbf{v}|$
3. $p_\Omega = \dfrac{d^2}{|\cos\theta_L| \cdot A}$

### Sphere lights (cone sampling)

Sphere lights subtend a cone of half-angle $\theta_\text{max} = \arcsin(R / d)$ from the shading
point.  Sampling uniformly within this cone is more efficient than uniform sphere surface sampling:

1. $\cos\theta_\text{max} = \sqrt{1 - R^2/d^2}$
2. $\cos\theta \sim U[\cos\theta_\text{max}, 1]$
3. Solid angle $= 2\pi(1 - \cos\theta_\text{max})$
4. $p_\Omega = \dfrac{1}{2\pi(1 - \cos\theta_\text{max})}$

### Mixture distribution over multiple lights

When a scene has $N$ lights, RayON builds a **cumulative area CDF** over them.  At each shading
point, one light is selected proportionally to its area, then a point is sampled on that light.
The combined solid-angle PDF is:

$$p_\text{total}(\omega) = \sum_{i=1}^{N} p_\text{select}(i) \cdot p_{\Omega,i}(\omega)$$

On the CPU, the `Hittable_list::pdf_value()` averages over all lights (equivalent to uniform
selection); `random_direction()` picks one light uniformly at random.  On the GPU, the area CDF
stored in `scene.light_cdfs[]` provides area-proportional selection.

---

## GPU Implementation Details

### Light tracking in `CudaScene::Scene`

```cpp
struct Scene {
    // … existing fields …
    int   *light_indices; // device ptr: indices into geometries[] for LIGHT materials
    int    num_lights;
    float *light_cdfs;   // device ptr: cumulative area CDF [num_lights + 1]
};
```

Populated at scene build time in `scene_builder_cuda.cu` by scanning all geometry for
`MaterialType::LIGHT`.

### Shadow ray: `hit_scene_shadow()`

An early-exit BVH traversal that returns `true` as soon as any geometry occludes the path.  Stack
depth is capped at 16 (shadow rays rarely need deep traversal).  This avoids the full closest-hit
bookkeeping of `hit_scene()`, making shadow tests ~40–60% cheaper than full ray hits.

```cpp
__device__ bool hit_scene_shadow(const CudaScene::Scene &scene,
                                 const ray_simple &r, float t_min, float t_max);
```

### BSDF evaluation and PDF: `eval_bsdf_gpu()` / `scatter_pdf_gpu()`

Currently implemented for **Lambertian** only (all other materials treated as delta via
`is_delta_material()`):

```cpp
__device__ f3 eval_bsdf_gpu(const hit_record_simple &rec, const f3 &wi)
{
    if (rec.material == LAMBERTIAN)
        return rec.color / π;
    return f3(0);
}

__device__ float scatter_pdf_gpu(const hit_record_simple &rec, const f3 &wi)
{
    if (rec.material == LAMBERTIAN)
        return max(0, dot(wi, rec.normal)) / π;
    return 0.0f;
}
```

### MIS weight for emissive hits: `light_dir_pdf_gpu()`

When the BSDF-sampled path hits a `LIGHT` surface at bounce $b > 0$, the MIS weight requires the
solid-angle PDF that NEE would have assigned to the same direction from the previous hit point:

```cpp
__device__ float light_dir_pdf_gpu(const CudaScene::Scene &scene,
                                   int geom_idx, const f3 &prev_p, const f3 &dir);
```

The function checks whether `geom_idx` is in the light list, retrieves its `select_pdf` from the
CDF, then computes the solid-angle PDF for the rectangle or sphere as described above.

### Sobol sampling dimensions

One Sobol effect ID is used to ensure the NEE samples on the light surface are well-stratified and
independent from other random decisions along the path:

| ID | Name | Usage |
|---|---|---|
| 11 | `SOBOL_EFFECT_NEE_POINT` | 2D sample for the point on the selected light |

Light *selection* itself no longer uses a dedicated Sobol effect. Instead, a scalar `rand_float()`
is drawn from the per-thread RNG and mapped through the light CDF to choose which light to sample.
This keeps the sequence simple while still gaining stratification on the light surface via
`SOBOL_EFFECT_NEE_POINT`.

### `hit_record_simple.geom_idx`

A new `int geom_idx` field records which entry in `scene.geometries[]` was hit.  This is set by
`hit_scene()` and used by `light_dir_pdf_gpu()` to look up the light's geometry for PDF
computation.

---

## Expected Variance Reduction

| Scene type | Typical variance reduction | Equivalent sample reduction |
|---|---|---|
| Interior, small area lights | Very high | 10–50× |
| Mixed diffuse + specular | High | 5–20× |
| Outdoor with HDR (diffuse) | Moderate | 2–3× |
| Pure mirror / glass | None (MIS skipped) | 0× |

### Performance cost

NEE adds one shadow ray per non-specular bounce.  Shadow rays cost roughly 40–60% of a full
`hit_scene()` call (early-exit BVH).  For a typical 8-bounce path with 2 diffuse bounces, total
ray count increases by ~25–40%.

**Break-even**: at 10× variance reduction, the same noise level is achieved with 10× fewer
samples.  Paying 40% more per sample still yields a ~7× wall-clock speedup.

---

## Files Modified

| File | Change |
|---|---|
| `data_structures/mis_utils.hpp` | **New** — `power_heuristic()`, `LightSample` struct |
| `data_structures/material.hpp` | `ScatterRecord`, `Light` class, `Lambertian::scatter_mis()` + `eval_bsdf()` + `scatter_pdf()`, default `scatter_mis()` fallback |
| `data_structures/hittable.hpp` | Added `pdf_value()` and `random_direction()` virtual methods |
| `data_structures/hittable_list.hpp` | Mixture-PDF `pdf_value()` and `random_direction()` over object list |
| `cpu_shapes/sphere.hpp` | Cone solid-angle sampling for sphere lights |
| `cpu_shapes/rectangle.hpp` | Area-to-solid-angle sampling for rectangle lights |
| `scenes/scene_builder.hpp` | `CPUScene` struct (`scene` + `lights`), proper `Light` material, `THIN_FILM` support |
| `cpu_renderers/cpu_ray_tracer.hpp` | Iterative MIS loop replacing recursive `ray_color()` |
| `cpu_renderers/renderer_cpu_single_thread.hpp` | Updated to use `CPUScene` |
| `cpu_renderers/renderer_cpu_parallel.hpp` | Updated to use `CPUScene` |
| `gpu_renderers/cuda_utils.cuh` | `SOBOL_EFFECT_NEE_POINT` / `NEE_SELECT` |
| `gpu_renderers/cuda_scene.cuh` | `light_indices`, `num_lights`, `light_cdfs` in `Scene` |
| `gpu_renderers/scene_builder_cuda.cu` | Builds light list and area CDF; frees new device arrays |
| `gpu_renderers/cuda_raytracer.cuh` | `hit_scene_shadow()`, `eval_bsdf_gpu()`, `scatter_pdf_gpu()`, `sample_light_gpu()`, `light_dir_pdf_gpu()`, `is_delta_material()`, `geom_idx` in `hit_record_simple`, updated bounce loop |

---

## Verification Checklist

1. **Furnace test** — place a diffuse sphere inside a uniformly-emitting sphere (albedo 1.0).
   Result must be flat white; any colour bias indicates an energy leak in the MIS weights.

2. **Light count at startup** — both backends now print the number of detected lights:
   ```
   MIS: 2 light(s) detected
   CPU scene: 8 objects, 2 lights
   ```

3. **CPU/GPU consistency** — render a simple scene (e.g. `01_bvh_test_scene.yaml`) with both
   backends at high sample counts.  Mean pixel values should agree to within statistical noise.

4. **Variance comparison** — render a Cornell-box-style scene at 64 spp with and without MIS.
   The MIS image should show dramatically lower noise around shadow edges and on diffuse surfaces
   facing the light.

5. **Register budget** — run `nvcc --ptxas-options=-v` on `render_acc_kernel.cu`.  Target ≤ 64
   registers/thread to maintain 4 active warps per SM.

---

## References

- Veach, E. & Guibas, L. (1995). *Optimally Combining Sampling Techniques for Monte Carlo Rendering*. SIGGRAPH '95.
- Pharr, M., Jakob, W. & Humphreys, G. *Physically Based Rendering: From Theory to Implementation*, 4th ed. §13.4 (MIS), §12.2 (area lights).
- Shirley, P. *Ray Tracing: The Rest of Your Life*. §7 (mixture PDFs), §8 (light sampling).
