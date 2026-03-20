# Visual Comparisons — Sobol' & MIS

This page shows the concrete, pixel-level effect of RayON's two key variance-reduction
techniques side by side.  Drag the **vertical slider** in each pair to reveal how the right-hand
image differs from the left-hand one.

---

## How to read the comparisons

Each scene is shown with **three slider pairs**, isolating one variable at a time:

| Pair | Left image | Right image | What you see |
|---|---|---|---|
| **Sobol' effect** | PCG sampler, MIS off | Sobol' sampler, MIS off | Noise reduction from quasi-random sampling alone |
| **MIS effect** | Sobol' sampler, MIS off | Sobol' sampler, MIS on | Noise reduction from Next-Event-Estimation alone |
| **Combined** | PCG sampler, MIS off | Sobol' sampler, MIS on | Full improvement — the real-world default vs worst-case baseline |

All renders: **1280 × 720 px**, offline CUDA mode.
The caustics chapel uses **512 SPP**; all other scenes use **64 SPP**.

!!! tip "How much noise?"
    At 64 SPP, differences are dramatic — this is deliberately the "hard" regime where
    both techniques matter most.  In production at 2 048+ SPP the images converge and the
    differences narrow, but the wall-clock time saving (fewer samples needed) is the same.

---

<div class="scene-header">
<h2>Scene 1 — OBJ Statue</h2>
<p class="scene-meta">rough-mirror gold statue · area light · 64 SPP · BVH enabled</p>
</div>

A metallic statue illuminated by a single rectangular area light. The rough-mirror surface
scatters light in a broad lobe — NEE (MIS) is very effective here because diffuse-like surfaces
benefit most from direct light sampling.  Sobol' reduces the structured noise visible in the
unlit regions of the ground plane.

### Sobol' effect (MIS disabled)

<div class="comparison-wrap comparison-block">
  <span class="label-left">PCG</span>
  <span class="label-right">Sobol'</span>
  <img-comparison-slider>
    <img slot="first"  src="../../assets/images/comparisons/04_statue_pcg_nomis.png"   alt="OBJ Statue — PCG sampler, MIS off">
    <img slot="second" src="../../assets/images/comparisons/04_statue_sobol_nomis.png" alt="OBJ Statue — Sobol' sampler, MIS off">
  </img-comparison-slider>
  <p class="comparison-caption">
    <strong>Left: PCG (pseudo-random) &nbsp;|&nbsp; Right: Sobol' (quasi-random)</strong> — MIS disabled on both.
    Sobol' reduces high-frequency grain in the shadow regions and on the ground plane.
  </p>
</div>

### MIS / NEE effect (Sobol' sampler)

<div class="comparison-wrap comparison-block">
  <span class="label-left">No MIS</span>
  <span class="label-right">MIS on</span>
  <img-comparison-slider>
    <img slot="first"  src="../../assets/images/comparisons/04_statue_sobol_nomis.png" alt="OBJ Statue — Sobol', MIS off">
    <img slot="second" src="../../assets/images/comparisons/04_statue_sobol_mis.png"   alt="OBJ Statue — Sobol', MIS on">
  </img-comparison-slider>
  <p class="comparison-caption">
    <strong>Left: MIS disabled &nbsp;|&nbsp; Right: MIS enabled</strong> — Sobol' sampler on both.
    Next-Event Estimation dramatically reduces noise on the lit portions of the statue and ground.
  </p>
</div>

### Combined effect (baseline vs full)

<div class="comparison-wrap comparison-block">
  <span class="label-left">Baseline</span>
  <span class="label-right">Full</span>
  <img-comparison-slider>
    <img slot="first"  src="../../assets/images/comparisons/04_statue_pcg_nomis.png" alt="OBJ Statue — PCG, no MIS">
    <img slot="second" src="../../assets/images/comparisons/04_statue_sobol_mis.png" alt="OBJ Statue — Sobol' + MIS">
  </img-comparison-slider>
  <p class="comparison-caption">
    <strong>Left: PCG + no MIS (worst-case baseline) &nbsp;|&nbsp; Right: Sobol' + MIS (production default)</strong>.
    The improvement is substantial — equivalent to rendering 8–15× more samples with the baseline.
  </p>
</div>

---

<div class="scene-header">
<h2>Scene 2 — Caustics Chapel</h2>
<p class="scene-meta">glass spheres · small bright area light · 512 SPP · indoor scene</p>
</div>

The hardest test: a small, very bright area light illuminating through refractive glass spheres
in a dark room.  Without MIS, most rays never find the light and the image is dominated by
fireflies. At 512 SPP the effect of each technique is still clearly visible.

### Sobol' effect (MIS disabled)

<div class="comparison-wrap comparison-block">
  <span class="label-left">PCG</span>
  <span class="label-right">Sobol'</span>
  <img-comparison-slider>
    <img slot="first"  src="../../assets/images/comparisons/06_caustics_pcg_nomis.png"   alt="Caustics — PCG, no MIS">
    <img slot="second" src="../../assets/images/comparisons/06_caustics_sobol_nomis.png" alt="Caustics — Sobol', no MIS">
  </img-comparison-slider>
  <p class="comparison-caption">
    <strong>Left: PCG &nbsp;|&nbsp; Right: Sobol'</strong> — MIS disabled on both (512 SPP).
    Sobol' produces more evenly distributed noise; the caustic light patches are less grainy.
  </p>
</div>

### MIS / NEE effect (Sobol' sampler)

<div class="comparison-wrap comparison-block">
  <span class="label-left">No MIS</span>
  <span class="label-right">MIS on</span>
  <img-comparison-slider>
    <img slot="first"  src="../../assets/images/comparisons/06_caustics_sobol_nomis.png" alt="Caustics — Sobol', no MIS">
    <img slot="second" src="../../assets/images/comparisons/06_caustics_sobol_mis.png"   alt="Caustics — Sobol', MIS on">
  </img-comparison-slider>
  <p class="comparison-caption">
    <strong>Left: No MIS &nbsp;|&nbsp; Right: MIS on</strong> — Sobol' sampler on both.
    MIS eliminates most of the fireflies and dramatically brightens the correctly lit diffuse walls.
    Direct light contributions which were rare without NEE are now sampled explicitly.
  </p>
</div>

### Combined effect (baseline vs full)

<div class="comparison-wrap comparison-block">
  <span class="label-left">Baseline</span>
  <span class="label-right">Full</span>
  <img-comparison-slider>
    <img slot="first"  src="../../assets/images/comparisons/06_caustics_pcg_nomis.png" alt="Caustics — PCG, no MIS">
    <img slot="second" src="../../assets/images/comparisons/06_caustics_sobol_mis.png" alt="Caustics — Sobol' + MIS">
  </img-comparison-slider>
  <p class="comparison-caption">
    <strong>Left: PCG + no MIS &nbsp;|&nbsp; Right: Sobol' + MIS</strong>.
    This is the most dramatic scene for both techniques combined: the wall illumination and caustic
    floor patterns are only visible with the full stack enabled.
  </p>
</div>

---

<div class="scene-header">
<h2>Scene 3 — Color Bleed Box</h2>
<p class="scene-meta">Cornell box · red / green / blue walls · area light · 64 SPP</p>
</div>

A classic Cornell-box variant with strongly coloured walls. Color bleeding — the indirect
bounce of coloured light onto neutral surfaces — is a purely diffuse effect that benefits
significantly from both NEE and quasi-random sampling.

### Sobol' effect (MIS disabled)

<div class="comparison-wrap comparison-block">
  <span class="label-left">PCG</span>
  <span class="label-right">Sobol'</span>
  <img-comparison-slider>
    <img slot="first"  src="../../assets/images/comparisons/09_colorbleed_pcg_nomis.png"   alt="Color Bleed — PCG, no MIS">
    <img slot="second" src="../../assets/images/comparisons/09_colorbleed_sobol_nomis.png" alt="Color Bleed — Sobol', no MIS">
  </img-comparison-slider>
  <p class="comparison-caption">
    <strong>Left: PCG &nbsp;|&nbsp; Right: Sobol'</strong> — MIS disabled.
    Sobol' distributes samples more evenly, giving smoother color gradients on the diffuse walls.
  </p>
</div>

### MIS / NEE effect (Sobol' sampler)

<div class="comparison-wrap comparison-block">
  <span class="label-left">No MIS</span>
  <span class="label-right">MIS on</span>
  <img-comparison-slider>
    <img slot="first"  src="../../assets/images/comparisons/09_colorbleed_sobol_nomis.png" alt="Color Bleed — Sobol', no MIS">
    <img slot="second" src="../../assets/images/comparisons/09_colorbleed_sobol_mis.png"   alt="Color Bleed — Sobol', MIS on">
  </img-comparison-slider>
  <p class="comparison-caption">
    <strong>Left: No MIS &nbsp;|&nbsp; Right: MIS on</strong> — Sobol' sampler.
    NEE resolves the direct illumination on the floor and ceiling cleanly.
    The soft shadow under the spheres and the colour bleeding on the far wall are both sharper.
  </p>
</div>

### Combined effect (baseline vs full)

<div class="comparison-wrap comparison-block">
  <span class="label-left">Baseline</span>
  <span class="label-right">Full</span>
  <img-comparison-slider>
    <img slot="first"  src="../../assets/images/comparisons/09_colorbleed_pcg_nomis.png" alt="Color Bleed — PCG, no MIS">
    <img slot="second" src="../../assets/images/comparisons/09_colorbleed_sobol_mis.png" alt="Color Bleed — Sobol' + MIS">
  </img-comparison-slider>
  <p class="comparison-caption">
    <strong>Left: PCG + no MIS &nbsp;|&nbsp; Right: Sobol' + MIS</strong>.
  </p>
</div>

---

<div class="scene-header">
<h2>Scene 4 — Default Scene (area-light only)</h2>
<p class="scene-meta">rough mirrors · glass sphere · Fibonacci-dot sphere · single area light · 64 SPP · ambient off</p>
</div>

The standard RayON default scene with ambient lighting disabled so that all illumination comes
from a single rectangular area light.  The mix of rough-mirror, glass, and diffuse surfaces
exercises all code paths: MIS helps diffuse surfaces; Sobol' helps specular highlights.

### Sobol' effect (MIS disabled)

<div class="comparison-wrap comparison-block">
  <span class="label-left">PCG</span>
  <span class="label-right">Sobol'</span>
  <img-comparison-slider>
    <img slot="first"  src="../../assets/images/comparisons/default_pcg_nomis.png"   alt="Default scene — PCG, no MIS">
    <img slot="second" src="../../assets/images/comparisons/default_sobol_nomis.png" alt="Default scene — Sobol', no MIS">
  </img-comparison-slider>
  <p class="comparison-caption">
    <strong>Left: PCG &nbsp;|&nbsp; Right: Sobol'</strong> — MIS disabled.
    Look at the unlit ground plane and the rough-mirror sphere: Sobol' produces finer, more
    uniform grain vs the clumpy PCG noise pattern.
  </p>
</div>

### MIS / NEE effect (Sobol' sampler)

<div class="comparison-wrap comparison-block">
  <span class="label-left">No MIS</span>
  <span class="label-right">MIS on</span>
  <img-comparison-slider>
    <img slot="first"  src="../../assets/images/comparisons/default_sobol_nomis.png" alt="Default scene — Sobol', no MIS">
    <img slot="second" src="../../assets/images/comparisons/default_sobol_mis.png"   alt="Default scene — Sobol', MIS on">
  </img-comparison-slider>
  <p class="comparison-caption">
    <strong>Left: No MIS &nbsp;|&nbsp; Right: MIS on</strong> — Sobol' on both.
    The shadow boundary under the glass sphere, the lit side of the coloured balls,
    and the highlight on the rough-mirror sphere all resolve much more cleanly with NEE.
  </p>
</div>

### Combined effect (baseline vs full)

<div class="comparison-wrap comparison-block">
  <span class="label-left">Baseline</span>
  <span class="label-right">Full</span>
  <img-comparison-slider>
    <img slot="first"  src="../../assets/images/comparisons/default_pcg_nomis.png" alt="Default scene — PCG, no MIS">
    <img slot="second" src="../../assets/images/comparisons/default_sobol_mis.png" alt="Default scene — Sobol' + MIS">
  </img-comparison-slider>
  <p class="comparison-caption">
    <strong>Left: PCG + no MIS (baseline) &nbsp;|&nbsp; Right: Sobol' + MIS (production default)</strong>.
    At 64 SPP the default configuration is already approaching 256+ SPP quality with the baseline.
  </p>
</div>

---

## Quantitative summary

| Technique | Dominant benefit | Cost |
|---|---|---|
| **Sobol' sampler** | Uniform noise, faster convergence everywhere | ~0% overhead (same ops, better coverage) |
| **MIS / NEE** | Massive firefly reduction; direct lighting resolved at very low SPP | +1 shadow ray per diffuse bounce (~25–40% more rays) |
| **Sobol' + MIS** | Best of both: low-discrepancy + direct sampling | Combined but non-additive overhead |

For the theoretical background, see:

- [Sobol' Quasi-Random Sampling](sobol-sampling.md) — direction vectors, Gray-code ordering, Owen scrambling
- [Multiple Importance Sampling](mis.md) — power heuristic, NEE, GPU implementation
