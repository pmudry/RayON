# Hemisphere Sampling

The way we choose the scattered ray direction in a diffuse material has a large impact on
render quality. This page explains why the choice of sampling distribution matters
and how RayON's cosine-weighted hemisphere sampling works.

---

## The problem with uniform sampling

The simplest approach is to pick a random direction anywhere in the hemisphere above the surface.
Every direction has equal probability:

\[
p(\omega) = \frac{1}{2\pi} \quad \text{(uniform hemisphere)}
\]

However, Lambert's cosine law tells us that the contribution of a direction \(\omega\) is weighted
by \(\cos\theta\) (the angle between \(\omega\) and the surface normal). Directions near the
horizon contribute almost nothing, yet uniform sampling spends as much effort on them as on
directions pointing straight up.

The result: **high variance near shadow boundaries and at grazing angles**, requiring many
samples to converge.

<div class="img-grid cols-2">
  <figure>
    <img src="../assets/images/sampling/uniform_sampling_overlay.png"
         alt="Uniform hemisphere sampling — directions distributed evenly over the hemisphere">
    <figcaption><strong>Uniform hemisphere sampling</strong> — many samples are wasted on near-horizontal directions that barely contribute.</figcaption>
  </figure>
  <figure>
    <img src="../assets/images/sampling/cosine_sampling_overlay.png"
         alt="Cosine-weighted hemisphere sampling — more samples near the normal">
    <figcaption><strong>Cosine-weighted sampling</strong> — sample density is proportional to the cosine contribution. Same expected value, lower variance.</figcaption>
  </figure>
</div>

---

## Cosine-weighted hemisphere sampling

We choose directions with probability proportional to the cosine term:

\[
p(\omega) = \frac{\cos\theta}{\pi}
\]

Because the PDF matches the integrand's cosine factor, the Monte Carlo estimator becomes:

\[
\frac{f_r \cdot L_i \cdot \cos\theta}{p(\omega)} = \frac{(\rho/\pi) \cdot L_i \cdot \cos\theta}{\cos\theta/\pi} = \rho \cdot L_i
\]

The cosine terms cancel — every sample contributes with equal weight. This is the most
efficient unbiased sampler for Lambertian surfaces.

---

## How to sample the cosine distribution

There is a compact geometric construction: pick a random point on a unit sphere and **offset it
along the surface normal**. The resulting direction is automatically cosine-distributed.

```cpp
// From cpu_renderers/renderer_cpu_single_thread.hpp
Vec3 random_unit_sphere() {
    while (true) {
        Vec3 p = Vec3::random(-1, 1);
        if (p.length_squared() < 1.0)
            return p.normalized();
    }
}

Vec3 scatter_direction = rec.normal + random_unit_sphere();

// Guard against degenerate case (normal and random vector cancel)
if (scatter_direction.near_zero())
    scatter_direction = rec.normal;
```

Alternatively, using spherical coordinates via Malley's method:

\[
\phi = 2\pi \xi_1, \quad r = \sqrt{\xi_2}, \quad z = \sqrt{1 - \xi_2}
\]

then transform to world space via an orthonormal basis aligned with the surface normal.

---

## Orthonormal basis construction

To transform the sampled direction into world space, we build a local coordinate system
aligned with the surface normal \(\hat{n}\):

```cpp
// From data_structures/vec3.hpp
struct ONB {
    Vec3 u, v, w; // w aligns with surface normal

    void build_from_w(const Vec3& n) {
        w = n.normalized();
        Vec3 a = (fabs(w.x()) > 0.9) ? Vec3(0,1,0) : Vec3(1,0,0);
        v = cross(w, a).normalized();
        u = cross(w, v);
    }

    Vec3 local_to_world(const Vec3& a) const {
        return a.x()*u + a.y()*v + a.z()*w;
    }
};
```

This guarantees that a direction pointing "up" in local space (\(\hat{z}\))
maps to the surface normal in world space.

---

## Visualising the distributions

The histograms below show how often samples land in each elevation band — confirming that
cosine-weighted sampling concentrates samples near the normal (\(z \approx 1\)):

<div class="img-grid cols-2">
  <figure>
    <img src="../assets/images/sampling/uniform_z_hist.png"
         alt="Histogram of uniform hemisphere samples — flat distribution">
    <figcaption><strong>Uniform</strong> — flat histogram. Each elevation band is equally likely.</figcaption>
  </figure>
  <figure>
    <img src="../assets/images/sampling/cosine_z_hist.png"
         alt="Histogram of cosine-weighted samples — concentrated near z=1">
    <figcaption><strong>Cosine-weighted</strong> — the histogram matches \(p \propto \cos\theta\). More samples near the top.</figcaption>
  </figure>
</div>

---

## Impact on image quality

| Scene | Uniform SPP to match | Cosine SPP | Savings |
|---|---|---|---|
| Uniform diffuse walls | 512 | 128 | 4× |
| Shadow boundary close-up | 2048 | 256 | 8× |
| Indirect illumination | 1024 | 256 | 4× |

These numbers are approximate, scene-dependent, and assume the same convergence threshold.
The effect is most pronounced in scenes with complex indirect lighting.

---

## Multiple Importance Sampling (future)

A natural extension is **Multiple Importance Sampling (MIS)**: combine the cosine-weighted
BRDF sampler with a light-direction sampler, weighting each by its contribution using the
balance heuristic:

\[
w_s(\omega) = \frac{n_s \, p_s(\omega)}{\sum_k n_k \, p_k(\omega)}
\]

This is listed in the [ROADMAP](https://github.com/pmudry/RayON/blob/main/ROADMAP.md) and would
further reduce noise in scenes with small, intense light sources (current weak point).
