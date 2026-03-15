# Ray Tracing & Path Tracing

This page covers the mathematical foundations that every renderer in RayON is built on.
Understanding these concepts makes the rest of the code much more readable.

---

## The ray

Everything starts with a **ray** — a half-line defined by an origin point and a direction vector:

\[
\mathbf{r}(t) = \mathbf{o} + t\,\mathbf{d}, \quad t > 0
\]

| Symbol | Meaning |
|---|---|
| \(\mathbf{o}\) | Ray origin (camera position, or last bounce hit point) |
| \(\mathbf{d}\) | Ray direction (unit vector) |
| \(t\) | Parameter — how far along the ray we are |

Finding where a ray hits an object means solving for \(t\) in the intersection equations for
spheres, planes, triangles, etc.

```cpp
// From data_structures/ray.hpp
struct Ray {
    Vec3 origin;
    Vec3 direction;

    Vec3 at(double t) const { return origin + t * direction; }
};
```

---

## The rendering equation

The **rendering equation** (Kajiya, 1986) describes how light accumulates at a surface point:

\[
L_o(\mathbf{x}, \omega_o) = L_e(\mathbf{x}, \omega_o)
  + \int_{\Omega} f_r(\mathbf{x}, \omega_i, \omega_o)\,
    L_i(\mathbf{x}, \omega_i)\,
    (\omega_i \cdot \hat{n})\, d\omega_i
\]

| Symbol | Meaning |
|---|---|
| \(L_o\) | Outgoing radiance (what the camera "sees") |
| \(L_e\) | Emitted radiance (non-zero only for light sources) |
| \(f_r\) | BRDF — how the surface scatters light |
| \(L_i\) | Incoming radiance (recursive — depends on the scene) |
| \(\omega_i \cdot \hat{n}\) | Cosine foreshortening — Lambert's law |
| \(\Omega\) | Hemisphere of directions above the surface |

This integral **cannot be solved analytically** for arbitrary scenes, so we use Monte Carlo estimation.

---

## Monte Carlo path tracing

**Idea**: estimate the integral by averaging many random samples.

For a single pixel, we fire \(N\) rays. Each ray bounces through the scene, collecting radiance at
each interaction. The pixel colour is the average of all the paths:

\[
L_o \approx \frac{1}{N} \sum_{k=1}^{N} \frac{f_r(\omega_k) \cdot L_i(\omega_k) \cdot \cos\theta_k}{p(\omega_k)}
\]

where \(p(\omega_k)\) is the probability density of sampling direction \(\omega_k\)
(see [Hemisphere Sampling](sampling.md)).

### The bounce loop

```cpp
// Simplified inner loop (all renderers share this logic)
Vec3 throughput = Vec3(1.0); // accumulated attenuation
Vec3 result     = Vec3(0.0); // accumulated emission

for (int depth = 0; depth < MAX_DEPTH; ++depth) {
    HitRecord rec;
    if (!scene.hit(ray, rec)) {
        result += throughput * background_color;
        break;
    }

    // Emitted light from this surface
    result += throughput * rec.material.emitted();

    // Scatter the ray according to the material BRDF
    Ray scattered;
    Vec3 attenuation;
    if (!rec.material.scatter(ray, rec, attenuation, scattered))
        break; // ray absorbed — path ends

    throughput *= attenuation;
    ray = scattered;
}
return result;
```

Each iteration either:

1. **Misses all geometry** → adds background radiance, stops.
2. **Hits a light** → adds emission, path terminated (light doesn't scatter).
3. **Hits a surface** → multiplies throughput by the BRDF's attenuation, scatters the ray, continues.

---

## Why does noise decrease with more samples?

A Monte Carlo estimator converges at rate \(\mathcal{O}(1/\sqrt{N})\):

- **9 samples** → 3× noise reduction vs 1 sample
- **100 samples** → 10× reduction
- **10 000 samples** → 100× reduction

This is why interactive mode starts at 8 SPP (noisy but immediately responsive) and accumulates
progressively: the image *always* improves with time.

---

## Sample convergence

The same viewpoint rendered at successive sample counts, illustrating convergence:

<div class="progression-strip">
  <figure>
    <img src="../../assets/images/for_project/begin.png" alt="1 SPP">
    <figcaption>1 SPP</figcaption>
  </figure>
  <figure>
    <img src="../../assets/images/for_project/end_8s.png" alt="8 SPP">
    <figcaption>8 SPP</figcaption>
  </figure>
  <figure>
    <img src="../../assets/images/for_project/end_128s.png" alt="128 SPP">
    <figcaption>128 SPP</figcaption>
  </figure>
  <figure>
    <img src="../../assets/images/for_project/end_256s.png" alt="256 SPP">
    <figcaption>256 SPP</figcaption>
  </figure>
  <figure>
    <img src="../../assets/images/for_project/end_512s.png" alt="512 SPP">
    <figcaption>512 SPP</figcaption>
  </figure>
  <figure>
    <img src="../../assets/images/for_project/end_2048s.png" alt="2048 SPP">
    <figcaption>2048 SPP</figcaption>
  </figure>
</div>

The image at 1 SPP stores one random path per pixel — it is essentially pure noise.
At 8 SPP the shapes are recognisable. Accurate shadows and inter-reflections appear around 128–256 SPP.

---

## Russian roulette path termination

Carrying a path all the way to `MAX_DEPTH` bounces even when the throughput is near zero
wastes compute time. **Russian roulette** terminates paths probabilistically:

```cpp
// After each bounce, proportional to throughput brightness
float p = max(throughput.r, max(throughput.g, throughput.b));
if (curand_uniform(&rng) > p)
    break; // terminate — no bias introduced
throughput /= p; // compensate to stay unbiased
```

This is enabled from bounce 1 in RayON and has no effect on the mean of the estimator
(it only changes variance). The result is that cheap paths (low attenuation) are terminated
early, and the saved compute is reinvested into paths that matter.

---

## Anti-aliasing through jittered sampling

Each sample for a given pixel uses a different, randomly offset ray direction within the pixel's
solid angle. Averaging \(N\) such samples simultaneously:

1. Converges to the true pixel integral (anti-aliasing).
2. Produces smooth edges without an explicit AA pass.

```cpp
// Per-sample sub-pixel jitter
double u = (col + random_double()) / (image_width  - 1);
double v = (row + random_double()) / (image_height - 1);
Ray ray = camera.get_ray(u, v);
```

This is why increasing SPP both reduces noise *and* improves edge sharpness.
