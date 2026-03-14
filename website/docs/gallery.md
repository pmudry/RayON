# Gallery

A curated collection of renders from RayON's built-in scenes, showcasing the range of materials,
lighting effects, and geometric complexity the renderer supports.

---

=== "Scenes"

    <div class="img-grid cols-2">
      <figure>
        <img src="../assets/images/samples/cornell.png" alt="Cornell box — colour bleeding between red and green walls onto the ceiling and white floor spheres">
        <figcaption><strong>Cornell Box</strong> — classic test scene. Coloured diffuse walls create colour bleeding on nearby surfaces. Soft shadow from an area light overhead.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/indoor spheres.png" alt="Indoor scene with multiple reflective and diffuse spheres">
        <figcaption><strong>Indoor Spheres</strong> — assorted dielectric, metallic, and diffuse spheres in an enclosed room with area lighting.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/isc spheres.png" alt="Scene with spheres arranged on a reflective floor">
        <figcaption><strong>ISC Spheres</strong> — multi-material arrangement with strong inter-reflections between specular surfaces.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/golf.png" alt="Golf ball displacement mapping — dimpled surface under directional light">
        <figcaption><strong>Golf Ball</strong> — procedural displacement mapping creates the characteristic dimple pattern on a sphere. The BRDF captures specular highlights across the surface microstructure.</figcaption>
      </figure>
    </div>

    <div class="img-grid cols-2">
      <figure>
        <img src="../assets/images/samples/rayon_4k_render.png" alt="Wide view of diverse material spheres on a brushed-metal ground plane — 4K master render">
        <figcaption><strong>4K Master Render</strong> — full default scene at 4K resolution. Glass, rough mirror, Lambertian, tinted metal, and light-emitting materials share the scene with SDF shapes in the background.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/for_project/end_2048s_brushed_ground.png" alt="Default scene with brushed-metal ground at 2048 SPP">
        <figcaption><strong>Brushed-Metal Ground</strong> — anisotropic reflection on the floor plane from 2048 accumulated samples. Long-exposure look with near-zero noise.</figcaption>
      </figure>
    </div>

=== "Materials"

    <div class="img-grid cols-2">
      <figure>
        <img src="../assets/images/samples/metals shine.png" alt="High-roughness metallic spheres showing warm specular highlights">
        <figcaption><strong>Tinted Rough Mirrors</strong> — gold, copper, and steel preset tints. Roughness controls how blurry the reflections appear (0 = perfect mirror, 1 ≈ diffuse).</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/metals shine 2.png" alt="Alternate metallic render with different lighting angle">
        <figcaption><strong>Metals — Alternate View</strong> — the same scene relit from a different angle. The highlight shape reveals the roughness distribution on each sphere.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/dielectric metsals.png" alt="Glass dielectric spheres with refractive caustics next to metallic surfaces">
        <figcaption><strong>Dielectrics &amp; Metals</strong> — glass spheres use Snell's law for refraction and Schlick's approximation for the reflection/transmission ratio at grazing angles.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/dev/output_rough_mirror.png" alt="Rough mirror with configurable roughness parameter">
        <figcaption><strong>Rough Mirror Development Render</strong> — early validation render showing the fuzzy reflection model. The reflected image becomes progressively blurred as roughness increases from left to right.</figcaption>
      </figure>
    </div>

    <div class="img-grid cols-2">
      <figure>
        <img src="../assets/images/dev/new_lambertian.png" alt="Improved cosine-weighted Lambertian scattering">
        <figcaption><strong>Cosine-weighted Lambertian</strong> — improved hemisphere sampling (see <a href="how-it-works/sampling.md">Sampling</a>). Noise is concentrated in shadow rather than spread uniformly.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/plastic_shading.png" alt="Plastic shading — diffuse base with specular highlight layer">
        <figcaption><strong>Plastic Shading</strong> — two-layer model: a Lambertian base coat with a specular clearcoat layer on top. The highlight is view-dependent.</figcaption>
      </figure>
    </div>

=== "Visualisation & Debug"

    <div class="img-grid cols-2">
      <figure>
        <img src="../assets/images/samples/normals.png" alt="Normal visualisation mode — surfaces coloured by surface normal direction">
        <figcaption><strong>Normal Visualisation</strong> — the <code>ShowNormals</code> material maps surface normals directly to RGB: <em>R=X, G=Y, B=Z</em>. Useful for debugging geometry orientation and smooth normals on mesh imports.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/obj_loading.png" alt="OBJ mesh loading test — low-poly model with triangulated faces">
        <figcaption><strong>OBJ Mesh Loading</strong> — imported triangle mesh, rendered with smooth interpolated normals from the <code>.obj</code> file's vertex normal list.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/dev/dof.png" alt="Depth of field with bokeh blur">
        <figcaption><strong>Depth of Field</strong> — aperture and focus distance are adjustable at runtime via ImGui sliders in interactive mode. The lens model is a thin lens approximation.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/real_time_raytrace.png" alt="Interactive SDL2 window showing real-time accumulative path tracing at 100 Hz">
        <figcaption><strong>Interactive Mode</strong> — SDL2 window at ~100 Hz with Dear ImGui overlaid. The sample counter and convergence indicator are visible in the top-right panel.</figcaption>
      </figure>
    </div>

=== "Progressive Quality"

    <p>Same viewpoint accumulated from 1 to 2048 samples per pixel. Every doubling halves Monte Carlo noise — error
    decays as \(\mathcal{O}(1/\sqrt{N})\).</p>

    <div class="img-grid cols-2">
      <figure>
        <img src="../assets/images/for_project/begin.png" alt="1 sample per pixel — heavily noisy">
        <figcaption>1 SPP — pure noise; each pixel stores a single random path.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/for_project/end_8s.png" alt="8 samples — shapes visible through noise">
        <figcaption>8 SPP — shapes emerge but reflections are still very noisy.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/for_project/end_128s.png" alt="128 samples — mostly clean">
        <figcaption>128 SPP — good quality for interactive use.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/for_project/end_256s.png" alt="256 samples">
        <figcaption>256 SPP — sharp shadows, clean glass caustics.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/for_project/end_512s.png" alt="512 samples">
        <figcaption>512 SPP — wall colour bleeding fully resolved.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/for_project/end_2048s.png" alt="2048 samples — near-converged">
        <figcaption>2048 SPP — near-converged; remaining noise is sub-pixel.</figcaption>
      </figure>
    </div>
