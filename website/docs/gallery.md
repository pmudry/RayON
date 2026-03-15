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
        <img src="../assets/images/samples/gui_debug/indoor spheres.png" alt="Indoor scene with multiple reflective and diffuse spheres">
        <figcaption><strong>Indoor Spheres</strong> — assorted dielectric, metallic, and diffuse spheres in an enclosed room with area lighting.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/isc spheres.png" alt="Scene with spheres arranged on a reflective floor">
        <figcaption><strong>ISC Spheres</strong> — multi-material arrangement with strong inter-reflections between specular surfaces.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/bvh_testing.png" alt="Many spheres rendered">
        <figcaption><strong>Multiple spheres</strong> — scene, using for performance assessment in realtime..</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/golf.png" alt="Golf ball displacement mapping — dimpled surface under directional light">
        <figcaption><strong>Golf Ball</strong> — procedural displacement mapping creates the characteristic dimple pattern on a sphere. The BRDF captures specular highlights across the surface microstructure.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/large/rayon_4k_render.png" alt="Wide view of diverse material spheres on a brushed-metal ground plane — 4K master render">
        <figcaption><strong>4K Master Render</strong> — full default scene at 4K resolution. Glass, rough mirror, Lambertian, tinted metal, and light-emitting materials share the scene with SDF shapes in the background.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/for_project/sampling/aniso_512spp.png" alt="Anisotropic metals scene at 512 SPP — directional highlight streaks fully resolved">
        <figcaption><strong>Anisotropic Metals — 512 SPP</strong> — directional highlight streaks fully resolved across all roughness and anisotropy levels. Rendered with the CUDA path tracer.</figcaption>
      </figure>
    </div>

=== "Materials"

    <div class="img-grid cols-2">
      <figure>
        <img src="../assets/images/samples/metals shine 2.png" alt="Tinted rough mirrors — gold, copper, and steel preset tints">
        <figcaption><strong>Tinted Rough Mirrors</strong> — gold, copper, and steel preset tints. Roughness controls how blurry the reflections appear (0 = perfect mirror, 1 ≈ diffuse).</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/dielectric metsals.png" alt="Glass dielectric spheres with refractive caustics next to metallic surfaces">
        <figcaption><strong>Dielectrics &amp; Metals</strong> — glass spheres use Snell's law for refraction and Schlick's approximation for the reflection/transmission ratio at grazing angles.</figcaption>
      </figure>
      
      <figure>
        <img src="../assets/images/samples/thin_film_shader.png" alt="Soap-bubble like thin film shading">
        <figcaption><strong>Thin-film rendering</strong> — mimics oil stains or soap bubbles.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/plastic_shading.png" alt="Plastic shading — diffuse base with specular highlight layer">
        <figcaption><strong>Plastic Shading</strong> — two-layer model: a Lambertian base coat with a specular clearcoat layer on top. The highlight is view-dependent.</figcaption>
      </figure>
    </div>

=== "Visualisation & Debug"

    <div class="img-grid cols-2">
      <figure>
        <img src="../assets/images/samples/gui_debug/normals.png" alt="Normal visualisation mode — surfaces coloured by surface normal direction">
        <figcaption><strong>Normal Visualisation</strong> — the <code>ShowNormals</code> material maps surface normals directly to RGB: <em>R=X, G=Y, B=Z</em>. Useful for debugging geometry orientation and smooth normals on mesh imports.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/gui_debug/statue.png" alt="Interactive SDL2 window showing real-time accumulative path tracing at 100 Hz">
        <figcaption><strong>Interactive Mode</strong> — SDL2 window at ~100 Hz with Dear ImGui overlaid. The sample counter and convergence indicator are visible in the top-right panel.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/obj_loading.png" alt="OBJ mesh loading test — low-poly model with triangulated faces">
        <figcaption><strong>OBJ Mesh Loading</strong> — imported triangle mesh, rendered with smooth interpolated normals from the <code>.obj</code> file's vertex normal list.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/dev/dof.png" alt="Depth of field with bokeh blur">
        <figcaption><strong>Depth of Field</strong> — aperture and focus distance are adjustable at runtime via ImGui sliders in interactive mode. The lens model is a thin lens approximation.</figcaption>
      </figure>
    </div>

=== "Progressive Quality"

    <p>Anisotropic metals scene rendered at increasing sample counts. Every doubling halves Monte Carlo noise — error
    decays as \(\mathcal{O}(1/\sqrt{N})\). The anisotropic specular lobes are particularly sensitive to sample count.</p>

    <div class="img-grid cols-2">
      <figure>
        <img src="../assets/images/for_project/sampling/aniso_4spp.png" alt="4 samples per pixel — heavy noise across specular lobes">
        <figcaption><strong>4 SPP</strong> — specular highlights are buried in noise; anisotropic streaks invisible.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/for_project/sampling/aniso_16spp.png" alt="16 samples per pixel — rough shapes visible">
        <figcaption><strong>16 SPP</strong> — sphere shapes emerge; directional highlights faintly visible.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/for_project/sampling/aniso_32spp.png" alt="32 samples per pixel">
        <figcaption><strong>32 SPP</strong> — anisotropic streaks begin to form; still noticeably noisy.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/for_project/sampling/aniso_64spp.png" alt="64 samples per pixel">
        <figcaption><strong>64 SPP</strong> — usable quality; highlight directionality clear on low-roughness spheres.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/for_project/sampling/aniso_128spp.png" alt="128 samples per pixel — clean highlights">
        <figcaption><strong>128 SPP</strong> — clean for most materials; high-anisotropy spheres still show slight graininess.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/for_project/sampling/aniso_512spp.png" alt="512 samples per pixel — near-converged">
        <figcaption><strong>512 SPP</strong> — near-converged; anisotropic lobes fully resolved across all roughness levels.</figcaption>
      </figure>
    </div>
