# Gallery

Renders from the bundled scenes, grouped by what they show. Every image can be reproduced from
`resources/scenes/` with the offline modes (`-m 4` or `-m 2`).

---

=== "Volumes & SDF"

    <div class="img-grid cols-2">
      <figure>
        <img src="../assets/images/cloud_bunny.png" alt="The Stanford bunny as a cumulus cloud on a ground plane">
        <figcaption><strong>Cloud bunny</strong> — <code>17_cloud_bunny.yaml</code>. The volume material with procedural shape noise, a hidden sun and volumetric shadow rays. OptiX, 512 spp, 720p.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/vdb_bunny.png" alt="OpenVDB bunny cloud density grid">
        <figcaption><strong>VDB bunny</strong> — <code>19_vdb_bunny.yaml</code>. The OpenVDB <code>bunny_cloud</code> grid, 289×286×219 cells, read straight into the 3D density texture. OptiX, 512 spp, 720p.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/full_bunny.png" alt="The VDB bunny cloud at 2048 samples per pixel in 1080p">
        <figcaption><strong>VDB bunny, 2048 spp, 1080p</strong> — the same scene left to converge.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/sdf_gallery.png" alt="Ten ray-marched SDF primitives in a studio room">
        <figcaption><strong>SDF gallery</strong> — <code>18_sdf_shapes.yaml</code>. Mandelbulb, death star, cut hollow sphere, torus, glass octahedron, pyramid, capsule, box, cylinder and sphere, sphere-traced on the GPU. OptiX, 384 spp, 720p.</figcaption>
      </figure>
    </div>

=== "Scenes"

    <div class="img-grid cols-2">
      <figure>
        <img src="../assets/images/samples/cornell.png" alt="Cornell box with colour bleeding between the coloured walls and the spheres">
        <figcaption><strong>Cornell box</strong> — <code>09_color_bleed_box.yaml</code>. Coloured diffuse walls bleed onto the neighbouring surfaces; the area light overhead gives soft shadows.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/gui_debug/indoor spheres.png" alt="Indoor scene with reflective and diffuse spheres">
        <figcaption><strong>Indoor spheres</strong> — dielectric, metallic and diffuse spheres in an enclosed room with area lighting.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/isc_spheres.png" alt="Spheres on a reflective floor">
        <figcaption><strong>ISC spheres</strong> — the five colours of the ISC programme's majors, with strong inter-reflections between specular surfaces.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/bvh_testing.png" alt="Many spheres in a grid">
        <figcaption><strong>BVH test scene</strong> — <code>01_bvh_test_scene.yaml</code>, 150 spheres used to measure the acceleration structure.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/golf.png" alt="Golf ball with procedural displacement">
        <figcaption><strong>Golf ball</strong> — a displaced sphere: the dimple pattern is procedural, and its count, radius and depth are live sliders in the GUI.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/for_project/sampling/aniso_512spp.png" alt="Anisotropic metals scene at 512 spp">
        <figcaption><strong>Anisotropic metals</strong> — <code>01_anisotropic_metals_test.yaml</code> at 512 spp: anisotropy along the rows, roughness along the columns.</figcaption>
      </figure>
    </div>

=== "Materials"

    <div class="img-grid cols-2">
      <figure>
        <img src="../assets/images/samples/metals%20shine%202.png" alt="Tinted rough mirrors: gold, copper and steel">
        <figcaption><strong>Tinted rough mirrors</strong> — gold, copper and steel tints. The roughness blurs the reflection, from a perfect mirror at 0 towards diffuse at 1.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/dielectric%20metsals.png" alt="Glass spheres with caustics next to metallic surfaces">
        <figcaption><strong>Dielectrics and metals</strong> — glass follows Snell's law with Schlick's Fresnel approximation; the caustic under the sphere emerges from the path tracing alone.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/thin_film_shader.png" alt="Soap-bubble thin-film shading">
        <figcaption><strong>Thin film</strong> — <code>11_soap_bubbles.yaml</code>. The Airy interference formula at several film thicknesses sweeps through the spectrum.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/plastic_shading.png" alt="Clear coat over a diffuse base on the dragon">
        <figcaption><strong>Clear coat</strong> — a Lambertian base under a smooth dielectric coat; the highlight is view-dependent.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/textures/uv_mapping.png" alt="UV textured cube, sphere and plane">
        <figcaption><strong>UV texture mapping</strong> — <code>13_texture_test1.yaml</code>. A grid texture through MTL files, interpolated with barycentric coordinates.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/textures/texture_orig.png" alt="Blender export with multi-material textures">
        <figcaption><strong>MTL-driven textures</strong> — <code>14_texture_test2.yaml</code>. Each group of a Blender export picks its texture from the <code>.mtl</code> file.</figcaption>
      </figure>
    </div>

=== "Visualisation & Debug"

    <div class="img-grid cols-2">
      <figure>
        <img src="../assets/images/samples/gui_debug/normals.png" alt="Normal visualisation mode">
        <figcaption><strong>Normals</strong> — key <b>N</b> maps the surface normal to RGB (R = X, G = Y, B = Z). The smooth interpolation hides the face boundaries of the mesh.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/gui_debug/statue.png" alt="Interactive window with the Dear ImGui panel">
        <figcaption><strong>Interactive mode</strong> — the SDL2 window with the Dear ImGui panel: sample counter, convergence and the live sliders.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/samples/obj_loading.png" alt="OBJ mesh with smooth normals">
        <figcaption><strong>OBJ loading</strong> — an imported mesh rendered with the vertex normals of the file.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/dev/dof.png" alt="Depth of field with bokeh blur">
        <figcaption><strong>Depth of field</strong> — thin-lens aperture and focus distance, adjustable live in the <em>Camera Settings</em> section.</figcaption>
      </figure>
    </div>

=== "Progressive Quality"

    <p>The anisotropic metals scene at increasing sample counts. Monte Carlo noise decays as
    \(\mathcal{O}(1/\sqrt{N})\): every doubling of the samples removes about 30 % of the noise.
    The anisotropic highlights are the slowest to resolve.</p>

    <div class="img-grid cols-2">
      <figure>
        <img src="../assets/images/for_project/sampling/aniso_4spp.png" alt="4 samples per pixel">
        <figcaption><strong>4 spp</strong> — the highlights are buried in noise.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/for_project/sampling/aniso_16spp.png" alt="16 samples per pixel">
        <figcaption><strong>16 spp</strong> — shapes emerge, the streaks are faintly visible.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/for_project/sampling/aniso_32spp.png" alt="32 samples per pixel">
        <figcaption><strong>32 spp</strong> — the anisotropic streaks form.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/for_project/sampling/aniso_64spp.png" alt="64 samples per pixel">
        <figcaption><strong>64 spp</strong> — usable; the direction of the highlights is clear on the smooth spheres.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/for_project/sampling/aniso_128spp.png" alt="128 samples per pixel">
        <figcaption><strong>128 spp</strong> — clean for most materials.</figcaption>
      </figure>
      <figure>
        <img src="../assets/images/for_project/sampling/aniso_512spp.png" alt="512 samples per pixel">
        <figcaption><strong>512 spp</strong> — near convergence.</figcaption>
      </figure>
    </div>

=== "HDR Skies"

    <p>Scenes lit by equirectangular HDR environments from <a href="https://polyhaven.com">Poly Haven</a>
    (CC0). Six environments are downloaded by <code>bash resources/hdri/download_hdri.sh 8k</code>,
    see <a href="../features/hdr-sky/">HDR Sky Dome</a>.</p>

    <div class="img-grid cols-3">
      <img src="../assets/images/samples/hdri/hdri_0.png" alt="HDR sky render">
      <img src="../assets/images/samples/hdri/hdri_1.png" alt="HDR sky render">
      <img src="../assets/images/samples/hdri/hdri_3.png" alt="HDR sky render">
      <img src="../assets/images/samples/hdri/hdri_4.png" alt="HDR sky render">
      <img src="../assets/images/samples/hdri/hdri_5.png" alt="HDR sky render">
      <img src="../assets/images/samples/hdri/hdri_6.png" alt="HDR sky render">
      <img src="../assets/images/samples/hdri/hdri_7.png" alt="HDR sky render">
      <img src="../assets/images/samples/hdri/hdri_8.png" alt="HDR sky render">
      <img src="../assets/images/samples/hdri/hdri_9.png" alt="HDR sky render">
    </div>
