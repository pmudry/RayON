/**
 * @class RendererOptiXProgressive
 * @brief Interactive SDL renderer with progressive sample accumulation using OptiX hardware RT cores.
 *
 * Mirrors RendererCUDAProgressive but drives OptiX instead of the CUDA path-tracer.
 * All SDL/GUI/camera-control infrastructure is shared verbatim; only renderBatch()
 * and device-memory management differ.
 *
 * Key differences from the CUDA progressive renderer:
 *  - No d_rand_states / d_accum_buffer / gpu_scene pointers — OptiX manages its own device memory.
 *  - Scene and BVH are built once via optixRendererBuildScene() and persist in g_state.
 *  - Camera change reset → optixRendererResetAccum() instead of freeing/reallocating CUDA buffers.
 *  - Per-frame render → optixRendererLaunch() + optixRendererDownloadAccum().
 *  - light_intensity, light_falloff and background_intensity are launch parameters (live);
 *    metal_fuzziness / glass_refraction_index are passed per launch as multipliers too.
 *  - While the camera moves, quality is traded for framerate (MotionQualitySettings): reduced
 *    resolution (upscaled for display), one sample per frame, PCG sampler, capped depth.
 *  - Converged-pixel skipping (adaptive sampling) is implemented on the OptiX side with a
 *    per-pixel sample-count buffer; the GUI heatmap and convergence readout use it.
 */
#pragma once

#if defined(SDL2_FOUND) && defined(OPTIX_FOUND)

#include <SDL.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "imgui.h"
#include "render/render_utils.hpp"
#include "render/renderer_interface.hpp"
#include "scene_builder.hpp"
#include "scene_factory.hpp"
#include "sdl_gui_controls.hpp"
#include "sdl_gui_handler.hpp"
#include "hdr_env_cache.hpp"

// Forward declarations of OptiX host functions (implemented in optix/optix_renderer.cu)
extern "C"
{
   void optixRendererInit();
   void optixRendererBuildScene(const Scene::SceneDescription &scene);
   void optixRendererResetAccum(int width, int height);
   unsigned long long optixRendererLaunch(int width, int height, int num_materials, int samples_to_add,
                                          int total_samples_so_far, int max_depth, float cam_cx, float cam_cy,
                                          float cam_cz, float p00x, float p00y, float p00z, float dux, float duy,
                                          float duz, float dvx, float dvy, float dvz, float cam_ux, float cam_uy,
                                          float cam_uz, float cam_vx, float cam_vy, float cam_vz,
                                          float bg_intensity, bool dof_enabled, float dof_aperture,
                                          float dof_focus_dist, float light_intensity, float metal_fuzziness,
                                          float glass_ior_multiplier);
   void optixRendererDownloadAccum(float *host_accum_buffer, int width, int height);
   void optixRendererConvertAccumToDisplay(unsigned char *display_image, int width, int height,
                                           int channels, int num_samples, float gamma, bool despeckle,
                                           float despeckle_ratio, float despeckle_floor, bool denoise,
                                           float denoise_blend);
   float optixRendererLastDenoiseMs();
   void optixRendererCleanup();
   void optixRendererSetGolfDimples(int count, float radius, float depth);
   void optixRendererUpdateMaterial(int idx, const Scene::MaterialDesc &m);
   bool optixRendererUploadHdrEnv(const float *rgba_data, int w, int h);
   bool optixRendererUploadHdrEnvHalf(const uint16_t *rgba16, int w, int h);
   void optixRendererClearHdrEnv();
   void setOptiXSobolSampler(bool use_sobol);
   void setOptiXBackgroundColor(float r, float g, float b);
   void setOptiXMISEnabled(bool enabled);
   void setOptiXFireflyClampCeiling(float ceiling);
   void setOptiXNEEStride(int stride);
   void setOptiXLightFalloff(float falloff);
   void setOptiXIndirectClamp(float ceiling);
   void setOptiXAdaptiveSampling(bool enabled, int min_samples, float threshold);
   bool optixRendererHandoverPrior(int new_width, int new_height, float strength);
   int  optixRendererCountConvergedPixels();
   void optixRendererRenderSampleHeatmap(unsigned char *display_image, int width, int height, int channels,
                                         int max_samples_for_scale);
}

class RendererOptiXProgressive : public IRenderer
{
 public:
   struct Settings
   {
      int samples_per_batch = constants::INTERACTIVE_SAMPLES_PER_BATCH;
      bool samples_per_batch_forced = false; ///< set on the command line: scenes may not override it
      bool auto_accumulate = true;
      bool adaptive_sampling = true; ///< converged-pixel skipping
      bool hdr_cache = true; ///< use .hdrcache sidecar to speed up repeated HDR loads
      std::string sky;       ///< --sky: HDR map to start with (part of its file name), empty = gradient
      GuiTheme theme = GuiTheme::NORD;
      bool mis_enabled           = true;
      bool motion_gate_mis       = false;
      int  nee_stride            = 1;
      MotionQualitySettings motion_quality{}; ///< quality reductions while the camera moves
   };

   RendererOptiXProgressive() = default;
   explicit RendererOptiXProgressive(Settings settings) : settings_(settings) {}

   void setSettings(const Settings &settings) { settings_ = settings; }

   void render(const RenderRequest &request, RenderContext &context) override
   {
      int samples_per_batch = settings_.samples_per_batch;
      bool auto_accumulate = settings_.auto_accumulate;
      const bool hdr_cache = settings_.hdr_cache;

      auto &camera = request.camera;
      auto &scene = request.scene;
      RenderTargetView target = request.target;

      Point3 &look_from = camera.look_from;
      Point3 &look_at = camera.look_at;
      Vec3 &vup = camera.vup;
      CameraFrame frame = camera.buildFrame();
      Vec3 basis_w = frame.w;

      const int image_width = target.width;
      const int image_height = target.height;
      const int image_channels = target.channels;

      auto refreshCameraFrame = [&]()
      {
         camera.updateFrame();
         frame = camera.buildFrame();
         basis_w = frame.w;
      };

      refreshCameraFrame();

      // Initialize GUI
      SDLGuiHandler gui(target.width, target.height, settings_.theme, "OptiX");
      if (!gui.initialize())
         return;
      int max_samples = camera.samples_per_pixel;

      // Initialize camera controls
      CameraControlHandler camera_control;
      camera_control.initializeCameraControls(look_from, look_at);

      // Ray-tracing state
      bool running = true;
      bool camera_changed = true;
      bool accumulation_enabled = auto_accumulate;
      int current_samples = 0;
      float gamma = 2.0f;
      float light_intensity = scene.light_intensity;        // UI only — baked into SBT at build time
      float light_falloff   = scene.light_falloff;          // distance exponent of the lights' reach
      float indirect_clamp  = scene.indirect_clamp;         // ceiling on light gathered after the first diffuse bounce
      float background_intensity = scene.background_intensity;
      float metal_fuzziness = 1.0f;        // UI only — baked into SBT at build time
      float glass_refraction_index = 1.5f; // UI only — baked into SBT at build time
      bool dof_enabled = false;
      float dof_aperture = 0.1f;
      float dof_focus_distance = 10.0f;
      float cam_fov_ui = static_cast<float>(camera.vfov);
      bool        save_camera_request = false; // GUI button: write the camera into the scene YAML
      std::string camera_save_status;            // feedback shown under the button for a few seconds
      auto        camera_save_status_time = std::chrono::steady_clock::now();
      bool needs_rerender = false;
      bool force_immediate_render = false;
      float samples_per_batch_float = static_cast<float>(samples_per_batch);
      // A scene may ask for its own batch size (settings.samples_per_batch) unless the command
      // line fixed one; the renderer default is used otherwise
      auto sceneBatchSize = [&](const Scene::SceneDescription &sd)
      {
         return (sd.samples_per_batch > 0 && !settings_.samples_per_batch_forced) ? sd.samples_per_batch
                                                                                 : settings_.samples_per_batch;
      };
      samples_per_batch_float = static_cast<float>(sceneBatchSize(scene));
      // A scene may also ask for its own adaptive sampling threshold (settings.adaptive_threshold)
      auto sceneAdaptiveThreshold = [](const Scene::SceneDescription &sd)
      { return sd.adaptive_threshold > 0.0f ? sd.adaptive_threshold : 0.01f; }; // 0.01 ~ 2.5/255
      float current_sps = 0.0f;
      float current_ms_per_sample = 0.0f;
      float current_fps = 0.0f;

      // MIS / NEE options
      bool mis_enabled           = settings_.mis_enabled;
      bool motion_gate_mis       = settings_.motion_gate_mis;
      int  nee_stride            = settings_.nee_stride;
      bool use_sobol             = true; // default: Sobol sampler
      MotionQualitySettings motion_quality = settings_.motion_quality;
      bool  motion_despeckle       = true;                             // display-time despeckle of motion frames
      float motion_despeckle_ratio = constants::MOTION_DESPECKLE_RATIO; // vs the mean of the 8 neighbours

      // Motion detection
      bool is_camera_moving = false;
      int  stage_index      = 0; // resolution stage after a motion (index into the stage fractions)
      // Frame-rate regulator state (motion frames only)
      float dyn_fraction        = 0.75f; // render fraction currently used while moving
      int   dyn_spp             = 0;     // samples per frame currently used while moving (0 = unset)
      float motion_k_ms         = 0.0f;  // smoothed render cost in ms per pixel-sample (motion frames)
      float motion_d_ms         = 0.0f;  // smoothed denoiser cost in ms per pixel (motion frames)
      float last_denoise_ms     = 0.0f;  // denoiser time of the previous frame (0 when it was not denoised)
      float motion_overhead_ms  = 0.0f;  // smoothed (frame - render) time: GUI, upload, present
      float last_render_ms      = 0.0f;  // render time of the previous frame
      int   last_motion_pixels  = 0;     // pixels and samples of the previous motion frame (for k)
      int   last_motion_spp     = 0;
      bool  last_frame_motion   = false; // the previous frame was a motion frame (its timing is usable)
      int   headroom_frames     = 0;     // consecutive motion frames with spare time
      auto  last_raise_time     = std::chrono::high_resolution_clock::now() - std::chrono::seconds(10);
      auto  raise_block_until   = last_raise_time;
      auto last_camera_change_time = std::chrono::high_resolution_clock::now();
      auto stage_start_time        = last_camera_change_time; // when the current render config was applied

      // Overlay / visualization state (mirrors CUDA renderer)
      int visualization_mode = static_cast<int>(VisualizationMode::NORMAL);
      bool show_spps_counter = true;
      bool show_heatmap = false;
      bool adaptive_sampling_enabled = settings_.adaptive_sampling;
      const int min_adaptive_samples = 64;  // Don't check convergence before this many samples
      float adaptive_threshold = sceneAdaptiveThreshold(scene); // Std. error of the pixel mean in display space
      const float ADAPTIVE_STOP_PCT = 99.9f; // Stop when this share of pixels has converged
      float convergence_pct = 0.0f;
      int frames_since_convergence_check = 0;

      // Per-frame render configuration. A change of resolution, sampler or depth between frames
      // invalidates the accumulation (such samples must not be mixed). Adaptive sampling and MIS
      // can be toggled mid-accumulation: the per-pixel sample counts are always maintained and
      // both estimators are unbiased, so no restart is needed for those.
      struct FrameConfig
      {
         int  width = 0, height = 0;
         int  depth = 0;
         bool sobol = true;
         bool adaptive = false;
         float threshold = 0.0f;
         bool invalidates(const FrameConfig &o) const
         {
            return width != o.width || height != o.height || depth != o.depth || sobol != o.sobol;
         }
         bool operator!=(const FrameConfig &o) const { return invalidates(o) || adaptive != o.adaptive || threshold != o.threshold; }
      };
      FrameConfig last_config{};

      // Runtime-tweakable golf ball dimple parameters
      bool scene_has_golf_ball = false;
      int   golf_dimple_count  = 128;
      float golf_dimple_radius = 0.155f;
      float golf_dimple_depth  = 0.25f;
      // Volume (cloud / smoke) material of the scene, editable from the GUI (-1 = none)
      int                    volume_mat_idx = -1;
      // Live material swap of the meshes (PgUp/PgDn): every material slot used by a triangle is
      // rewritten in place on the GPU with another material of the scene, no rebuild
      std::vector<int> mesh_mat_slots;        // material slots of the triangles (empty: no mesh in the scene)
      int              mesh_mat_choice = -1;  // index into mesh_mat_candidates (-1: the scene's own materials)
      std::vector<int> mesh_mat_candidates;
      VolumeMaterialSettings volume_settings;

      auto syncSamplesFromSlider = [&]()
      { samples_per_batch = std::max(1, static_cast<int>(samples_per_batch_float)); };

      auto propagateAccumulationToggle = [&]()
      {
         if (accumulation_enabled != auto_accumulate)
            auto_accumulate = accumulation_enabled;
      };

      // Scene selection: built-ins + all YAML files discovered at runtime
      struct SceneEntry
      {
         std::string label;
         std::string yaml_path;
      };

      std::vector<SceneEntry> scene_entries;
      scene_entries.push_back({"Default Scene", ""});

      std::set<std::string> seen_yaml_paths;
      std::vector<std::string> yaml_files;

      auto appendYAMLFromDirectory = [&](const std::string &dir)
      {
         namespace fs = std::filesystem;
         std::error_code ec;
         if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec))
            return;
         for (const auto &entry : fs::directory_iterator(dir, ec))
         {
            if (ec) break;
            if (!entry.is_regular_file(ec)) continue;
            fs::path path = entry.path();
            std::string ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext != ".yaml" && ext != ".yml") continue;
            std::string key;
            std::error_code canon_ec;
            fs::path canonical_path = fs::weakly_canonical(path, canon_ec);
            key = canon_ec ? path.lexically_normal().string() : canonical_path.string();
            if (seen_yaml_paths.insert(key).second)
               yaml_files.push_back(path.lexically_normal().string());
         }
      };

      appendYAMLFromDirectory("../resources/scenes");
      appendYAMLFromDirectory("resources/scenes");
      appendYAMLFromDirectory("../resources");
      appendYAMLFromDirectory("resources");

      std::sort(yaml_files.begin(), yaml_files.end());
      for (const auto &yaml_file : yaml_files)
      {
         std::string stem = std::filesystem::path(yaml_file).stem().string();
         scene_entries.push_back({"YAML: " + stem, yaml_file});
      }

      std::vector<const char *> scene_name_ptrs;
      scene_name_ptrs.reserve(scene_entries.size());
      for (const auto &entry : scene_entries)
         scene_name_ptrs.push_back(entry.label.c_str());

      const char *const *scene_names = scene_name_ptrs.empty() ? nullptr : scene_name_ptrs.data();
      const int scene_count = static_cast<int>(scene_name_ptrs.size());
      int current_scene_index = 0;
      Scene::SceneDescription active_scene = scene;
      Scene::SceneDescription original_scene = scene;

      // Scan active scene for displaced spheres so the GUI section appears when relevant
      auto scanProceduralPatterns = [&]() {
         scene_has_golf_ball = false;
         for (const auto &g : active_scene.geometries)
            if (g.type == Scene::GeometryType::DISPLACED_SPHERE)
               scene_has_golf_ball = true;
         volume_mat_idx = -1;
         for (size_t i = 0; i < active_scene.materials.size() && volume_mat_idx < 0; ++i)
            if (active_scene.materials[i].type == Scene::MaterialType::VOLUME)
               volume_mat_idx = (int)i;

         // Mesh material swap: targets = the material slots of the triangles (OBJ meshes load as
         // triangles), all of them taking the chosen material together. Candidates: every material
         // whose GPU data is self-contained (lights feed the NEE list, volumes carry an SDF and
         // Fibonacci dots a table, all fixed at build time). A mesh's own material is a candidate
         // when there are several meshes (the others take it too). The materials no other object
         // uses come first: they are the ones a scene declares for this.
         mesh_mat_slots.clear();
         mesh_mat_choice = -1;
         mesh_mat_candidates.clear();
         const size_t      num_mats = original_scene.materials.size();
         std::vector<bool> used_elsewhere(num_mats, false), is_slot(num_mats, false);
         for (const auto &g : original_scene.geometries)
         {
            if (g.material_id < 0 || g.material_id >= (int)num_mats)
               continue;
            if (g.type == Scene::GeometryType::TRIANGLE)
            {
               if (!is_slot[g.material_id])
                  mesh_mat_slots.push_back(g.material_id);
               is_slot[g.material_id] = true;
            }
            else
               used_elsewhere[g.material_id] = true;
         }
         if (!mesh_mat_slots.empty())
            for (int pass = 0; pass < 2; ++pass) // pass 0: unused materials, pass 1: shared ones
               for (size_t i = 0; i < num_mats; ++i)
               {
                  const auto &m = original_scene.materials[i];
                  if ((is_slot[i] && mesh_mat_slots.size() == 1) || used_elsewhere[i] != (pass == 1) ||
                      m.type == Scene::MaterialType::LIGHT || m.type == Scene::MaterialType::VOLUME ||
                      m.type == Scene::MaterialType::SHOW_NORMALS ||
                      m.pattern == Scene::ProceduralPattern::FIBONACCI_DOTS)
                     continue;
                  mesh_mat_candidates.push_back((int)i);
               }
         if (volume_mat_idx >= 0)
         {
            const auto &m = active_scene.materials[volume_mat_idx];
            volume_settings.albedo[0]      = (float)m.albedo.x();
            volume_settings.albedo[1]      = (float)m.albedo.y();
            volume_settings.albedo[2]      = (float)m.albedo.z();
            volume_settings.density        = m.density;
            volume_settings.g              = m.anisotropy;
            volume_settings.noise_scale    = m.noise_scale;
            volume_settings.noise_strength = m.noise_strength;
            volume_settings.edge           = m.edge;
            volume_settings.lumps        = m.lumps;
            volume_settings.billow         = m.billow;
            volume_settings.changed        = false;
         }
      };

      // --- HDR Environment Map ---
      std::vector<std::string> hdr_files;
      std::vector<std::string> hdr_labels;
      hdr_labels.push_back("Gradient Sky (built-in)");

      auto scanHdriDir = [&](const std::string &dir)
      {
         namespace fs = std::filesystem;
         std::error_code ec;
         if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec))
            return;
         for (const auto &entry : fs::directory_iterator(dir, ec))
         {
            if (ec) break;
            if (!entry.is_regular_file(ec)) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext == ".hdr")
               hdr_files.push_back(entry.path().lexically_normal().string());
         }
      };
      scanHdriDir("../resources/hdri");
      scanHdriDir("resources/hdri");
      std::sort(hdr_files.begin(), hdr_files.end());
      for (const auto &f : hdr_files)
         hdr_labels.push_back(std::filesystem::path(f).stem().string());

      std::vector<const char *> hdr_name_ptrs;
      for (const auto &l : hdr_labels) hdr_name_ptrs.push_back(l.c_str());
      int hdr_count = static_cast<int>(hdr_labels.size());
      int current_hdr_index = 0;

      auto applyHdrChange = [&](int new_index)
      {
         new_index = std::max(0, std::min(new_index, hdr_count - 1));
         current_hdr_index = new_index;
         if (new_index == 0)
         {
            ::optixRendererClearHdrEnv();
            std::cout << "HDR sky: Gradient Sky (built-in)\n";
         }
         else
         {
            const std::string &path = hdr_files[new_index - 1];
            int w = 0, h = 0;
            auto half_data = loadHdrEnvHalf(path, w, h, hdr_cache);
            if (half_data.empty())
            {
               std::cerr << "HDR: Failed to load '" << path << "'\n";
               return;
            }
            if (!::optixRendererUploadHdrEnvHalf(half_data.data(), w, h))
            {
               std::cerr << "HDR: GPU upload failed for '" << path << "'\n";
               ::optixRendererClearHdrEnv();
               current_hdr_index = 0;
               return;
            }
            std::cout << "HDR sky: '" << hdr_labels[new_index] << "' (" << w << "x" << h << ")\n";
         }
         camera_changed = true;
      };

      // Index of the first HDR label containing `name` (case-insensitive), -1 when none
      auto findHdrIndex = [&](const std::string &name) -> int {
         auto lower = [](std::string t) {
            std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return t;
         };
         const std::string wanted = lower(name);
         for (int i = 1; i < hdr_count; ++i)
            if (lower(hdr_labels[i]).find(wanted) != std::string::npos)
               return i;
         return -1;
      };
      // A scene's settings.hdr_sky selects its sky when it is loaded; without one, the gradient sky
      auto applySceneSky = [&](const Scene::SceneDescription &sc) {
         int idx = 0;
         if (!sc.hdr_sky.empty())
         {
            idx = findHdrIndex(sc.hdr_sky);
            if (idx < 0)
            {
               std::cerr << "HDR: no sky matching '" << sc.hdr_sky << "' in resources/hdri/ (gradient sky)\n";
               idx = 0;
            }
         }
         if (idx != current_hdr_index)
            applyHdrChange(idx);
      };

      // Triangle count shown by the GUI: recounted only when the scene changes (a scan holds
      // millions of geometries, too many to walk every frame)
      int tri_count = -1;
      auto applyVisualizationToActiveScene = [&]()
      {
         tri_count = -1;
         active_scene = original_scene;
         if (mesh_mat_choice >= 0)
            for (int slot : mesh_mat_slots)
               active_scene.materials[slot] = original_scene.materials[mesh_mat_candidates[mesh_mat_choice]];
         if (visualization_mode == static_cast<int>(VisualizationMode::SHOW_NORMALS))
         {
            int mat = active_scene.addMaterial(Scene::MaterialDesc::normal());
            for (auto &geom : active_scene.geometries)
               geom.material_id = mat;
         }
      };

      // Write the chosen material into every mesh slot, on the host copy and on the GPU
      auto materialLabel = [&](int idx) -> std::string {
         if (idx >= 0 && idx < (int)original_scene.material_names.size() && !original_scene.material_names[idx].empty())
            return original_scene.material_names[idx];
         return "material #" + std::to_string(idx);
      };
      auto applyMeshMaterial = [&]()
      {
         if (mesh_mat_slots.empty())
            return;
         std::string label;
         for (int slot : mesh_mat_slots)
         {
            const int src = mesh_mat_choice < 0 ? slot : mesh_mat_candidates[mesh_mat_choice];
            active_scene.materials[slot] = original_scene.materials[src];
            ::optixRendererUpdateMaterial(slot, active_scene.materials[slot]);
            if (mesh_mat_choice >= 0 && label.empty())
               label = materialLabel(src);
            else if (mesh_mat_choice < 0)
               label += (label.empty() ? "" : ", ") + materialLabel(src);
         }
         if (mesh_mat_choice < 0)
            label += " (scene default)";
         gui.showToast("Mesh material: " + label);
         std::cout << "Mesh material: " << label << "\n";
         camera_changed = true;
      };
      // step = +1 / -1 through [scene default, candidates...]
      auto cycleMeshMaterial = [&](int step)
      {
         if (mesh_mat_slots.empty())
         {
            gui.showToast("No mesh in this scene");
            return;
         }
         if (mesh_mat_candidates.empty())
         {
            gui.showToast("No other material in this scene");
            return;
         }
         const int n     = (int)mesh_mat_candidates.size() + 1;
         mesh_mat_choice = ((mesh_mat_choice + 1 + step) % n + n) % n - 1;
         applyMeshMaterial();
      };

      auto applySceneSelectionChange = [&]()
      {
         if (current_scene_index < 0 || current_scene_index >= scene_count)
            current_scene_index = 0;
         mesh_mat_choice = -1; // the swap belongs to the scene being left

         const SceneEntry &selected = scene_entries[current_scene_index];
         std::cout << "Switching to scene: " << selected.label;
         if (!selected.yaml_path.empty())
            std::cout << " (" << selected.yaml_path << ")";
         std::cout << std::endl;
         // The load and the BVH build block this thread: show what is going on before they start
         gui.showBusyMessage("Loading " + (selected.yaml_path.empty() ? selected.label : std::filesystem::path(selected.yaml_path).stem().string()) +
                             " ... building BVH");

         if (current_scene_index == 0)
            active_scene = Scene::SceneFactory::createDefaultScene();
         else
            active_scene = Scene::SceneFactory::fromYAML(selected.yaml_path, /*skip_cpu_bvh=*/true);

         original_scene = active_scene;
         applyVisualizationToActiveScene();

         look_from = active_scene.camera_position;
         look_at = active_scene.camera_look_at;
         camera.vup = active_scene.camera_up;
         camera.vfov = active_scene.camera_fov;
         cam_fov_ui = static_cast<float>(camera.vfov);
         camera_control.initializeCameraControls(look_from, look_at);

         background_intensity = active_scene.background_intensity;
         light_intensity      = active_scene.light_intensity;
         light_falloff        = active_scene.light_falloff;
         indirect_clamp       = active_scene.indirect_clamp;
         samples_per_batch_float = static_cast<float>(sceneBatchSize(active_scene));
         adaptive_threshold      = sceneAdaptiveThreshold(active_scene);
         ::setOptiXBackgroundColor((float)active_scene.background_color.x(), (float)active_scene.background_color.y(),
                                   (float)active_scene.background_color.z());
         applySceneSky(active_scene);

         // Re-scan for procedural patterns after scene change
         scanProceduralPatterns();
         ::optixRendererSetGolfDimples(golf_dimple_count, golf_dimple_radius, golf_dimple_depth);

         // Rebuild OptiX scene
         optixRendererBuildScene(active_scene);
         camera_changed = true;
      };

      // Scan scene initially (after active_scene is set)
      scanProceduralPatterns();
      ::setOptiXBackgroundColor((float)active_scene.background_color.x(), (float)active_scene.background_color.y(),
                                (float)active_scene.background_color.z());
      ::optixRendererSetGolfDimples(golf_dimple_count, golf_dimple_radius, golf_dimple_depth);
      ::setOptiXNEEStride(nee_stride);

      // Rendering buffers
      SDL_Event event;
      std::vector<unsigned char> display_image(image_width * image_height * image_channels);
      std::vector<float> accum_buffer(image_width * image_height * image_channels, 0.0f);
      RenderTargetView display_view{&display_image, image_width, image_height, image_channels};

      // Initialize OptiX pipeline and build scene
      optixRendererInit();
      optixRendererBuildScene(active_scene);

      // Starting sky: --sky <name> wins, else the scene's settings.hdr_sky, else the gradient
      if (!settings_.sky.empty())
      {
         const int found = findHdrIndex(settings_.sky);
         if (found > 0)
            applyHdrChange(found);
         else
            std::cerr << "HDR: no sky matching '" << settings_.sky << "' in resources/hdri/ (gradient sky kept)\n";
      }
      else
         applySceneSky(scene);

      auto last_frame_time = std::chrono::high_resolution_clock::now();
      auto total_start = std::chrono::high_resolution_clock::now();

      // ─── Main loop ───────────────────────────────────────────────────────────
      while (running)
      {
         bool visualization_toggled_by_key = false;
         bool scene_switched_by_key = false;

         while (SDLGuiHandler::pollEvent(event))
         {
            if (event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE))
            {
               running = false;
            }

            ImGuiIO &io = ImGui::GetIO();
            if (io.WantCaptureMouse)
            {
               if (event.type == SDL_MOUSEBUTTONDOWN || event.type == SDL_MOUSEBUTTONUP ||
                   event.type == SDL_MOUSEMOTION || event.type == SDL_MOUSEWHEEL)
                  continue;
            }

            if (event.type == SDL_KEYDOWN && !io.WantCaptureKeyboard)
            {
               if (event.key.keysym.sym == SDLK_h)
               {
                  gui.toggleControls();
                  show_spps_counter = !show_spps_counter;
               }
               else if (event.key.keysym.sym == SDLK_RETURN)
               {
                  gui.toggleWindowCollapse();
               }
               else if (event.key.keysym.sym == SDLK_c)
               {
                  gui.toggleHeaderCollapse();
               }
               else if (event.key.keysym.sym == SDLK_r)
               {
                  light_intensity = 1.0f; background_intensity = 1.0f; light_falloff = 2.0f;
                  indirect_clamp = constants::INDIRECT_CLAMP_DEFAULT;
                  metal_fuzziness = 1.0f; glass_refraction_index = 1.5f;
                  dof_enabled = false; dof_aperture = 0.1f; dof_focus_distance = 10.0f;
                  show_spps_counter = true;
                  gui.setLogoVisible(true);
                  samples_per_batch_float = static_cast<float>(settings_.samples_per_batch);
                  camera_control.setAutoOrbit(false);
                  // Reset MIS / motion options to settings_ defaults
                  mis_enabled           = settings_.mis_enabled;
                  motion_gate_mis       = settings_.motion_gate_mis;
                  nee_stride            = settings_.nee_stride;
                  use_sobol             = true;
                  motion_quality        = settings_.motion_quality;
                  adaptive_sampling_enabled = settings_.adaptive_sampling;
                  if (mesh_mat_choice >= 0)
                  {
                     mesh_mat_choice = -1;
                     applyMeshMaterial();
                  }
                  camera_changed = true;
               }
               else if (event.key.keysym.sym == SDLK_f)
               {
                  show_spps_counter = !show_spps_counter;
               }
               else if (event.key.keysym.sym == SDLK_l)
               {
                  gui.toggleLogo();
               }
               else if (event.key.keysym.sym == SDLK_n)
               {
                  visualization_mode =
                      (visualization_mode == static_cast<int>(VisualizationMode::SHOW_NORMALS))
                          ? static_cast<int>(VisualizationMode::NORMAL)
                          : static_cast<int>(VisualizationMode::SHOW_NORMALS);
                  visualization_toggled_by_key = true;
               }
               else if (event.key.keysym.sym == SDLK_LEFT)
               {
                  current_scene_index = (current_scene_index - 1 + scene_count) % scene_count;
                  scene_switched_by_key = true;
               }
               else if (event.key.keysym.sym == SDLK_RIGHT)
               {
                  current_scene_index = (current_scene_index + 1) % scene_count;
                  scene_switched_by_key = true;
               }
               else if (event.key.keysym.sym == SDLK_KP_PLUS || event.key.keysym.sym == SDLK_EQUALS)
               {
                  applyHdrChange((current_hdr_index + 1) % hdr_count);
               }
               else if (event.key.keysym.sym == SDLK_KP_MINUS || event.key.keysym.sym == SDLK_MINUS)
               {
                  applyHdrChange((current_hdr_index - 1 + hdr_count) % hdr_count);
               }
               else if (event.key.keysym.sym == SDLK_PAGEUP)
               {
                  cycleMeshMaterial(+1);
               }
               else if (event.key.keysym.sym == SDLK_PAGEDOWN)
               {
                  cycleMeshMaterial(-1);
               }
               else if (camera_control.handleKeyDown(event, accumulation_enabled, samples_per_batch_float,
                                                     light_intensity, background_intensity, needs_rerender,
                                                     camera_changed))
               {
                  syncSamplesFromSlider();
                  propagateAccumulationToggle();
               }
            }
            else if (event.type == SDL_MOUSEBUTTONDOWN)
            {
               camera_control.handleMouseButtonDown(event);
            }
            else if (event.type == SDL_MOUSEBUTTONUP)
            {
               camera_control.handleMouseButtonUp(event);
            }
            else if (event.type == SDL_MOUSEMOTION)
            {
               if (camera_control.handleMouseMotion(event, look_from, look_at, vup, basis_w))
                  camera_changed = true;
            }
            else if (event.type == SDL_MOUSEWHEEL)
            {
               if (camera_control.handleMouseWheel(event, look_from, look_at))
                  camera_changed = true;
            }
         }

         if (scene_switched_by_key)
         {
            applySceneSelectionChange();
         }

         if (visualization_toggled_by_key)
         {
            applyVisualizationToActiveScene();
            optixRendererBuildScene(active_scene);
            camera_changed = true;
         }

         // Auto-orbit
         auto current_frame_time = std::chrono::high_resolution_clock::now();
         std::chrono::duration<float> delta = current_frame_time - last_frame_time;
         last_frame_time = current_frame_time;
         if (delta.count() > 0.0f)
            current_fps = 1.0f / delta.count();

         if (camera_control.updateAutoOrbit(look_from, look_at, delta.count()))
            camera_changed = true;

         // Motion detection
         auto now = std::chrono::high_resolution_clock::now();
         std::chrono::duration<float> time_since_last_change = now - last_camera_change_time;
         is_camera_moving = (time_since_last_change.count() < motion_quality.cooldown_seconds);

         // Camera changed → the camera is moving; the accumulation restarts below
         bool restart_accumulation = camera_changed;
         if (camera_changed)
         {
            camera_changed = false;
            last_camera_change_time = now;
            is_camera_moving = true;
            refreshCameraFrame();
         }

         // Per-frame configuration: trade quality for framerate while the camera moves
         FrameConfig config;
         const bool reduce = is_camera_moving;
         // Resolution stages: motion_fraction while moving; once the camera stops, every standard
         // fraction above it (1/2, 3/4) for stage_seconds each, then full. Every step is a config
         // upgrade, so the previous image is handed over as a display prior that fades out over
         // fade_seconds and the picture sharpens in place.
         float motion_frac = std::clamp(motion_quality.motion_fraction, 0.1f, 1.0f);
         int   motion_spp  = motion_quality.single_sample ? 1 : samples_per_batch;
         if (motion_quality.hold_target_fps)
         {
            // Regulator: hold target_fps while moving. The cost of a motion frame is modelled as
            // overhead + k x pixels x samples, with k (ms per pixel-sample) and the overhead (GUI,
            // upload, present) measured on the previous motion frame and smoothed. Each frame the
            // best (fraction, samples) pair is chosen for the budget, adapting the resolution
            // first: with the full motion sample count, the highest fraction (in steps of 1/16
            // between min and max) that fits; only when even the floor does not fit are the
            // samples reduced, again with the highest fraction that fits. Going down is immediate;
            // going up needs several frames of headroom, and a raise that had to be undone blocks
            // further raises for a while, so the loop does not hunt when the model is optimistic
            // (a small frame under-uses the GPU, so k grows as the frame shrinks).
            const float min_frac = std::clamp(motion_quality.min_fraction, 0.05f, motion_frac);
            const int   min_spp  = std::clamp(motion_quality.min_spp, 1, std::max(1, motion_spp));
            if (dyn_spp <= 0)
               dyn_spp = motion_spp;
            dyn_fraction = std::clamp(dyn_fraction, min_frac, motion_frac);
            dyn_spp      = std::clamp(dyn_spp, min_spp, std::max(1, motion_spp));
            if (is_camera_moving && last_frame_motion && delta.count() > 0.0f && last_motion_pixels > 0)
            {
               const float target_ms = 1000.0f / std::max(1.0f, motion_quality.target_fps);
               // A hitch (scene load, shader compile) must not throw the loop to the floor: cap one
               // frame's weight at a few times the target before smoothing.
               const float frame_ms  = std::min(delta.count() * 1000.0f, 4.0f * target_ms);
               const float over_ms   = std::max(0.0f, frame_ms - last_render_ms);
               // The denoiser's time scales with pixels only, so it is modelled apart from the
               // per-sample tracing cost (or lowering the sample count would look cheaper than it is)
               const float trace_ms  = std::max(0.0f, last_render_ms - last_denoise_ms);
               const float k_sample  = trace_ms / (float)std::max(1, last_motion_pixels * last_motion_spp);
               const float d_pixel   = last_denoise_ms / (float)std::max(1, last_motion_pixels);
               motion_k_ms        = (motion_k_ms > 0.0f) ? 0.8f * motion_k_ms + 0.2f * k_sample : k_sample;
               motion_d_ms        = (motion_d_ms > 0.0f) ? 0.8f * motion_d_ms + 0.2f * d_pixel : d_pixel;
               if (last_denoise_ms <= 0.0f)
                  motion_d_ms = 0.0f;
               motion_overhead_ms = (motion_overhead_ms > 0.0f) ? 0.9f * motion_overhead_ms + 0.1f * over_ms : over_ms;
               const float budget_ms = std::max(0.5f, target_ms - motion_overhead_ms);

               auto pixels_at = [&](float f) {
                  return std::max(1, (int)std::lround(image_width * f)) * std::max(1, (int)std::lround(image_height * f));
               };
               auto cost_at = [&](float f, int spp) {
                  const float px = (float)pixels_at(f);
                  return motion_k_ms * px * (float)spp + motion_d_ms * px;
               };
               float best_frac = min_frac;
               int   best_spp  = min_spp;
               bool  found     = false;
               for (int spp = std::max(1, motion_spp); spp >= min_spp && !found; --spp)
                  for (float f = motion_frac; f >= min_frac - 1e-3f; f -= 0.0625f)
                     if (cost_at(f, spp) <= budget_ms)
                     {
                        best_frac = std::max(f, min_frac);
                        best_spp  = spp;
                        found     = true;
                        break;
                     }
               const bool same_frac = std::fabs(best_frac - dyn_fraction) < 1e-3f;
               const bool better    = best_frac > dyn_fraction + 1e-3f || (same_frac && best_spp > dyn_spp);
               const bool worse     = best_frac < dyn_fraction - 1e-3f || (same_frac && best_spp < dyn_spp);
               if (worse)
               {
                  // Undoing a recent raise: the model was optimistic, hold off further raises
                  if (std::chrono::duration<float>(now - last_raise_time).count() < 1.0f)
                     raise_block_until = now + std::chrono::milliseconds(2000);
                  dyn_fraction = best_frac;
                  dyn_spp      = best_spp;
                  headroom_frames = 0;
               }
               else if (better)
               {
                  const float predicted = cost_at(best_frac, best_spp);
                  if (predicted <= 0.85f * budget_ms && now >= raise_block_until && ++headroom_frames >= 6)
                  {
                     dyn_fraction    = best_frac;
                     dyn_spp         = best_spp;
                     last_raise_time = now;
                     headroom_frames = 0;
                  }
               }
               else
                  headroom_frames = 0;
            }
            motion_frac = dyn_fraction;
            motion_spp  = dyn_spp;
         }
         if (is_camera_moving)
         {
            motion_quality.current_fraction = motion_frac;
            motion_quality.current_spp      = motion_spp;
         }
         last_frame_motion = is_camera_moving;
         float stages[3];
         int   n_stages = 0;
         if (motion_quality.refine_stages)
            for (float f : {0.5f, 0.75f})
               if (f > motion_frac + 1e-3f)
                  stages[n_stages++] = f;
         stages[n_stages++] = 1.0f;
         auto stage_size = [&](float frac, int full) { return std::max(1, (int)std::lround(full * frac)); };
         if (is_camera_moving)
            stage_index = 0; // first stage after the motion
         else if (stage_index < n_stages - 1 && last_config.width == stage_size(stages[stage_index], image_width) &&
                  std::chrono::duration<float>(now - stage_start_time).count() >= motion_quality.stage_seconds)
            ++stage_index; // the current stage has run its time: next one
         stage_index = std::min(stage_index, n_stages - 1);
         const float res_frac = is_camera_moving ? motion_frac : stages[stage_index];
         config.width    = stage_size(res_frac, image_width);
         config.height   = stage_size(res_frac, image_height);
         config.depth    = (reduce && motion_quality.limit_depth) ? std::max(1, motion_quality.motion_max_depth)
                                                                  : frame.max_depth;
         config.sobol    = use_sobol && !(reduce && motion_quality.fast_sampler);
         config.adaptive = adaptive_sampling_enabled && !reduce;
         config.threshold = adaptive_threshold;
         const int frame_spp = reduce ? motion_spp : samples_per_batch;

         // Configuration changed (e.g. motion cooldown elapsed → back to full resolution): the
         // accumulated samples were taken under a different setup, so start over. This is not
         // camera motion, so it must not touch the motion timer.
         const bool config_upgrade = config.invalidates(last_config) && !restart_accumulation &&
                                     current_samples > 0 && config.width >= last_config.width &&
                                     config.height >= last_config.height && config.depth >= last_config.depth;
         if (config.invalidates(last_config) && current_samples > 0)
            restart_accumulation = true;

         if (restart_accumulation)
         {
            convergence_pct = 0.0f;
            force_immediate_render = true;
            std::fill(accum_buffer.begin(), accum_buffer.end(), 0.0f);
            // Hand the previous image over as a display prior that fades out over fade_seconds, so
            // each step to a higher quality refines the picture instead of snapping to a fresh, noisy
            // one. The prior's weight is a sample count: estimate the samples the new stage will
            // accumulate in fade_seconds from the measured frame rate, scaled by the pixel-count
            // ratio since a larger frame renders proportionally slower.
            float fade_samples = 48.0f;
            if (config_upgrade && current_fps > 0.0f)
            {
               const float pixel_ratio = (float)(last_config.width * last_config.height) /
                                         (float)std::max(1, config.width * config.height);
               fade_samples = motion_quality.fade_seconds * current_fps * pixel_ratio * (float)std::max(1, frame_spp);
               fade_samples = std::clamp(fade_samples, 16.0f, 100000.0f);
            }
            bool handed_over = config_upgrade &&
                               ::optixRendererHandoverPrior(config.width, config.height, fade_samples);
            if (!handed_over)
               optixRendererResetAccum(image_width, image_height);
            current_samples  = 0;
            stage_start_time = now;
         }
         if (config != last_config)
         {
            ::setOptiXSobolSampler(config.sobol);
            ::setOptiXAdaptiveSampling(config.adaptive, min_adaptive_samples, adaptive_threshold);
            last_config = config;
         }

         // Denoiser schedule: every motion frame is fully denoised; once still, the denoiser fades out
         // (blend 0 -> 1) as the accumulation approaches denoise_until_samples, then stops.
         auto denoiseFor = [&](int samples, bool moving, bool &denoise_out, float &blend_out)
         {
            denoise_out = false;
            blend_out   = 0.0f;
            if (!motion_quality.denoise)
               return;
            if (moving)
               denoise_out = true;
            else if (motion_quality.denoise_until_samples > 0 && samples < motion_quality.denoise_until_samples)
            {
               denoise_out = true;
               blend_out   = (float)samples / (float)motion_quality.denoise_until_samples;
            }
         };

         // Redisplay after overlay/settings change without adding new samples
         if (needs_rerender && current_samples > 0)
         {
            bool  denoise_now;
            float denoise_blend;
            denoiseFor(current_samples, reduce, denoise_now, denoise_blend);
            optixRendererConvertAccumToDisplay(display_image.data(), image_width, image_height, image_channels,
                                               current_samples, gamma, reduce && motion_despeckle,
                                               motion_despeckle_ratio, constants::MOTION_DESPECKLE_FLOOR,
                                               denoise_now, denoise_blend);
            if (target.pixels) *target.pixels = display_image;
            needs_rerender = false;
         }

         // Stop rendering when max SPP reached, or when adaptive sampling reports that (almost) every
         // pixel has converged. The last few pixels of a scene lit by caustics can keep an enormous
         // variance for a very long time; waiting for them keeps the GPU busy for no visible gain.
         bool all_converged = config.adaptive && convergence_pct >= ADAPTIVE_STOP_PCT;
         bool should_render = (current_samples < max_samples && !all_converged && !camera_changed && running) ||
                              force_immediate_render;
         bool needs_initial_render = current_samples == 0 && !accumulation_enabled;

         if (should_render && (accumulation_enabled || needs_initial_render || force_immediate_render))
         {
            force_immediate_render = false;
            syncSamplesFromSlider();

            // Motion-gate MIS — auto-disable NEE/MIS while the camera is moving
            bool effective_mis = mis_enabled && !(motion_gate_mis && is_camera_moving);
            ::setOptiXMISEnabled(effective_mis);
            // Low fixed firefly ceiling while moving: each frame is displayed on its own
            ::setOptiXFireflyClampCeiling(reduce ? constants::MOTION_FIREFLY_CLAMP : 0.0f);

            auto frame_start = std::chrono::high_resolution_clock::now();

            const int num_materials_active = static_cast<int>(active_scene.materials.size());
            bool  denoise_now;
            float denoise_blend;
            denoiseFor(current_samples + frame_spp, reduce, denoise_now, denoise_blend);
            ::setOptiXLightFalloff(light_falloff);
            ::setOptiXIndirectClamp(indirect_clamp);
            renderBatch(frame, config.width, config.height, config.depth, display_view, current_samples,
                        max_samples, frame_spp, gamma, context, num_materials_active, background_intensity,
                        dof_enabled, dof_aperture, dof_focus_distance, light_intensity, metal_fuzziness,
                        glass_refraction_index / constants::GLASS_IOR_REFERENCE, reduce && motion_despeckle, motion_despeckle_ratio, denoise_now,
                        denoise_blend);
            last_denoise_ms = denoise_now ? ::optixRendererLastDenoiseMs() : 0.0f;
            if (denoise_now)
               motion_quality.denoise_ms = last_denoise_ms;

            auto frame_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> frame_time = frame_end - frame_start;
            if (frame_time.count() > 0.0f)
            {
               // Converged pixels are skipped by the kernel, so count only the pixels still being
               // sampled (from the last convergence readout) or the throughput balloons as the
               // image converges.
               float active_pixels = static_cast<float>(config.width) * config.height;
               if (config.adaptive)
                  active_pixels *= std::max(0.0f, 1.0f - convergence_pct / 100.0f);
               float total_samples = static_cast<float>(frame_spp) * active_pixels;
               current_sps = (total_samples * 1000.0f) / frame_time.count();
               current_ms_per_sample = frame_time.count() / static_cast<float>(frame_spp);
               last_render_ms = frame_time.count();
               last_motion_pixels = config.width * config.height;
               last_motion_spp    = frame_spp;
            }

            // Convergence readout for the GUI (every few frames: it costs a small kernel + sync)
            if (config.adaptive && ++frames_since_convergence_check >= 8)
            {
               frames_since_convergence_check = 0;
               int converged = ::optixRendererCountConvergedPixels();
               if (converged >= 0)
                  convergence_pct = 100.0f * (float)converged / (float)(config.width * config.height);
            }

            // Heatmap overlay replaces the image with the per-pixel sample counts
            if (show_heatmap && config.adaptive)
               ::optixRendererRenderSampleHeatmap(display_image.data(), image_width, image_height, image_channels,
                                                  std::max(1, current_samples));


            if (target.pixels) *target.pixels = display_image;
         }
         else
         {
            current_sps = 0.0f; // idle: nothing is being sampled
            SDL_Delay(16);
         }

         // Snapshot current values to detect ImGui changes
         bool old_dof = dof_enabled;
         float old_aperture = dof_aperture;
         float old_focus = dof_focus_distance;
         float old_light = light_intensity;
         float old_falloff = light_falloff;
         float old_indirect_clamp = indirect_clamp;
         float old_background = background_intensity;
         float old_fuzz = metal_fuzziness;
         float old_ior = glass_refraction_index;
         float old_cam_fov = cam_fov_ui;
         int old_scene_index = current_scene_index;
         bool old_adaptive = adaptive_sampling_enabled;
         float old_adaptive_thresh = adaptive_threshold;
         int old_visualization_mode = visualization_mode;
         int   old_golf_dimple_count  = golf_dimple_count;
         float old_golf_dimple_radius = golf_dimple_radius;
         float old_golf_dimple_depth  = golf_dimple_depth;
         int   old_hdr_index          = current_hdr_index;
         // MIS option snapshots
         bool old_mis_enabled           = mis_enabled;
         bool old_motion_gate_mis       = motion_gate_mis;
         int  old_nee_stride            = nee_stride;
         bool old_use_sobol             = use_sobol;

         bool auto_orbit = camera_control.isAutoOrbitEnabled();
         float cam_pos[3] = {(float)look_from.x(), (float)look_from.y(), (float)look_from.z()};
         float cam_lookat[3] = {(float)look_at.x(), (float)look_at.y(), (float)look_at.z()};

         if (tri_count < 0)
            tri_count = static_cast<int>(std::count_if(active_scene.geometries.begin(), active_scene.geometries.end(),
                                                       [](const Scene::GeometryDesc &g)
                                                       { return g.type == Scene::GeometryType::TRIANGLE; }));

         gui.updateDisplay(display_image, image_channels, current_sps, current_ms_per_sample, current_fps, current_samples,
                           &dof_enabled, &dof_aperture, &dof_focus_distance, &light_intensity,
                           &background_intensity, &metal_fuzziness, &glass_refraction_index,
                           &samples_per_batch_float, &accumulation_enabled,
                           &auto_orbit, &current_scene_index, scene_names, scene_count,
                           cam_pos, cam_lookat, &cam_fov_ui,
                           &adaptive_sampling_enabled, &adaptive_threshold, convergence_pct, &show_heatmap,
                           &visualization_mode, nullptr, nullptr,
                           nullptr, nullptr, &show_spps_counter, tri_count,
                           nullptr,
                           scene_has_golf_ball ? &golf_dimple_count  : nullptr,
                           scene_has_golf_ball ? &golf_dimple_radius : nullptr,
                           scene_has_golf_ball ? &golf_dimple_depth  : nullptr,
                           &current_hdr_index,
                           hdr_count > 0 ? hdr_name_ptrs.data() : nullptr,
                           hdr_count,
                           &mis_enabled, &motion_gate_mis, &nee_stride,
                           &use_sobol, &motion_quality, &motion_despeckle, &motion_despeckle_ratio,
                           volume_mat_idx >= 0 ? &volume_settings : nullptr,
                           &save_camera_request, active_scene.source_file.c_str(), camera_save_status.c_str(),
                           &light_falloff, &indirect_clamp);

         if (save_camera_request)
         {
            save_camera_request = false;
            if (Scene::saveViewToYAML(active_scene.source_file.c_str(), look_from, look_at, camera.vfov,
                                      background_intensity, light_intensity, light_falloff, indirect_clamp))
               camera_save_status = "Saved to " + active_scene.source_file;
            else
               camera_save_status = "Could not write " + active_scene.source_file;
            camera_save_status_time = std::chrono::steady_clock::now();
         }
         else if (!camera_save_status.empty() &&
                  std::chrono::steady_clock::now() - camera_save_status_time > std::chrono::seconds(4))
            camera_save_status.clear();

         if (auto_orbit != camera_control.isAutoOrbitEnabled())
            camera_control.setAutoOrbit(auto_orbit);

         // Cloud / smoke parameters edited in the GUI: update the material in place and restart
         if (volume_settings.changed && volume_mat_idx >= 0)
         {
            auto &m          = active_scene.materials[volume_mat_idx];
            m.albedo         = Vec3(volume_settings.albedo[0], volume_settings.albedo[1], volume_settings.albedo[2]);
            m.density        = volume_settings.density;
            m.anisotropy     = volume_settings.g;
            m.noise_scale    = volume_settings.noise_scale;
            m.noise_strength = volume_settings.noise_strength;
            m.edge           = volume_settings.edge;
            m.lumps        = volume_settings.lumps;
            m.billow         = volume_settings.billow;
            ::optixRendererUpdateMaterial(volume_mat_idx, m);
            volume_settings.changed = false;
            camera_changed = true;
         }

         gui.drawLogo();
         gui.present();

         // Handle scene change from ImGui (may have switched the sky: do not re-apply it below)
         if (current_scene_index != old_scene_index)
         {
            applySceneSelectionChange();
            old_hdr_index = current_hdr_index;
         }

         // Handle visualization mode change from ImGui
         if (visualization_mode != old_visualization_mode)
         {
            applyVisualizationToActiveScene();
            optixRendererBuildScene(active_scene);
            camera_changed = true;
         }

         // Detect ImGui-driven HDR sky change
         if (current_hdr_index != old_hdr_index)
            applyHdrChange(current_hdr_index);

         // Detect if ImGui changed rendering parameters
         if (dof_enabled != old_dof || dof_aperture != old_aperture || dof_focus_distance != old_focus ||
             light_intensity != old_light || light_falloff != old_falloff || background_intensity != old_background ||
             indirect_clamp != old_indirect_clamp ||
             metal_fuzziness != old_fuzz || glass_refraction_index != old_ior || cam_fov_ui != old_cam_fov)
         {
            if (cam_fov_ui != old_cam_fov)
               camera.vfov = cam_fov_ui;
            camera_changed = true;
         }

         // Golf ball dimple params changed — push to GPU via OptiX launch params
         if (golf_dimple_count != old_golf_dimple_count || golf_dimple_radius != old_golf_dimple_radius ||
             golf_dimple_depth != old_golf_dimple_depth)
         {
            ::optixRendererSetGolfDimples(golf_dimple_count, golf_dimple_radius, golf_dimple_depth);
            camera_changed = true;
         }

         // Adaptive sampling toggled or threshold changed — restart accumulation
         if (adaptive_sampling_enabled != old_adaptive || adaptive_threshold != old_adaptive_thresh)
            camera_changed = true;

         // MIS option changes — push updated constants to GPU and restart accumulation
         if (nee_stride != old_nee_stride)
         {
            ::setOptiXNEEStride(nee_stride);
            camera_changed = true;
         }
         if (mis_enabled != old_mis_enabled || motion_gate_mis != old_motion_gate_mis)
            camera_changed = true;
         if (use_sobol != old_use_sobol)
            camera_changed = true; // applied through FrameConfig on the next frame

         // Arrow overlay removed (CPU renderer archived on legacy/cpu-renderer branch)
      }

      auto total_end = std::chrono::high_resolution_clock::now();
      std::cout << "\nTotal session time: " << render::timeStr(total_end - total_start) << std::endl;

      optixRendererCleanup();
      // note: optixRendererCleanup() now calls optixRendererClearHdrEnv() internally
   }

 private:
   Settings settings_{};

   /// Trace one batch at render_width x render_height (may be smaller than the display while the
   /// camera moves) and convert the accumulation buffer into the full-size display image.
   void renderBatch(const CameraFrame &frame, int render_width, int render_height, int depth,
                    RenderTargetView display_target, int &current_samples, int max_samples, int samples_per_batch,
                    float gamma, RenderContext &context, int num_materials, float background_intensity,
                    bool dof_enabled, float dof_aperture, float dof_focus_distance, float light_intensity,
                    float metal_fuzziness, float glass_ior_multiplier, bool despeckle, float despeckle_ratio,
                    bool denoise, float denoise_blend)
   {
      // If we've already reached or exceeded the maximum, do not render more samples.
      if (current_samples >= max_samples) return;

      const int remaining           = max_samples - current_samples;
      const int actual_samples      = std::min(samples_per_batch, remaining);
      const int samples_before_batch = current_samples; // total BEFORE adding this batch
      const int new_total_samples   = current_samples + actual_samples;

      // Re-grid the camera's pixel lattice for the render resolution: same viewport, coarser
      // pixels. pixel00 is the centre of pixel (0,0); the viewport corner is pixel00 - (du+dv)/2.
      const double sx = static_cast<double>(frame.image_width) / render_width;
      const double sy = static_cast<double>(frame.image_height) / render_height;
      const Vec3 du = frame.pixel_delta_u * sx;
      const Vec3 dv = frame.pixel_delta_v * sy;
      const Vec3 corner = frame.pixel00_loc - 0.5 * frame.pixel_delta_u - 0.5 * frame.pixel_delta_v;
      const Vec3 pixel00 = corner + 0.5 * du + 0.5 * dv;

      unsigned long long ray_count = optixRendererLaunch(
          render_width, render_height, num_materials,
          actual_samples, samples_before_batch, depth,
          static_cast<float>(frame.camera_center.x()), static_cast<float>(frame.camera_center.y()),
          static_cast<float>(frame.camera_center.z()),
          static_cast<float>(pixel00.x()), static_cast<float>(pixel00.y()), static_cast<float>(pixel00.z()),
          static_cast<float>(du.x()), static_cast<float>(du.y()), static_cast<float>(du.z()),
          static_cast<float>(dv.x()), static_cast<float>(dv.y()), static_cast<float>(dv.z()),
          static_cast<float>(frame.u.x()), static_cast<float>(frame.u.y()), static_cast<float>(frame.u.z()),
          static_cast<float>(frame.v.x()), static_cast<float>(frame.v.y()), static_cast<float>(frame.v.z()),
          background_intensity, dof_enabled, dof_aperture, dof_focus_distance,
          light_intensity, metal_fuzziness, glass_ior_multiplier);

      // GPU-side gamma correction (and upscale when rendering at reduced resolution): converts the
      // float4 accum buffer straight to the uint8 display image, no float4 D2H + host conversion.
      optixRendererConvertAccumToDisplay(display_target.pixels->data(), frame.image_width, frame.image_height,
                                         display_target.channels, new_total_samples, gamma, despeckle,
                                         despeckle_ratio, constants::MOTION_DESPECKLE_FLOOR, denoise, denoise_blend);

      context.ray_counter.fetch_add(ray_count, std::memory_order_relaxed);

      current_samples = new_total_samples;
   }

   // (display is now composited inline in the main loop — no displayFrame helper needed)
};

#endif // SDL2_FOUND && OPTIX_FOUND
