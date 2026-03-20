/**
 * @class RendererProgressiveSDL
 * @brief Interactive SDL renderer with progressive sample accumulation in CUDA
 *
 * This renderer focuses on ray-tracing logic with progressive quality improvement.
 * GUI and camera control are delegated to separate handler classes.
 */
#pragma once

#ifdef SDL2_FOUND

#include "render/render_utils.hpp"
#include "render/renderer_interface.hpp"
#include "renderer_cuda_host.hpp"
#include "scene_builder.hpp"
#include "scene_factory.hpp"
#include "sdl_gui_controls.hpp"
#include "sdl_gui_handler.hpp"
#include "hdr_env_cache.hpp"

#include <SDL.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <cmath>
#include <set>
#include <string>
#include <vector>

class RendererCUDAProgressive : public IRenderer
{
 public:
   struct Settings
   {
      int samples_per_batch = constants::INTERACTIVE_SAMPLES_PER_BATCH;
      bool auto_accumulate = true;
      bool adaptive_sampling = true;
      bool hdr_cache = true; ///< use .hdrcache sidecar to speed up repeated HDR loads
      GuiTheme theme = GuiTheme::NORD;
      // MIS / NEE options
      bool mis_enabled          = true;  ///< Master MIS toggle
      bool motion_gate_mis      = true;  ///< Option A: auto-disable MIS during camera motion
      bool nee_first_bounce_only = false; ///< Option B: NEE on first bounce only
      int  nee_stride           = 1;     ///< Option C: do NEE every N samples (1 = always)
   };

   RendererCUDAProgressive() = default;
   explicit RendererCUDAProgressive(Settings settings) : settings_(settings) {}

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
      SDLGuiHandler gui(target.width, target.height, settings_.theme, "CUDA");
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
      float light_intensity = 1.0f;
      float background_intensity = scene.background_intensity;
      float metal_fuzziness = 1.0f;
      float glass_refraction_index = 1.5f;
      bool dof_enabled = false;
      float dof_aperture = 0.1f;
      float dof_focus_distance = 10.0f;
      float cam_fov_ui = static_cast<float>(camera.vfov);
      bool needs_rerender = false;
      bool force_immediate_render = false;
      float samples_per_batch_float = static_cast<float>(samples_per_batch);
      float current_sps = 0.0f;
      float current_ms_per_sample = 0.0f;
      float current_fps = 0.0f;

      // MIS / NEE options (Options A–C)
      bool mis_enabled           = settings_.mis_enabled;
      bool motion_gate_mis       = settings_.motion_gate_mis;
      bool nee_first_bounce_only = settings_.nee_first_bounce_only;
      int  nee_stride            = settings_.nee_stride;
      bool use_sobol             = true; // default: Sobol sampler

      // Motion detection for adaptive quality
      bool is_camera_moving = false;
      auto last_camera_change_time = std::chrono::high_resolution_clock::now();
      const float motion_cooldown_seconds = 0.5f;

      // Adaptive sample rate
      int adaptive_samples_per_batch = samples_per_batch;

      // Runtime-tweakable procedural pattern parameters (declared before applySceneSettings)
      bool scene_has_golf_ball      = false;
      int   golf_dimple_count       = 150;
      float golf_dimple_radius      = 0.24f;
      float golf_dimple_depth       = 0.35f;

      auto syncSamplesFromSlider = [&]()
      { samples_per_batch = std::max(1, static_cast<int>(samples_per_batch_float)); };

      auto applySceneSettings = [&]()
      {
         ::setLightIntensity(light_intensity);
         ::setBackgroundIntensity(background_intensity);
         ::setMetalFuzziness(metal_fuzziness);
         ::setGlassRefractionIndex(glass_refraction_index);
         ::setDOFEnabled(dof_enabled);
         ::setDOFAperture(dof_aperture);
         ::setDOFFocusDistance(dof_focus_distance);
         ::setGolfDimpleCount(golf_dimple_count);
         ::setGolfDimpleRadius(golf_dimple_radius);
         ::setGolfDimpleDepth(golf_dimple_depth);
         ::setNEEFirstBounceOnly(nee_first_bounce_only);
         ::setNEEStride(nee_stride);
      };

      auto propagateAccumulationToggle = [&]()
      {
         if (accumulation_enabled != auto_accumulate)
            auto_accumulate = accumulation_enabled;
      };

      applySceneSettings();

      SDL_Event event;
      vector<unsigned char> display_image(image_width * image_height * image_channels);
      vector<unsigned char> base_display_image(image_width * image_height * image_channels);
      RenderTargetView display_view{&display_image, image_width, image_height, image_channels};

      void *d_rand_states = nullptr;
      void *d_accum_buffer = nullptr; // Persistent device accumulation buffer

      // Initialize CUDA display stream for async gamma-correct + D2H pipeline
      ::initCudaStreams();

      // Adaptive sampling state
      void *d_pixel_sample_counts = nullptr; // Per-pixel sample counts (null = disabled)
      bool adaptive_sampling_enabled = scene.adaptive_sampling;
      int min_adaptive_samples = 32;         // Don't check convergence before this many samples
      float adaptive_threshold = 3.16e-5f;   // Relative luminance change threshold (default ~10^-4.5)
      float convergence_pct = 0.0f;          // % of pixels that have converged (for display)
      bool show_heatmap = false;              // Toggle to display sample count heatmap
      int visualization_mode = static_cast<int>(VisualizationMode::NORMAL); // Visualization mode (normal vs show normals)
      bool show_spps_counter = true;          // SDL overlay throughput counter under logo

      // --- HDR Environment Map ---
      // Scan resources/hdri/ for .hdr files; index 0 = built-in gradient sky.
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
      int current_hdr_index = 0; // 0 = gradient sky

      // Lambda: load (or unload) an HDR environment by display index.
      // Uses the disk cache (.hdrcache sidecar) for repeated loads (~5-10x faster for 4K/8K).
      auto applyHdrChange = [&](int new_index)
      {
         new_index = std::max(0, std::min(new_index, hdr_count - 1));
         current_hdr_index = new_index;

         if (new_index == 0)
         {
            ::clearHdrEnvironment();
            std::cout << "HDR sky: Gradient Sky (built-in)\n";
         }
         else
         {
            const std::string &path = hdr_files[new_index - 1]; // entry 0 is gradient
            int w = 0, h = 0;
            auto half_data = loadHdrEnvHalf(path, w, h, hdr_cache);
            if (half_data.empty())
            {
               std::cerr << "HDR: Failed to load '" << path << "'\n";
               return;
            }
            if (!::uploadHdrEnvironmentHalf(half_data.data(), w, h))
            {
               std::cerr << "HDR: GPU upload failed for '" << path << "'\n";
               ::clearHdrEnvironment();
               current_hdr_index = 0;
               return;
            }
            std::cout << "HDR sky: '" << hdr_labels[new_index] << "' (" << w << "x" << h << ")\n";
         }
         camera_changed = true;
      };

      // Scene selection: built-ins + all YAML files discovered at runtime.
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
            if (ec)
               break;
            if (!entry.is_regular_file(ec))
               continue;

            fs::path path = entry.path();
            std::string ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c)
                           { return static_cast<char>(std::tolower(c)); });
            if (ext != ".yaml" && ext != ".yml")
               continue;

            std::string key;
            std::error_code canon_ec;
            fs::path canonical_path = fs::weakly_canonical(path, canon_ec);
            key = canon_ec ? path.lexically_normal().string() : canonical_path.string();

            if (seen_yaml_paths.insert(key).second)
            {
               yaml_files.push_back(path.lexically_normal().string());
            }
         }
      };

      // Path depend on where the main program is run, so check multiple likely locations for resources. This allows flexibility
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
      {
         scene_name_ptrs.push_back(entry.label.c_str());
      }

      const char *const *scene_names = scene_name_ptrs.empty() ? nullptr : scene_name_ptrs.data();
      const int scene_count = static_cast<int>(scene_name_ptrs.size());
      int current_scene_index = 0; // Start with whatever was passed in
      Scene::SceneDescription active_scene = scene; // Mutable copy
      Scene::SceneDescription original_scene = scene; // Keep original to restore materials

      auto applyVisualizationToActiveScene = [&]() {
         // Always start from original materials, then apply visualization override.
         active_scene = original_scene;
         if (visualization_mode == static_cast<int>(VisualizationMode::SHOW_NORMALS))
         {
            int material_index = active_scene.addMaterial(Scene::MaterialDesc::normal());
            for (auto &geom : active_scene.geometries)
            {
               geom.material_id = material_index;
            }
         }
      };

      // Build initial GPU scene
      CudaScene::Scene *gpu_scene = Scene::CudaSceneBuilder::buildGPUScene(active_scene);

      // Scan for procedural geometry/materials — must be after active_scene is initialized
      auto scanProceduralPatterns = [&]() {
         scene_has_golf_ball = false;
         for (const auto &g : active_scene.geometries)
         {
            if (g.type == Scene::GeometryType::DISPLACED_SPHERE)
               scene_has_golf_ball = true;
         }
      };
      scanProceduralPatterns(); // Initial scan

      auto applySceneSelectionChange = [&]() {
         if (current_scene_index < 0 || current_scene_index >= scene_count)
            current_scene_index = 0;

         const SceneEntry &selected = scene_entries[current_scene_index];
         std::cout << "Switching to scene: " << selected.label;
         if (!selected.yaml_path.empty())
            std::cout << " (" << selected.yaml_path << ")";
         std::cout << std::endl;

         if (current_scene_index == 0)
         {
            active_scene = Scene::SceneFactory::createDefaultScene();
         }
         else
         {
            active_scene = Scene::SceneFactory::fromYAML(selected.yaml_path);
         }

         // Update original_scene as well, then re-apply visualization mode.
         original_scene = active_scene;
         applyVisualizationToActiveScene();

         // Apply scene camera
         look_from = active_scene.camera_position;
         look_at = active_scene.camera_look_at;
         camera.vup = active_scene.camera_up;
         camera.vfov = active_scene.camera_fov;
         cam_fov_ui = static_cast<float>(camera.vfov);
         camera_control.initializeCameraControls(look_from, look_at);

         // Apply scene-specific rendering settings
         background_intensity = active_scene.background_intensity;
         adaptive_sampling_enabled = active_scene.adaptive_sampling;

         // Re-scan for procedural patterns so GUI sections update correctly
         scanProceduralPatterns();

         // Rebuild GPU scene
         Scene::CudaSceneBuilder::freeGPUScene(gpu_scene);
         gpu_scene = Scene::CudaSceneBuilder::buildGPUScene(active_scene);

         // Reset rendering state
         camera_changed = true;
         applySceneSettings();
      };

      // Timing for auto-orbit
      auto last_frame_time = std::chrono::high_resolution_clock::now();

      auto total_start = std::chrono::high_resolution_clock::now();

      // Main rendering loop
      while (running)
      {
         bool visualization_toggled_by_key = false;
         bool scene_switched_by_key = false;

         // Handle events
         while (SDLGuiHandler::pollEvent(event))
         {
            if (event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE))
            {
               running = false;
            }

            // Prevent camera/scene interaction if ImGui is using inputs
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
                  if (show_spps_counter) show_spps_counter = false;
                  else                  show_spps_counter = true;
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
                  // Reset to defaults
                  light_intensity = 1.0f;
                  background_intensity = 1.0f;
                  metal_fuzziness = 1.0f;
                  glass_refraction_index = 1.5f;
                  dof_enabled = false;
                  dof_aperture = 0.1f;
                  dof_focus_distance = 10.0f;
                  show_spps_counter = true;
                  gui.setLogoVisible(true);
                  samples_per_batch_float = static_cast<float>(settings_.samples_per_batch);
                  camera_control.setAutoOrbit(false);
                  // Reset MIS options to settings_ defaults
                  mis_enabled           = settings_.mis_enabled;
                  motion_gate_mis       = settings_.motion_gate_mis;
                  nee_first_bounce_only = settings_.nee_first_bounce_only;
                  nee_stride            = settings_.nee_stride;
                  use_sobol             = true;
                  camera_changed = true;
                  applySceneSettings();
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
               else if (camera_control.handleKeyDown(event, accumulation_enabled, samples_per_batch_float,
                                                      light_intensity, background_intensity, needs_rerender,
                                                      camera_changed))
               {
                  syncSamplesFromSlider();
                  if (camera_changed)
                     applySceneSettings();
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
               {
                  camera_changed = true;
               }
            }
            else if (event.type == SDL_MOUSEWHEEL)
            {
               if (camera_control.handleMouseWheel(event, look_from, look_at))
               {
                  camera_changed = true;
               }
            }
         }

         if (scene_switched_by_key)
         {
            applySceneSelectionChange();
         }

         if (visualization_toggled_by_key)
         {
            applyVisualizationToActiveScene();
            Scene::CudaSceneBuilder::freeGPUScene(gpu_scene);
            gpu_scene = Scene::CudaSceneBuilder::buildGPUScene(active_scene);
            camera_changed = true;
            applySceneSettings();
         }

         // Update auto-orbit
         auto current_frame_time = std::chrono::high_resolution_clock::now();
         std::chrono::duration<float> delta = current_frame_time - last_frame_time;
         last_frame_time = current_frame_time;
         if (delta.count() > 0.0f)
            current_fps = 1.0f / delta.count();

         if (camera_control.updateAutoOrbit(look_from, look_at, delta.count()))
         {
            camera_changed = true;
         }

         // Update motion detection
         auto now = std::chrono::high_resolution_clock::now();
         std::chrono::duration<float> time_since_last_change = now - last_camera_change_time;
         is_camera_moving = (time_since_last_change.count() < motion_cooldown_seconds);

         // Handle camera changes - restart rendering
         if (camera_changed)
         {
            camera_changed = false;
            current_samples = 0;
            force_immediate_render = true;

            last_camera_change_time = now;
            is_camera_moving = true;

            // Zero the device accumulation buffer in-place (no free/realloc)
            if (d_accum_buffer != nullptr)
            {
               ::resetDeviceAccumBuffer(d_accum_buffer, image_width * image_height);
            }

            // Reset adaptive sampling state so all pixels start fresh
            ::resetAdaptiveBuffer(d_pixel_sample_counts, image_width * image_height);
            convergence_pct = 0.0f;

            refreshCameraFrame();
         }

         // Re-display after slider change without re-rendering
         if (needs_rerender && current_samples > 0)
         {
            auto &display_img = *display_view.pixels;
            ::convertAccumToDisplayCUDA(d_accum_buffer, display_img.data(), display_view.width, display_view.height,
                                        display_view.channels, current_samples, gamma,
                                        adaptive_sampling_enabled ? d_pixel_sample_counts : nullptr);

            // Allow heatmap visualization refresh even when no new samples are rendered.
            if (show_heatmap && d_pixel_sample_counts != nullptr)
            {
               ::renderSampleHeatmapCUDA(d_pixel_sample_counts, display_img.data(), display_view.width,
                                         display_view.height, display_view.channels, current_samples);
            }

            // Keep a clean, overlay-free base image for deterministic per-frame compositing.
            base_display_image = display_image;

            if (target.pixels)
               *target.pixels = display_image;
            needs_rerender = false;
         }

         // Render logic
         // Stop rendering when max SPP reached, OR when adaptive sampling reports 100% convergence
         bool all_converged = adaptive_sampling_enabled && convergence_pct >= 100.0f;
         bool should_render = (current_samples < max_samples && !all_converged && !camera_changed && running) || force_immediate_render;
         bool needs_initial_render = current_samples == 0 && !accumulation_enabled;

         if (should_render && (accumulation_enabled || needs_initial_render || force_immediate_render))
         {
            force_immediate_render = false;

            syncSamplesFromSlider();
            adaptive_samples_per_batch = samples_per_batch;

            // Option A: motion-gate MIS — auto-disable NEE/MIS while the camera is moving
            // so the GPU spends its budget on ray throughput rather than shadow rays.
            bool effective_mis = mis_enabled && !(motion_gate_mis && is_camera_moving);
            ::setMISEnabled(effective_mis);

            auto frame_start = std::chrono::high_resolution_clock::now();

            // Allocate adaptive sampling buffer on first use (lazy init)
            if (adaptive_sampling_enabled && d_pixel_sample_counts == nullptr)
            {
               ::allocateAdaptiveBuffer(&d_pixel_sample_counts, image_width * image_height);
            }

            renderBatch(frame, display_view, current_samples, max_samples, adaptive_samples_per_batch, gamma,
                        d_rand_states, d_accum_buffer, gpu_scene, is_camera_moving, context,
                        adaptive_sampling_enabled ? d_pixel_sample_counts : nullptr,
                        min_adaptive_samples, adaptive_threshold);

            auto frame_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> frame_time = frame_end - frame_start;

            if (frame_time.count() > 0.0f)
            {
               // SPS = total samples computed per second (samples_per_pixel * pixel_count / time)
               float total_samples = static_cast<float>(adaptive_samples_per_batch) * image_width * image_height;
               current_sps = (total_samples * 1000.0f) / frame_time.count();
               // ms per sample-pass (one pass = all pixels get one more sample)
               current_ms_per_sample = frame_time.count() / static_cast<float>(adaptive_samples_per_batch);
            }

            // Update convergence percentage for GUI display (every 10th frame to avoid overhead)
            if (adaptive_sampling_enabled && d_pixel_sample_counts != nullptr && current_samples % 50 < adaptive_samples_per_batch)
            {
               int num_pixels = image_width * image_height;
               int converged = ::countConvergedPixels(d_pixel_sample_counts, num_pixels);
               convergence_pct = 100.0f * (float)converged / (float)num_pixels;
            }

            // Overlay heatmap if enabled (replaces the normal display with sample count visualization)
            if (show_heatmap && d_pixel_sample_counts != nullptr)
            {
               ::renderSampleHeatmapCUDA(d_pixel_sample_counts, display_image.data(), image_width, image_height,
                                         image_channels, current_samples);
            }

            // Keep a clean, overlay-free base image for deterministic per-frame compositing.
            base_display_image = display_image;

            if (target.pixels)
               *target.pixels = display_image;
         }
         else
         {
            SDL_Delay(16); // Cap CPU usage when not rendering
         }

         // Snapshot current values to detect ImGui changes
         bool old_dof = dof_enabled;
         float old_aperture = dof_aperture;
         float old_focus = dof_focus_distance;
         float old_light = light_intensity;
         float old_background = background_intensity;
         float old_fuzz = metal_fuzziness;
         float old_ior = glass_refraction_index;
         float old_cam_fov = cam_fov_ui;
         int old_scene_index = current_scene_index;
         bool old_adaptive = adaptive_sampling_enabled;
         float old_adaptive_thresh = adaptive_threshold;
         bool old_show_heatmap = show_heatmap;
         int old_visualization_mode = visualization_mode;
         int   old_golf_dimple_count  = golf_dimple_count;
         float old_golf_dimple_radius = golf_dimple_radius;
         float old_golf_dimple_depth  = golf_dimple_depth;
         int   old_hdr_index          = current_hdr_index;
         // MIS option snapshots
         bool old_mis_enabled           = mis_enabled;
         bool old_motion_gate_mis       = motion_gate_mis;
         bool old_nee_first_bounce_only = nee_first_bounce_only;
         int  old_nee_stride            = nee_stride;
         bool old_use_sobol             = use_sobol;

         // Draw ImGui UI — passes pointers so ImGui can modify values directly
         bool auto_orbit = camera_control.isAutoOrbitEnabled();

         float cam_pos[3] = {(float)look_from.x(), (float)look_from.y(), (float)look_from.z()};
         float cam_lookat[3] = {(float)look_at.x(), (float)look_at.y(), (float)look_at.z()};

         // Recompose from a stable base every frame so overlays don't stack over time.
         display_image = base_display_image;

         int tri_count = 0;
         for (const auto &g : active_scene.geometries)
            if (g.type == Scene::GeometryType::TRIANGLE) ++tri_count;

         gui.updateDisplay(display_image, image_channels, current_sps, current_ms_per_sample, current_fps, current_samples,
                           &dof_enabled, &dof_aperture, &dof_focus_distance, &light_intensity, &background_intensity,
                           &metal_fuzziness, &glass_refraction_index, &samples_per_batch_float, &accumulation_enabled,
                           &auto_orbit, &current_scene_index, scene_names, scene_count,
                           cam_pos, cam_lookat, &cam_fov_ui,
                           &adaptive_sampling_enabled, &adaptive_threshold, convergence_pct, &show_heatmap,
                           &visualization_mode, nullptr, nullptr,
                           nullptr, nullptr, &show_spps_counter, tri_count,
                           nullptr,
                           scene_has_golf_ball      ? &golf_dimple_count  : nullptr,
                           scene_has_golf_ball      ? &golf_dimple_radius : nullptr,
                           scene_has_golf_ball      ? &golf_dimple_depth  : nullptr,
                           &current_hdr_index,
                           hdr_count > 0 ? hdr_name_ptrs.data() : nullptr,
                           hdr_count,
                           &mis_enabled, &motion_gate_mis, &nee_first_bounce_only, &nee_stride,
                           &use_sobol);

         if (auto_orbit != camera_control.isAutoOrbitEnabled())
         {
            camera_control.setAutoOrbit(auto_orbit);
         }

         gui.drawLogo();
         gui.present();

         // Handle scene change from UI
         if (current_scene_index != old_scene_index)
         {
            applySceneSelectionChange();
         }

         // Handle visualization mode change
         if (visualization_mode != old_visualization_mode)
         {
            std::cout << "Switching visualization mode" << std::endl;
            applyVisualizationToActiveScene();
            
            // Rebuild GPU scene
            Scene::CudaSceneBuilder::freeGPUScene(gpu_scene);
            gpu_scene = Scene::CudaSceneBuilder::buildGPUScene(active_scene);
            
            // Reset rendering state
            camera_changed = true;
            applySceneSettings();
         }

         // Detect ImGui-driven HDR sky change
         if (current_hdr_index != old_hdr_index)
         {
            applyHdrChange(current_hdr_index);
         }

         // Detect if ImGui changed any scene parameter
         if (dof_enabled != old_dof || dof_aperture != old_aperture || dof_focus_distance != old_focus ||
             light_intensity != old_light || background_intensity != old_background || metal_fuzziness != old_fuzz ||
             glass_refraction_index != old_ior || cam_fov_ui != old_cam_fov ||
             golf_dimple_count != old_golf_dimple_count || golf_dimple_radius != old_golf_dimple_radius ||
             golf_dimple_depth != old_golf_dimple_depth)
         {
            if (cam_fov_ui != old_cam_fov)
            {
               camera.vfov = cam_fov_ui;
            }
            camera_changed = true;
            applySceneSettings();
         }

         // MIS option changes — push updated constants to GPU and restart accumulation
         if (nee_first_bounce_only != old_nee_first_bounce_only || nee_stride != old_nee_stride)
         {
            ::setNEEFirstBounceOnly(nee_first_bounce_only);
            ::setNEEStride(nee_stride);
            camera_changed = true;
         }
         if (mis_enabled != old_mis_enabled || motion_gate_mis != old_motion_gate_mis)
         {
            // effective_mis is recalculated each frame before renderBatch; just restart accumulation
            camera_changed = true;
         }
         if (use_sobol != old_use_sobol)
         {
            ::setSobolSampler(use_sobol);
            camera_changed = true;
         }

         // Adaptive sampling toggled or threshold changed — restart accumulation
         if (adaptive_sampling_enabled != old_adaptive || adaptive_threshold != old_adaptive_thresh)
         {
            // Turning off adaptive sampling also disables the heatmap
            if (!adaptive_sampling_enabled)
               show_heatmap = false;

            camera_changed = true;
         }

         if (show_heatmap != old_show_heatmap)
         {
            // Heatmap is a display overlay; force refresh even when sampling is finished.
            needs_rerender = true;
         }

         // Arrow overlay settings affect the composited display image even when sampling is done,
         // so force a re-conversion from accumulation buffer when they change.
      }

      auto total_end = std::chrono::high_resolution_clock::now();
      std::cout << "\nTotal session time: " << render::timeStr(total_end - total_start) << std::endl;

      // Cleanup device resources
      if (d_rand_states != nullptr)
      {
         freeDeviceRandomStates(d_rand_states);
      }
      if (d_accum_buffer != nullptr)
      {
         freeDeviceAccumBuffer(d_accum_buffer);
      }
      if (d_pixel_sample_counts != nullptr)
      {
         freeAdaptiveBuffer(d_pixel_sample_counts);
      }

      // Cleanup scene
      Scene::CudaSceneBuilder::freeGPUScene(gpu_scene);
      ::clearHdrEnvironment();
      ::cleanupCudaStreams();
      // gui is cleaned up by its destructor
   }

 private:
   Settings settings_{};

   /**
    * @brief Render a batch of samples using CUDA
    *
    * The accumulation buffer stays entirely on GPU. After rendering, gamma correction
    * is done on GPU and only the small uint8 display image is copied back to host.
    */
   void renderBatch(const CameraFrame &frame, RenderTargetView display_target, int &current_samples, int max_samples,
                    int samples_per_batch, float gamma, void *&d_rand_states, void *&d_accum_buffer,
                    CudaScene::Scene *gpu_scene, bool is_moving, RenderContext &context,
                    void *d_pixel_sample_counts = nullptr, int min_adaptive_samples = 32,
                    float adaptive_threshold = 0.01f)
   {
      int samples_before_batch = current_samples; // total accumulated BEFORE this batch
      current_samples += samples_per_batch;

      if (current_samples > max_samples)
         current_samples = max_samples;

      int actual_samples_to_add = current_samples - samples_before_batch;

      const int progressive_depth = frame.max_depth;

      // Call CUDA to render and accumulate samples — accum buffer stays on GPU (pass nullptr for host buffer)
      unsigned long long cuda_ray_count = ::renderPixelsCUDAAccumulative(
          nullptr, nullptr, gpu_scene, frame.image_width, frame.image_height, frame.camera_center.x(),
          frame.camera_center.y(), frame.camera_center.z(), frame.pixel00_loc.x(), frame.pixel00_loc.y(),
          frame.pixel00_loc.z(), frame.pixel_delta_u.x(), frame.pixel_delta_u.y(), frame.pixel_delta_u.z(),
          frame.pixel_delta_v.x(), frame.pixel_delta_v.y(), frame.pixel_delta_v.z(), actual_samples_to_add,
          samples_before_batch, progressive_depth, &d_rand_states, &d_accum_buffer, frame.u.x(), frame.u.y(), frame.u.z(),
          frame.v.x(), frame.v.y(), frame.v.z(), d_pixel_sample_counts, min_adaptive_samples, adaptive_threshold);

      context.ray_counter.fetch_add(cuda_ray_count, std::memory_order_relaxed);

      // GPU-side gamma correction -> copy only uint8 display image to host
      // Pass per-pixel sample counts so each pixel divides by its own count
      auto &display_image = *display_target.pixels;
      ::convertAccumToDisplayCUDA(d_accum_buffer, display_image.data(), display_target.width, display_target.height,
                                  display_target.channels, current_samples, gamma, d_pixel_sample_counts);
   }
};

#endif // SDL2_FOUND
