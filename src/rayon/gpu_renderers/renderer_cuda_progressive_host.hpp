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
      int motion_samples = constants::INTERACTIVE_MOTION_SAMPLES;
      bool auto_accumulate = true;
      bool adaptive_depth = false;
      bool adaptive_sampling = true;
      GuiTheme theme = GuiTheme::NORD;
   };

   RendererCUDAProgressive() = default;
   explicit RendererCUDAProgressive(Settings settings) : settings_(settings) {}

   void setSettings(const Settings &settings) { settings_ = settings; }

   void render(const RenderRequest &request, RenderContext &context) override
   {
      int samples_per_batch = settings_.samples_per_batch;
      int motion_samples = settings_.motion_samples;
      bool auto_accumulate = settings_.auto_accumulate;
      bool adaptive_depth = settings_.adaptive_depth;

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
      SDLGuiHandler gui(target.width, target.height, settings_.theme);
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

      // Motion detection for adaptive quality
      bool is_camera_moving = false;
      auto last_camera_change_time = std::chrono::high_resolution_clock::now();
      const float motion_cooldown_seconds = 0.5f;

      // Adaptive sample rate
      int adaptive_samples_per_batch = samples_per_batch;
      int user_samples_per_batch = samples_per_batch;

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
      };

      auto propagateAccumulationToggle = [&]()
      {
         if (accumulation_enabled != auto_accumulate)
            auto_accumulate = accumulation_enabled;
      };

      applySceneSettings();

      // Rendering buffers
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
      bool show_normal_arrows = false;        // CPU overlay of normal vectors
      int normal_arrow_count = 2000;           // Target number of arrows on screen
      float normal_arrow_scale = 0.6f;        // Arrow length multiplier
      float normal_arrow_thickness = 1.2f;    // Arrow thickness in pixels
      bool show_spps_counter = true;          // SDL overlay throughput counter under logo

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
      Hittable_list cpu_scene_for_arrows = Scene::CPUSceneBuilder::buildCPUScene(original_scene);

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

      auto drawLineRGB = [&](vector<unsigned char> &img, int x0, int y0, int x1, int y1, unsigned char r,
                    unsigned char g, unsigned char b, float thickness) {
         int dx = std::abs(x1 - x0);
         int sx = x0 < x1 ? 1 : -1;
         int dy = -std::abs(y1 - y0);
         int sy = y0 < y1 ? 1 : -1;
         int err = dx + dy;
         const float radius_f = std::max(0.0f, thickness - 1.0f);
         const int radius = static_cast<int>(std::ceil(radius_f + 1.0f));

         while (true)
         {
            for (int oy = -radius; oy <= radius; ++oy)
            {
               const int py = y0 + oy;
               if (py < 0 || py >= image_height)
               {
                  continue;
               }
               for (int ox = -radius; ox <= radius; ++ox)
               {
                  const int px = x0 + ox;
                  if (px < 0 || px >= image_width)
                  {
                     continue;
                  }

                  const float dist = std::sqrt(static_cast<float>(ox * ox + oy * oy));
                  // Coverage ramps from center pixel to neighbors as thickness increases.
                  // Nonlinear gain makes 1.0/1.1/1.2 visibly different without hard jumps.
                  const float base = radius_f + 1.0f - dist;
                  const float coverage = std::clamp(std::pow(std::max(0.0f, base), 0.7f) * 1.8f, 0.0f, 1.0f);
                  if (coverage <= 0.0f)
                  {
                     continue;
                  }

                  const int idx = (py * image_width + px) * image_channels;
                  img[idx + 0] = static_cast<unsigned char>(
                      (1.0f - coverage) * static_cast<float>(img[idx + 0]) + coverage * static_cast<float>(r));
                  img[idx + 1] = static_cast<unsigned char>(
                      (1.0f - coverage) * static_cast<float>(img[idx + 1]) + coverage * static_cast<float>(g));
                  img[idx + 2] = static_cast<unsigned char>(
                      (1.0f - coverage) * static_cast<float>(img[idx + 2]) + coverage * static_cast<float>(b));
               }
            }
            if (x0 == x1 && y0 == y1)
               break;
            int e2 = 2 * err;
            if (e2 >= dy)
            {
               err += dy;
               x0 += sx;
            }
            if (e2 <= dx)
            {
               err += dx;
               y0 += sy;
            }
         }
      };

      auto drawCPUArrowOverlay = [&](vector<unsigned char> &img) {
         if (!show_normal_arrows || normal_arrow_count <= 0)
         {
            return;
         }

         const int pixel_count = image_width * image_height;
         const float target_density = static_cast<float>(pixel_count) / static_cast<float>(normal_arrow_count);
         const int step = std::max(6, static_cast<int>(std::sqrt(std::max(1.0f, target_density))));
         const float arrow_len = std::max(4.0f, normal_arrow_scale * static_cast<float>(step));
         const float head_len = 0.35f * arrow_len;
         const float c = 0.8660254f; // cos(30 deg)
         const float s = 0.5f;       // sin(30 deg)

         Hit_record rec;
         for (int y = step / 2; y < image_height; y += step)
         {
            for (int x = step / 2; x < image_width; x += step)
            {
               Point3 pixel_center = frame.pixel00_loc + static_cast<double>(x) * frame.pixel_delta_u +
                                     static_cast<double>(y) * frame.pixel_delta_v;
               Ray r(frame.camera_center, pixel_center - frame.camera_center);
               if (!cpu_scene_for_arrows.hit(r, Interval(0.0001, inf), rec))
               {
                  continue;
               }

               const double sx_n = dot(rec.normal, frame.u);
               const double sy_n = -dot(rec.normal, frame.v);
               const double mag2 = sx_n * sx_n + sy_n * sy_n;
               if (mag2 < 1e-8)
               {
                  continue;
               }

               const double inv_mag = 1.0 / std::sqrt(mag2);
               const double dir_x = sx_n * inv_mag;
               const double dir_y = sy_n * inv_mag;

               const int tip_x = static_cast<int>(std::lround(static_cast<double>(x) + dir_x * arrow_len));
               const int tip_y = static_cast<int>(std::lround(static_cast<double>(y) + dir_y * arrow_len));

               unsigned char rr = static_cast<unsigned char>(127.5 * (rec.normal.x() + 1.0));
               unsigned char gg = static_cast<unsigned char>(127.5 * (rec.normal.y() + 1.0));
               unsigned char bb = static_cast<unsigned char>(127.5 * (rec.normal.z() + 1.0));

               drawLineRGB(img, x, y, tip_x, tip_y, rr, gg, bb, normal_arrow_thickness);

               const double back_x = -dir_x;
               const double back_y = -dir_y;

               const double left_x = back_x * c - back_y * s;
               const double left_y = back_x * s + back_y * c;
               const double right_x = back_x * c + back_y * s;
               const double right_y = -back_x * s + back_y * c;

               const int left_tip_x = static_cast<int>(std::lround(static_cast<double>(tip_x) + left_x * head_len));
               const int left_tip_y = static_cast<int>(std::lround(static_cast<double>(tip_y) + left_y * head_len));
               const int right_tip_x =
                   static_cast<int>(std::lround(static_cast<double>(tip_x) + right_x * head_len));
               const int right_tip_y =
                   static_cast<int>(std::lround(static_cast<double>(tip_y) + right_y * head_len));

               drawLineRGB(img, tip_x, tip_y, left_tip_x, left_tip_y, rr, gg, bb, normal_arrow_thickness);
               drawLineRGB(img, tip_x, tip_y, right_tip_x, right_tip_y, rr, gg, bb, normal_arrow_thickness);
            }
         }
      };

      // Build initial GPU scene
      CudaScene::Scene *gpu_scene = Scene::CudaSceneBuilder::buildGPUScene(active_scene);

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
         cpu_scene_for_arrows = Scene::CPUSceneBuilder::buildCPUScene(original_scene);
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
                  show_normal_arrows = false;
                  normal_arrow_count = 2000;
                  normal_arrow_scale = 0.6f;
                  normal_arrow_thickness = 1.2f;
                  show_spps_counter = true;
                  gui.setLogoVisible(true);
                  samples_per_batch_float = static_cast<float>(settings_.samples_per_batch);
                  camera_control.setAutoOrbit(false);
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
               else if (event.key.keysym.sym == SDLK_a)
               {
                  show_normal_arrows = !show_normal_arrows;
                  // Overlay is composited in display buffer; force refresh even if accumulation stopped.
                  needs_rerender = true;
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
            user_samples_per_batch = samples_per_batch;

            if (is_camera_moving)
            {
               adaptive_samples_per_batch = motion_samples;
            }
            else
            {
               adaptive_samples_per_batch = user_samples_per_batch;
            }

            auto frame_start = std::chrono::high_resolution_clock::now();

            // Allocate adaptive sampling buffer on first use (lazy init)
            if (adaptive_sampling_enabled && d_pixel_sample_counts == nullptr)
            {
               ::allocateAdaptiveBuffer(&d_pixel_sample_counts, image_width * image_height);
            }

            renderBatch(frame, display_view, current_samples, max_samples, adaptive_samples_per_batch, gamma,
                        d_rand_states, d_accum_buffer, gpu_scene, is_camera_moving, adaptive_depth, context,
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

            if (is_camera_moving)
            {
               adaptive_samples_per_batch = motion_samples;
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
         bool old_show_normal_arrows = show_normal_arrows;
         int old_normal_arrow_count = normal_arrow_count;
         float old_normal_arrow_scale = normal_arrow_scale;
         float old_normal_arrow_thickness = normal_arrow_thickness;

         // Draw ImGui UI — passes pointers so ImGui can modify values directly
         bool auto_orbit = camera_control.isAutoOrbitEnabled();

         float cam_pos[3] = {(float)look_from.x(), (float)look_from.y(), (float)look_from.z()};
         float cam_lookat[3] = {(float)look_at.x(), (float)look_at.y(), (float)look_at.z()};

         // Recompose from a stable base every frame so overlays don't stack over time.
         display_image = base_display_image;
         drawCPUArrowOverlay(display_image);

         int tri_count = 0;
         for (const auto &g : active_scene.geometries)
            if (g.type == Scene::GeometryType::TRIANGLE) ++tri_count;

         gui.updateDisplay(display_image, image_channels, current_sps, current_ms_per_sample, current_samples,
                           &dof_enabled, &dof_aperture, &dof_focus_distance, &light_intensity, &background_intensity,
                           &metal_fuzziness, &glass_refraction_index, &samples_per_batch_float, &accumulation_enabled,
                           &auto_orbit, &current_scene_index, scene_names, scene_count,
                           cam_pos, cam_lookat, &cam_fov_ui,
                           &adaptive_sampling_enabled, &adaptive_threshold, convergence_pct, &show_heatmap,
                           &visualization_mode, &show_normal_arrows, &normal_arrow_count,
                           &normal_arrow_scale, &normal_arrow_thickness, &show_spps_counter, tri_count);

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

         // Detect if ImGui changed any scene parameter
         if (dof_enabled != old_dof || dof_aperture != old_aperture || dof_focus_distance != old_focus ||
             light_intensity != old_light || background_intensity != old_background || metal_fuzziness != old_fuzz ||
             glass_refraction_index != old_ior || cam_fov_ui != old_cam_fov)
         {
            if (cam_fov_ui != old_cam_fov)
            {
               camera.vfov = cam_fov_ui;
            }
            camera_changed = true;
            applySceneSettings();
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
         if (show_normal_arrows != old_show_normal_arrows || normal_arrow_count != old_normal_arrow_count ||
             normal_arrow_scale != old_normal_arrow_scale || normal_arrow_thickness != old_normal_arrow_thickness)
         {
            needs_rerender = true;
         }
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
      ::cleanupCudaStreams();
      // gui is cleaned up by its destructor
   }

 private:
   Settings settings_{};

   /**
    * @brief Calculate progressive max depth based on accumulated samples
    */
   int calculateProgressiveMaxDepth(int current_samples, bool is_moving, int max_depth) const
   {
      if (is_moving)
         return 3;

      if (current_samples <= 4)
         return 4;
      else if (current_samples <= 16)
         return 5;
      else if (current_samples <= 32)
         return 6;
      else if (current_samples <= 64)
         return 7;
      else if (current_samples <= 128)
         return 8;
      else if (current_samples <= 256)
         return 16;
      else if (current_samples <= 512)
         return 16;
      else if (current_samples <= 1024)
         return 24;
      else
         return std::min(512, max_depth);
   }

   /**
    * @brief Render a batch of samples using CUDA
    *
    * The accumulation buffer stays entirely on GPU. After rendering, gamma correction
    * is done on GPU and only the small uint8 display image is copied back to host.
    */
   void renderBatch(const CameraFrame &frame, RenderTargetView display_target, int &current_samples, int max_samples,
                    int samples_per_batch, float gamma, void *&d_rand_states, void *&d_accum_buffer,
                    CudaScene::Scene *gpu_scene, bool is_moving, bool adaptive_depth, RenderContext &context,
                    void *d_pixel_sample_counts = nullptr, int min_adaptive_samples = 32,
                    float adaptive_threshold = 0.01f)
   {
      current_samples += samples_per_batch;

      if (current_samples > max_samples)
         current_samples = max_samples;

      int actual_samples_to_add = samples_per_batch;
      if (current_samples > max_samples)
         actual_samples_to_add = max_samples - (current_samples - samples_per_batch);

      const int progressive_depth =
          adaptive_depth ? calculateProgressiveMaxDepth(current_samples, is_moving, frame.max_depth)
                         : frame.max_depth;

      // Call CUDA to render and accumulate samples — accum buffer stays on GPU (pass nullptr for host buffer)
      unsigned long long cuda_ray_count = ::renderPixelsCUDAAccumulative(
          nullptr, nullptr, gpu_scene, frame.image_width, frame.image_height, frame.camera_center.x(),
          frame.camera_center.y(), frame.camera_center.z(), frame.pixel00_loc.x(), frame.pixel00_loc.y(),
          frame.pixel00_loc.z(), frame.pixel_delta_u.x(), frame.pixel_delta_u.y(), frame.pixel_delta_u.z(),
          frame.pixel_delta_v.x(), frame.pixel_delta_v.y(), frame.pixel_delta_v.z(), actual_samples_to_add,
          current_samples, progressive_depth, &d_rand_states, &d_accum_buffer, frame.u.x(), frame.u.y(), frame.u.z(),
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
