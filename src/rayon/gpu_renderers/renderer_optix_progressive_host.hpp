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
 *  - light_intensity / metal_fuzziness / glass_refraction_index sliders are visible in the UI
 *    but have no effect (OptiX bakes those into the SBT at scene-build time).
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
                                          float dof_focus_dist);
   void optixRendererDownloadAccum(float *host_accum_buffer, int width, int height);
   void optixRendererCleanup();
}

class RendererOptiXProgressive : public IRenderer
{
 public:
   struct Settings
   {
      int samples_per_batch = constants::INTERACTIVE_SAMPLES_PER_BATCH;
      int motion_samples = constants::INTERACTIVE_MOTION_SAMPLES;
      bool auto_accumulate = true;
      int target_fps = 60;
      bool adaptive_depth = false;
      bool adaptive_sampling = true; // no-op for OptiX; kept for UI parity
      GuiTheme theme = GuiTheme::NORD;
   };

   RendererOptiXProgressive() = default;
   explicit RendererOptiXProgressive(Settings settings) : settings_(settings) {}

   void setSettings(const Settings &settings) { settings_ = settings; }

   void render(const RenderRequest &request, RenderContext &context) override
   {
      int samples_per_batch = settings_.samples_per_batch;
      int motion_samples = settings_.motion_samples;
      bool auto_accumulate = settings_.auto_accumulate;
      int target_fps = settings_.target_fps;
      bool adaptive_depth = settings_.adaptive_depth;
      (void)target_fps;

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
      float light_intensity = 1.0f;        // UI only — baked into SBT at build time
      float background_intensity = scene.background_intensity;
      float metal_fuzziness = 1.0f;        // UI only — baked into SBT at build time
      float glass_refraction_index = 1.5f; // UI only — baked into SBT at build time
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

      // Motion detection
      bool is_camera_moving = false;
      auto last_camera_change_time = std::chrono::high_resolution_clock::now();
      const float motion_cooldown_seconds = 0.5f;

      int adaptive_samples_per_batch = samples_per_batch;
      int user_samples_per_batch = samples_per_batch;

      // Overlay / visualization state (mirrors CUDA renderer)
      int visualization_mode = static_cast<int>(VisualizationMode::NORMAL);
      bool show_normal_arrows = false;
      int normal_arrow_count = 2000;
      float normal_arrow_scale = 0.6f;
      float normal_arrow_thickness = 1.2f;
      bool show_spps_counter = true;
      bool show_heatmap = false; // no-op for OptiX (kept for UI parity)
      bool adaptive_sampling_enabled = false; // no-op for OptiX
      float adaptive_threshold = 3.16e-5f;    // no-op for OptiX
      float convergence_pct = 0.0f;

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
      Hittable_list cpu_scene_for_arrows = Scene::CPUSceneBuilder::buildCPUScene(original_scene);

      auto applyVisualizationToActiveScene = [&]()
      {
         active_scene = original_scene;
         if (visualization_mode == static_cast<int>(VisualizationMode::SHOW_NORMALS))
         {
            int mat = active_scene.addMaterial(Scene::MaterialDesc::normal());
            for (auto &geom : active_scene.geometries)
               geom.material_id = mat;
         }
      };

      auto applySceneSelectionChange = [&]()
      {
         if (current_scene_index < 0 || current_scene_index >= scene_count)
            current_scene_index = 0;

         const SceneEntry &selected = scene_entries[current_scene_index];
         std::cout << "Switching to scene: " << selected.label;
         if (!selected.yaml_path.empty())
            std::cout << " (" << selected.yaml_path << ")";
         std::cout << std::endl;

         if (current_scene_index == 0)
            active_scene = Scene::SceneFactory::createDefaultScene();
         else
            active_scene = Scene::SceneFactory::fromYAML(selected.yaml_path, /*skip_cpu_bvh=*/true);

         original_scene = active_scene;
         cpu_scene_for_arrows = Scene::CPUSceneBuilder::buildCPUScene(original_scene);
         applyVisualizationToActiveScene();

         look_from = active_scene.camera_position;
         look_at = active_scene.camera_look_at;
         camera.vup = active_scene.camera_up;
         camera.vfov = active_scene.camera_fov;
         cam_fov_ui = static_cast<float>(camera.vfov);
         camera_control.initializeCameraControls(look_from, look_at);

         background_intensity = active_scene.background_intensity;

         // Rebuild OptiX scene
         optixRendererBuildScene(active_scene);
         camera_changed = true;
      };

      // CPU normal-arrow overlay helpers (identical to CUDA renderer)
      auto drawLineRGB = [&](std::vector<unsigned char> &img, int x0, int y0, int x1, int y1,
                              unsigned char r, unsigned char g, unsigned char b, float thickness)
      {
         int dx = std::abs(x1 - x0); int sx = x0 < x1 ? 1 : -1;
         int dy = -std::abs(y1 - y0); int sy = y0 < y1 ? 1 : -1;
         int err = dx + dy;
         const float radius_f = std::max(0.0f, thickness - 1.0f);
         const int radius = static_cast<int>(std::ceil(radius_f + 1.0f));
         while (true)
         {
            for (int oy = -radius; oy <= radius; ++oy)
            {
               const int py = y0 + oy;
               if (py < 0 || py >= image_height) continue;
               for (int ox = -radius; ox <= radius; ++ox)
               {
                  const int px = x0 + ox;
                  if (px < 0 || px >= image_width) continue;
                  const float dist = std::sqrt(static_cast<float>(ox * ox + oy * oy));
                  const float base = radius_f + 1.0f - dist;
                  const float coverage = std::clamp(std::pow(std::max(0.0f, base), 0.7f) * 1.8f, 0.0f, 1.0f);
                  if (coverage <= 0.0f) continue;
                  const int idx = (py * image_width + px) * image_channels;
                  img[idx+0] = static_cast<unsigned char>((1.f-coverage)*img[idx+0] + coverage*r);
                  img[idx+1] = static_cast<unsigned char>((1.f-coverage)*img[idx+1] + coverage*g);
                  img[idx+2] = static_cast<unsigned char>((1.f-coverage)*img[idx+2] + coverage*b);
               }
            }
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 >= dy) { err += dy; x0 += sx; }
            if (e2 <= dx) { err += dx; y0 += sy; }
         }
      };

      auto drawCPUArrowOverlay = [&](std::vector<unsigned char> &img)
      {
         if (!show_normal_arrows || normal_arrow_count <= 0) return;
         const int pixel_count = image_width * image_height;
         const float target_density = static_cast<float>(pixel_count) / static_cast<float>(normal_arrow_count);
         const int step = std::max(6, static_cast<int>(std::sqrt(std::max(1.0f, target_density))));
         const float arrow_len = std::max(4.0f, normal_arrow_scale * static_cast<float>(step));
         const float head_len = 0.35f * arrow_len;
         const float c = 0.8660254f; const float s = 0.5f;
         Hit_record rec;
         for (int y = step / 2; y < image_height; y += step)
         {
            for (int x = step / 2; x < image_width; x += step)
            {
               Point3 pixel_center = frame.pixel00_loc + static_cast<double>(x) * frame.pixel_delta_u
                                     + static_cast<double>(y) * frame.pixel_delta_v;
               Ray ray_r(frame.camera_center, pixel_center - frame.camera_center);
               if (!cpu_scene_for_arrows.hit(ray_r, Interval(0.0001, inf), rec)) continue;
               const double sx_n = dot(rec.normal, frame.u);
               const double sy_n = -dot(rec.normal, frame.v);
               const double mag2 = sx_n * sx_n + sy_n * sy_n;
               if (mag2 < 1e-8) continue;
               const double inv_mag = 1.0 / std::sqrt(mag2);
               const double dir_x = sx_n * inv_mag; const double dir_y = sy_n * inv_mag;
               const int tip_x = static_cast<int>(std::lround(x + dir_x * arrow_len));
               const int tip_y = static_cast<int>(std::lround(y + dir_y * arrow_len));
               unsigned char rr = static_cast<unsigned char>(127.5*(rec.normal.x()+1.0));
               unsigned char gg = static_cast<unsigned char>(127.5*(rec.normal.y()+1.0));
               unsigned char bb = static_cast<unsigned char>(127.5*(rec.normal.z()+1.0));
               drawLineRGB(img, x, y, tip_x, tip_y, rr, gg, bb, normal_arrow_thickness);
               const double bx = -dir_x, by = -dir_y;
               drawLineRGB(img, tip_x, tip_y,
                           static_cast<int>(std::lround(tip_x+(bx*c-by*s)*head_len)),
                           static_cast<int>(std::lround(tip_y+(bx*s+by*c)*head_len)), rr, gg, bb, normal_arrow_thickness);
               drawLineRGB(img, tip_x, tip_y,
                           static_cast<int>(std::lround(tip_x+(bx*c+by*s)*head_len)),
                           static_cast<int>(std::lround(tip_y+(-bx*s+by*c)*head_len)), rr, gg, bb, normal_arrow_thickness);
            }
         }
      };

      // Rendering buffers
      SDL_Event event;
      std::vector<unsigned char> display_image(image_width * image_height * image_channels);
      std::vector<unsigned char> base_display_image(image_width * image_height * image_channels);
      std::vector<float> accum_buffer(image_width * image_height * image_channels, 0.0f);
      RenderTargetView display_view{&display_image, image_width, image_height, image_channels};

      // Initialize OptiX pipeline and build scene
      optixRendererInit();
      optixRendererBuildScene(active_scene);

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
                  light_intensity = 1.0f; background_intensity = 1.0f;
                  metal_fuzziness = 1.0f; glass_refraction_index = 1.5f;
                  dof_enabled = false; dof_aperture = 0.1f; dof_focus_distance = 10.0f;
                  show_normal_arrows = false; normal_arrow_count = 2000;
                  normal_arrow_scale = 0.6f; normal_arrow_thickness = 1.2f;
                  show_spps_counter = true;
                  gui.setLogoVisible(true);
                  samples_per_batch_float = static_cast<float>(settings_.samples_per_batch);
                  camera_control.setAutoOrbit(false);
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
               else if (event.key.keysym.sym == SDLK_a)
               {
                  show_normal_arrows = !show_normal_arrows;
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
         is_camera_moving = (time_since_last_change.count() < motion_cooldown_seconds);

         // Camera changed → reset accumulation
         if (camera_changed)
         {
            camera_changed = false;
            current_samples = 0;
            force_immediate_render = true;
            std::fill(accum_buffer.begin(), accum_buffer.end(), 0.0f);
            last_camera_change_time = now;
            is_camera_moving = true;
            optixRendererResetAccum(image_width, image_height);
            refreshCameraFrame();
         }

         // Redisplay after overlay/settings change without adding new samples
         if (needs_rerender && current_samples > 0)
         {
            render::convertAccumBufferToImage(display_view, accum_buffer, current_samples, gamma);
            base_display_image = display_image;
            display_image = base_display_image;
            drawCPUArrowOverlay(display_image);
            if (target.pixels) *target.pixels = display_image;
            needs_rerender = false;
         }

         bool should_render = (current_samples < max_samples && !camera_changed && running) || force_immediate_render;
         bool needs_initial_render = current_samples == 0 && !accumulation_enabled;

         if (should_render && (accumulation_enabled || needs_initial_render || force_immediate_render))
         {
            force_immediate_render = false;
            syncSamplesFromSlider();
            user_samples_per_batch = samples_per_batch;

            adaptive_samples_per_batch = is_camera_moving ? motion_samples : user_samples_per_batch;

            auto frame_start = std::chrono::high_resolution_clock::now();

            const int num_materials_active = static_cast<int>(active_scene.materials.size());
            renderBatch(frame, accum_buffer, display_view, current_samples, max_samples,
                        adaptive_samples_per_batch, gamma, is_camera_moving, adaptive_depth, context,
                        num_materials_active, background_intensity, dof_enabled, dof_aperture, dof_focus_distance);

            auto frame_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> frame_time = frame_end - frame_start;
            if (frame_time.count() > 0.0f)
            {
               float total_samples = static_cast<float>(adaptive_samples_per_batch) * image_width * image_height;
               current_sps = (total_samples * 1000.0f) / frame_time.count();
               current_ms_per_sample = frame_time.count() / static_cast<float>(adaptive_samples_per_batch);
            }

            if (is_camera_moving)
               adaptive_samples_per_batch = motion_samples;

            base_display_image = display_image;
            display_image = base_display_image;
            drawCPUArrowOverlay(display_image);

            if (target.pixels) *target.pixels = display_image;
         }
         else
         {
            SDL_Delay(16);
         }

         // Snapshot current values to detect ImGui changes
         bool old_dof = dof_enabled;
         float old_aperture = dof_aperture;
         float old_focus = dof_focus_distance;
         float old_background = background_intensity;
         float old_cam_fov = cam_fov_ui;
         int old_scene_index = current_scene_index;
         int old_visualization_mode = visualization_mode;
         bool old_show_normal_arrows = show_normal_arrows;
         int old_normal_arrow_count = normal_arrow_count;
         float old_normal_arrow_scale = normal_arrow_scale;
         float old_normal_arrow_thickness = normal_arrow_thickness;

         bool auto_orbit = camera_control.isAutoOrbitEnabled();
         float cam_pos[3] = {(float)look_from.x(), (float)look_from.y(), (float)look_from.z()};
         float cam_lookat[3] = {(float)look_at.x(), (float)look_at.y(), (float)look_at.z()};

         int tri_count = 0;
         for (const auto &g : active_scene.geometries)
            if (g.type == Scene::GeometryType::TRIANGLE) ++tri_count;

         gui.updateDisplay(display_image, image_channels, current_sps, current_ms_per_sample, current_fps, current_samples,
                           &dof_enabled, &dof_aperture, &dof_focus_distance, &light_intensity,
                           &background_intensity, &metal_fuzziness, &glass_refraction_index,
                           &samples_per_batch_float, &accumulation_enabled,
                           &auto_orbit, &current_scene_index, scene_names, scene_count,
                           cam_pos, cam_lookat, &cam_fov_ui,
                           &adaptive_sampling_enabled, &adaptive_threshold, convergence_pct, &show_heatmap,
                           &visualization_mode, &show_normal_arrows, &normal_arrow_count,
                           &normal_arrow_scale, &normal_arrow_thickness, &show_spps_counter, tri_count);

         if (auto_orbit != camera_control.isAutoOrbitEnabled())
            camera_control.setAutoOrbit(auto_orbit);

         gui.drawLogo();
         gui.present();

         // Handle scene change from ImGui
         if (current_scene_index != old_scene_index)
            applySceneSelectionChange();

         // Handle visualization mode change from ImGui
         if (visualization_mode != old_visualization_mode)
         {
            applyVisualizationToActiveScene();
            optixRendererBuildScene(active_scene);
            camera_changed = true;
         }

         // Detect if ImGui changed rendering parameters
         if (dof_enabled != old_dof || dof_aperture != old_aperture || dof_focus_distance != old_focus ||
             background_intensity != old_background || cam_fov_ui != old_cam_fov)
         {
            if (cam_fov_ui != old_cam_fov)
               camera.vfov = cam_fov_ui;
            camera_changed = true;
         }

         // Arrow overlay settings changed — refresh display
         if (show_normal_arrows != old_show_normal_arrows || normal_arrow_count != old_normal_arrow_count ||
             normal_arrow_scale != old_normal_arrow_scale || normal_arrow_thickness != old_normal_arrow_thickness)
         {
            needs_rerender = true;
         }
      }

      auto total_end = std::chrono::high_resolution_clock::now();
      std::cout << "\nTotal session time: " << render::timeStr(total_end - total_start) << std::endl;

      optixRendererCleanup();
   }

 private:
   Settings settings_{};

   int calculateProgressiveMaxDepth(int current_samples, bool is_moving, int max_depth) const
   {
      if (is_moving) return 3;
      if (current_samples <= 4)   return 4;
      if (current_samples <= 16)  return 5;
      if (current_samples <= 32)  return 6;
      if (current_samples <= 64)  return 7;
      if (current_samples <= 128) return 8;
      if (current_samples <= 256) return 16;
      if (current_samples <= 512) return 16;
      if (current_samples <= 1024) return 24;
      return std::min(512, max_depth);
   }

   void renderBatch(const CameraFrame &frame, std::vector<float> &accum_buffer, RenderTargetView display_target,
                    int &current_samples, int max_samples, int samples_per_batch, float gamma,
                    bool is_moving, bool adaptive_depth, RenderContext &context, int num_materials,
                    float background_intensity, bool dof_enabled, float dof_aperture, float dof_focus_distance)
   {
      // If we've already reached or exceeded the maximum, do not render more samples.
      if (current_samples >= max_samples) return;

      const int remaining       = max_samples - current_samples;
      const int actual_samples  = std::min(samples_per_batch, remaining);
      const int new_total_samples = current_samples + actual_samples;

      const int depth = adaptive_depth
                            ? calculateProgressiveMaxDepth(new_total_samples, is_moving, frame.max_depth)
                            : frame.max_depth;

      unsigned long long ray_count = optixRendererLaunch(
          frame.image_width, frame.image_height, num_materials,
          actual_samples, new_total_samples, depth,
          static_cast<float>(frame.camera_center.x()), static_cast<float>(frame.camera_center.y()),
          static_cast<float>(frame.camera_center.z()),
          static_cast<float>(frame.pixel00_loc.x()), static_cast<float>(frame.pixel00_loc.y()),
          static_cast<float>(frame.pixel00_loc.z()),
          static_cast<float>(frame.pixel_delta_u.x()), static_cast<float>(frame.pixel_delta_u.y()),
          static_cast<float>(frame.pixel_delta_u.z()),
          static_cast<float>(frame.pixel_delta_v.x()), static_cast<float>(frame.pixel_delta_v.y()),
          static_cast<float>(frame.pixel_delta_v.z()),
          static_cast<float>(frame.u.x()), static_cast<float>(frame.u.y()), static_cast<float>(frame.u.z()),
          static_cast<float>(frame.v.x()), static_cast<float>(frame.v.y()), static_cast<float>(frame.v.z()),
          background_intensity, dof_enabled, dof_aperture, dof_focus_distance);

      // Download GPU accumulation buffer to host for display
      optixRendererDownloadAccum(accum_buffer.data(), frame.image_width, frame.image_height);

      context.ray_counter.fetch_add(ray_count, std::memory_order_relaxed);

      current_samples = new_total_samples;
      render::convertAccumBufferToImage(display_target, accum_buffer, current_samples, gamma);
   }

   // (display is now composited inline in the main loop — no displayFrame helper needed)
};

#endif // SDL2_FOUND && OPTIX_FOUND
