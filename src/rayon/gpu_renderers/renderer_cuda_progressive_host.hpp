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
#include "sdl_gui_controls.hpp"
#include "sdl_gui_handler.hpp"
#include "scenes/scene_factory.hpp"

#include <SDL.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <filesystem>

class RendererCUDAProgressive : public IRenderer
{
 public:
   struct Settings
   {
      int samples_per_batch = 8;
      bool auto_accumulate = true;
      int target_fps = 60;
      bool adaptive_depth = false;
      bool debug_mode = false;
      std::string initial_scene_path;
   };

   RendererCUDAProgressive() = default;
   explicit RendererCUDAProgressive(Settings settings) : settings_(settings) {}

   void setSettings(const Settings &settings) { settings_ = settings; }

   /**
    * @brief Interactive SDL rendering with continuous sample accumulation
    */
   void render(const RenderRequest &request, RenderContext &context) override
   {
      int samples_per_batch = settings_.samples_per_batch;
      bool auto_accumulate = settings_.auto_accumulate;
      int target_fps = settings_.target_fps;
      bool adaptive_depth = settings_.adaptive_depth;
      bool debug_mode = settings_.debug_mode;

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
      SDLGuiHandler gui(target.width, target.height);
      if (!gui.initialize())
         return;
      int max_samples = camera.samples_per_pixel;

      // Initialize camera controls
      CameraControlHandler camera_control;
      camera_control.initializeCameraControls(look_from, look_at);

      // --- SCENE CATEGORY SETUP ---
      std::vector<SceneCategory> categories;

      categories.push_back({
          "Phong Shading Demo",
          {"../resources/experiments/phong-shading-demo", "resources/experiments/phong-shading-demo"},
          {"obj", "yaml"},
          {},
          -1
      });
      
      categories.push_back({
          "Triangle Demo",
          {"../resources/experiments/triangle-demo", "resources/experiments/triangle-demo"},
          {"yaml"},
          {},
          -1
      });

      categories.push_back({
          "Scenes",
          {"../resources/scenes", "resources/scenes", "."},
          {"yaml", "yml"},
          {},
          -1
      });

      categories.push_back({
          "Benchmark",
          {"../resources/experiments/benchmark", "resources/experiments/benchmark"},
          {"yaml", "yml", "obj"},
          {},
          -1
      });

      categories.push_back({
          "Simple Obj",
          {"../resources/experiments/simple-obj", "resources/experiments/simple-obj"},
          {"obj"},
          {},
          -1
      });

      // SCANNING
      try {
          for (auto& cat : categories) {
              for (const auto& search_path : cat.search_paths) {
                  if (std::filesystem::exists(search_path)) {
                      for (const auto & entry : std::filesystem::directory_iterator(search_path)) {
                          if (entry.is_regular_file()) {
                             std::string path = entry.path().string();
                             std::string ext = path.substr(path.find_last_of(".") + 1);
                             std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                             
                             // Check if extension matches any allowed
                             bool match = false;
                             for (const auto& allowed : cat.allowed_extensions) {
                                 if (ext == allowed) { match = true; break; }
                             }

                             if (match) {
                                 if (std::find(cat.files.begin(), cat.files.end(), path) == cat.files.end()) {
                                     cat.files.push_back(path);
                                 }
                             }
                          }
                      }
                  }
              }
              std::sort(cat.files.begin(), cat.files.end());
              // Auto-select first if available
              if (!cat.files.empty()) cat.current_index = -1; // Don't select by default to avoid auto-loading
          }
      } catch (const std::exception& e) {
          cerr << "Error scanning scenes: " << e.what() << std::endl;
      }

      int active_category_idx = 0; // Default to "Scenes"

      // Auto-select based on initial_scene_path
      if (!settings_.initial_scene_path.empty()) {
          try {
              std::filesystem::path initial_path = std::filesystem::absolute(settings_.initial_scene_path);
              bool found = false;
              
              // 1. Try exact match
              for (int i = 0; i < static_cast<int>(categories.size()); ++i) {
                  auto& cat = categories[i];
                  for (int j = 0; j < static_cast<int>(cat.files.size()); ++j) {
                      std::filesystem::path cat_file_path = std::filesystem::absolute(cat.files[j]);
                      // Handle potential errors with equivalent
                      try {
                          if (std::filesystem::equivalent(initial_path, cat_file_path)) {
                              active_category_idx = i;
                              cat.current_index = j;
                              found = true;
                              break;
                          }
                      } catch (...) { continue; }
                  }
                  if (found) break;
              }
              
              // 2. Fallback: simple filename match
              if (!found) {
                  std::string initial_name = initial_path.filename().string();
                  for (int i = 0; i < static_cast<int>(categories.size()); ++i) {
                      auto& cat = categories[i];
                      for (int j = 0; j < static_cast<int>(cat.files.size()); ++j) {
                           if (std::filesystem::path(cat.files[j]).filename().string() == initial_name) {
                               active_category_idx = i;
                               cat.current_index = j;
                               found = true;
                               break;
                           }
                      }
                      if (found) break;
                  }
              }
          } catch (...) {
              // Ignore filesystem errors during matching
          }
      }

      bool force_tab_update = !settings_.initial_scene_path.empty();

      // Ray-tracing state
      bool running = true;
      bool scene_loaded = true; // Start with scene loaded from request
      bool camera_changed = true;
      bool accumulation_enabled = auto_accumulate;
      int current_samples = 0;
      float gamma = 2.0f; // Fixed gamma value
      
      // Initialize from scene description
      float light_intensity = request.scene.light_intensity;
      float background_intensity = request.scene.ambient_light;
      float metal_fuzziness = request.scene.global_metal_fuzziness;
      float glass_refraction_index = request.scene.global_glass_ior;
      bool dof_enabled = request.scene.dof_enabled;
      float dof_aperture = request.scene.dof_aperture;
      float dof_focus_distance = request.scene.dof_focus_distance;
      
      float fov = static_cast<float>(camera.vfov);
      float initial_fov = fov;
      bool needs_rerender = false;
      bool force_immediate_render = false; // Flag to force rendering immediately after state change
      float samples_per_batch_float = static_cast<float>(samples_per_batch); // Float version for slider
      float current_sps = 0.0f;           // Track Samples Per Second for UI
      float current_ms_per_sample = 0.0f; // Track Time Per Sample for UI

      // Motion detection for adaptive quality
      bool is_camera_moving = false;
      auto last_camera_change_time = std::chrono::high_resolution_clock::now();
      const float motion_cooldown_seconds = 0.5f; // Wait 0.5s after last input before considering stopped

      // Adaptive sample rate for smooth target FPS
      int adaptive_samples_per_batch = samples_per_batch;      // Actual samples to render (adapts during motion)
      int user_samples_per_batch = samples_per_batch;          // User's preferred samples (from UI slider)
      const float target_frame_time_ms = 1000.0f / target_fps; // Calculate target frame time from FPS
      const float adaptive_speed = 0.2f;                       // How quickly to adapt sample rate (lower = smoother)

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
      vector<float> accum_buffer(image_width * image_height * image_channels, 0.0f);
      RenderTargetView display_view{&display_image, image_width, image_height, image_channels};

      void *d_rand_states = nullptr;
      void *d_accum_buffer = nullptr; // Persistent device accumulation buffer

      // Create a local copy of the scene to allow switching
      Scene::SceneDescription active_scene = request.scene;
      background_intensity = active_scene.ambient_light;

      // Build scene once
      CudaScene::Scene *gpu_scene = Scene::CudaSceneBuilder::buildGPUScene(active_scene);
      
      int sphere_count = 0;
      int rect_count = 0;
      int tri_count = 0;
      
      auto updateSceneStats = [&]() {
          sphere_count = 0;
          rect_count = 0;
          tri_count = 0;
          for (const auto& geom : active_scene.geometries) {
              switch (geom.type) {
                  case Scene::GeometryType::SPHERE:
                  case Scene::GeometryType::DISPLACED_SPHERE:
                      sphere_count++;
                      break;
                  case Scene::GeometryType::RECTANGLE:
                      rect_count++;
                      break;
                  case Scene::GeometryType::TRIANGLE:
                      tri_count++;
                      break;
                  case Scene::GeometryType::TRIANGLE_MESH:
                      if (geom.data.mesh_instance.mesh_id >= 0 && geom.data.mesh_instance.mesh_id < active_scene.meshes.size()) {
                          tri_count += active_scene.meshes[geom.data.mesh_instance.mesh_id].triangles.size();
                      }
                      break;
                  default: break;
              }
          }
      };
      
      updateSceneStats();

      // Timing for auto-orbit
      auto last_frame_time = std::chrono::high_resolution_clock::now();
      auto total_start = std::chrono::high_resolution_clock::now();
      bool load_scene_request = false;

      // Helper for loading scenes
      auto loadSelectedScene = [&]() {
         if (active_category_idx < 0 || active_category_idx >= static_cast<int>(categories.size())) return; 
         
         SceneCategory& cat = categories[active_category_idx];
         if (cat.current_index < 0 || cat.current_index >= static_cast<int>(cat.files.size())) return;

         std::string file_to_load = cat.files[cat.current_index];

         if (!file_to_load.empty()) {
             std::cout << "Switching to: " << file_to_load << std::endl;
             
             Scene::CudaSceneBuilder::freeGPUScene(gpu_scene);
             active_scene = Scene::SceneFactory::load(file_to_load);
             
             camera.look_from = active_scene.camera_position;
             camera.look_at = active_scene.camera_look_at;
             camera.vup = active_scene.camera_up;
             camera.vfov = active_scene.camera_fov;
             refreshCameraFrame();
             camera_control.initializeCameraControls(look_from, look_at);
             updateSceneStats();

             gpu_scene = Scene::CudaSceneBuilder::buildGPUScene(active_scene);
             
             current_samples = 0;
             std::fill(accum_buffer.begin(), accum_buffer.end(), 0.0f);
             if (d_accum_buffer != nullptr) {
                 freeDeviceAccumBuffer(d_accum_buffer);
                 d_accum_buffer = nullptr;
             }
             
             // Update interactive state from new scene
             background_intensity = active_scene.ambient_light;
             light_intensity = active_scene.light_intensity;
             metal_fuzziness = active_scene.global_metal_fuzziness;
             glass_refraction_index = active_scene.global_glass_ior;
             dof_enabled = active_scene.dof_enabled;
             dof_aperture = active_scene.dof_aperture;
             dof_focus_distance = active_scene.dof_focus_distance;

             force_immediate_render = true;
             camera_changed = true;
             applySceneSettings();
             load_scene_request = false; // Reset flag
             scene_loaded = true; // Mark scene as loaded
         }
      };

      // Main rendering loop
      while (running)
      {

         // Handle events
         while (gui.pollEvent(event))
         {
            if (event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE))
            {
               running = false;
            }

            // Prevent camera/scene interaction if ImGui is using inputs
            ImGuiIO& io = ImGui::GetIO();
            if (io.WantCaptureMouse)
            {
               if (event.type == SDL_MOUSEBUTTONDOWN || event.type == SDL_MOUSEBUTTONUP ||
                   event.type == SDL_MOUSEMOTION || event.type == SDL_MOUSEWHEEL)
                  continue;
            }
            if (io.WantCaptureKeyboard)
            {
               if (event.type == SDL_KEYDOWN || event.type == SDL_KEYUP)
                  continue;
            }
            
            // Only handle scene-related inputs if loaded or if it's a load command
            if (event.type == SDL_KEYDOWN)
            {
               if (event.key.keysym.sym == SDLK_h)
               {
                  gui.toggleControls();
               }
               else if (event.key.keysym.sym == SDLK_m)
               {
                  // Cycle Category Mode
                  active_category_idx = (active_category_idx + 1) % categories.size();
                  force_tab_update = true;
                  
                  // Auto-select first if none selected
                  if (categories[active_category_idx].current_index == -1 && !categories[active_category_idx].files.empty()) {
                      categories[active_category_idx].current_index = 0;
                  }
                  
                  load_scene_request = true;
               }
               else if ((event.key.keysym.sym == SDLK_n || event.key.keysym.sym == SDLK_p))
               {
                  // Cycle Files in Active Category
                  SceneCategory& cat = categories[active_category_idx];
                  if (!cat.files.empty()) {
                      bool is_next = (event.key.keysym.sym == SDLK_n);
                      
                      if (cat.current_index == -1) cat.current_index = 0;
                      else {
                          if (is_next) cat.current_index = (cat.current_index + 1) % cat.files.size();
                          else cat.current_index = (cat.current_index - 1 + cat.files.size()) % cat.files.size();
                      }
                      load_scene_request = true;
                  }
               }
               else if (scene_loaded && event.key.keysym.sym == SDLK_c)
               {
                  gui.toggleHeaderCollapse();
               }
               else if (scene_loaded && event.key.keysym.sym == SDLK_r)
               {
                  // Reset to scene defaults
                  light_intensity = active_scene.light_intensity;
                  background_intensity = active_scene.ambient_light;
                  metal_fuzziness = active_scene.global_metal_fuzziness;
                  glass_refraction_index = active_scene.global_glass_ior;
                  dof_enabled = active_scene.dof_enabled;
                  dof_aperture = active_scene.dof_aperture;
                  dof_focus_distance = active_scene.dof_focus_distance;
                  fov = initial_fov;
                  samples_per_batch_float = static_cast<float>(settings_.samples_per_batch);
                  camera_control.setAutoOrbit(false);

                  camera.vfov = fov;
                  refreshCameraFrame();

                  camera_changed = true;
                  applySceneSettings();
               }
               else if (scene_loaded && camera_control.handleKeyDown(event, accumulation_enabled, samples_per_batch_float,
                                                     light_intensity, background_intensity, needs_rerender,
                                                     camera_changed))
               {
                  syncSamplesFromSlider();
                  if (camera_changed)
                     applySceneSettings();
                  propagateAccumulationToggle();
               }
            }
            else if (scene_loaded && event.type == SDL_MOUSEBUTTONDOWN)
            {
               syncSamplesFromSlider();

               if (camera_control.handleMouseButtonDown(event))
               {
                  syncSamplesFromSlider();

                  if (camera_changed)
                     applySceneSettings();
                  propagateAccumulationToggle();
               }
            }
            else if (scene_loaded && event.type == SDL_MOUSEBUTTONUP)
            {
               camera_control.handleMouseButtonUp(event);
            }
            else if (scene_loaded && event.type == SDL_MOUSEMOTION)
            {
               if (camera_control.handleMouseMotion(
                       event, look_from, look_at, vup, basis_w))
               {
                  syncSamplesFromSlider();

                  if (camera_changed)
                     applySceneSettings();
                  camera_changed = true;
               }
            }
            else if (scene_loaded && event.type == SDL_MOUSEWHEEL)
            {
               if (camera_control.handleMouseWheel(event, look_from, look_at))
               {
                  camera_changed = true;
               }
            }
         }

         // Update auto-orbit
         auto current_frame_time = std::chrono::high_resolution_clock::now();
         std::chrono::duration<float> delta = current_frame_time - last_frame_time;
         last_frame_time = current_frame_time;

         if (scene_loaded && camera_control.updateAutoOrbit(look_from, look_at, delta.count()))
         {
            camera_changed = true;
         }

         // Update motion detection
         auto now = std::chrono::high_resolution_clock::now();
         std::chrono::duration<float> time_since_last_change = now - last_camera_change_time;
         is_camera_moving = (time_since_last_change.count() < motion_cooldown_seconds);

         // Handle camera changes
         if (scene_loaded && camera_changed)
         {
            camera_changed = false;
            current_samples = 0;
            force_immediate_render = true; 
            std::fill(accum_buffer.begin(), accum_buffer.end(), 0.0f);
            last_camera_change_time = now;
            is_camera_moving = true;

            if (d_accum_buffer != nullptr)
            {
               freeDeviceAccumBuffer(d_accum_buffer);
               d_accum_buffer = nullptr;
            }

            refreshCameraFrame();
         }

         if (scene_loaded && needs_rerender && current_samples > 0)
         {
            render::convertAccumBufferToImage(display_view, accum_buffer, current_samples, gamma);
            if (target.pixels) *target.pixels = display_image;
            needs_rerender = false;
         }

         // Render logic
         bool should_render = scene_loaded && ((current_samples < max_samples && !camera_changed && running) || force_immediate_render);
         bool needs_initial_render = current_samples == 0 && !accumulation_enabled;

         if (should_render && (accumulation_enabled || needs_initial_render || force_immediate_render))
         {
            force_immediate_render = false;
            syncSamplesFromSlider();
            user_samples_per_batch = samples_per_batch;

            if (is_camera_moving) adaptive_samples_per_batch = std::max(5, adaptive_samples_per_batch);
            else adaptive_samples_per_batch = user_samples_per_batch;

            auto frame_start = std::chrono::high_resolution_clock::now();

            renderBatch(frame, accum_buffer, display_view, current_samples, max_samples, adaptive_samples_per_batch,
                        gamma, d_rand_states, d_accum_buffer, gpu_scene, is_camera_moving, adaptive_depth, context);

            auto frame_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> frame_time = frame_end - frame_start;
            
            if (frame_time.count() > 0.0f)
            {
               current_sps = (static_cast<float>(adaptive_samples_per_batch) * 1000.0f) / frame_time.count();
               current_ms_per_sample = frame_time.count() / static_cast<float>(adaptive_samples_per_batch);
            }

            if (is_camera_moving) adaptive_samples_per_batch = 5;
            if (target.pixels) *target.pixels = display_image;
         }
         else
         {
             if (!scene_loaded)
             {
                 // Clear screen if no scene loaded
                 std::fill(display_image.begin(), display_image.end(), 0);
             }
             SDL_Delay(16); 
         }

         // Check for UI changes
         bool old_dof = dof_enabled;
         float old_aperture = dof_aperture;
         float old_focus = dof_focus_distance;
         float old_fov = fov;
         float old_light = light_intensity;
         float old_background = background_intensity;
         float old_fuzz = metal_fuzziness;
         float old_ior = glass_refraction_index;

         // Find first light source for debug info
         Vec3 light_pos(0,0,0);
         bool has_light = false;
         if (settings_.debug_mode) {
             for(const auto& geom : active_scene.geometries) {
                 if (geom.material_id >= 0 && geom.material_id < active_scene.materials.size()) {
                     const auto& mat = active_scene.materials[geom.material_id];
                     if (mat.type == Scene::MaterialType::LIGHT || mat.emission.length_squared() > 0) {
                         if (geom.type == Scene::GeometryType::RECTANGLE) {
                             light_pos = geom.data.rectangle.corner + geom.data.rectangle.u * 0.5f + geom.data.rectangle.v * 0.5f;
                         } else if (geom.type == Scene::GeometryType::SPHERE) {
                             light_pos = geom.data.sphere.center;
                         } else if (geom.type == Scene::GeometryType::DISPLACED_SPHERE) {
                             light_pos = geom.data.displaced_sphere.center;
                         }
                         has_light = true;
                         break; // Just show the first one
                     }
                 }
             }
         }

         displayFrame(
             gui, display_image, current_samples, &samples_per_batch_float, &light_intensity, &background_intensity,
             &metal_fuzziness, &glass_refraction_index, &accumulation_enabled, camera_control, &dof_enabled,
             &dof_aperture, &dof_focus_distance, &fov, 
             image_channels, current_sps, current_ms_per_sample,
             categories, &active_category_idx,
             force_tab_update, &load_scene_request,
             debug_mode, look_from, look_at, vup, has_light, light_pos,
             sphere_count, rect_count, tri_count);

         if (load_scene_request)
         {
             loadSelectedScene();
         }

         if (dof_enabled != old_dof || dof_aperture != old_aperture || dof_focus_distance != old_focus || fov != old_fov ||
             light_intensity != old_light || background_intensity != old_background || metal_fuzziness != old_fuzz || glass_refraction_index != old_ior)
         {
            camera_changed = true;
            applySceneSettings();
            if (fov != old_fov)
            {
               camera.vfov = fov;
               refreshCameraFrame();
            }
         }
         
         force_tab_update = false; // Reset per frame at the END
      }

      auto total_end = std::chrono::high_resolution_clock::now();
      std::cout << "\nTotal session time: " << render::timeStr(total_end - total_start) << std::endl;

      if (d_rand_states != nullptr) freeDeviceRandomStates(d_rand_states);
      if (d_accum_buffer != nullptr) freeDeviceAccumBuffer(d_accum_buffer);
      Scene::CudaSceneBuilder::freeGPUScene(gpu_scene);
      gui.cleanup();
   }

 private:
   Settings settings_{};

   // Lightweight view describing which buffer SDL should read from after gamma correction.
   struct DisplayBufferView
   {
      const vector<unsigned char> *buffer;
      int channels;
   };

   /**
    * @brief Calculate progressive max depth based on accumulated samples
    * Starts at depth 1, gradually increases to max 256 for final quality
    */
   int calculateProgressiveMaxDepth(int current_samples, bool is_moving, int max_depth) const
   {
      // During camera motion, use reduced depth for faster preview
      if (is_moving)
         return 3; // Fast preview during motion

      // Progressive depth schedule for smooth quality ramp-up
      if (current_samples <= 4) {
         return 4; // First few samples: depth 1 (fastest preview)
      } else if (current_samples <= 16) {
         return 5; // Quick preview: depth 2
      } else if (current_samples <= 32) {
         return 6; // Early quality: depth 3
      } else if (current_samples <= 64) {
         return 7; // Building detail: depth 4
      } else if (current_samples <= 128) {
         return 8; // Good quality: depth 6
      } else if (current_samples <= 256) {
         return 16; // High quality: depth 8
      } else if (current_samples <= 512) {
         return 16; // Very high quality: depth 12
      } else if (current_samples <= 1024) {
         return 24; // Excellent quality: depth 16
      } else {
         return std::min(512, max_depth); // Final quality: depth up to 256
      }
   }

   /**
    * @brief Render a batch of samples using CUDA
    */
   void renderBatch(const CameraFrame &frame, vector<float> &accum_buffer, RenderTargetView display_target,
                    int &current_samples, int max_samples, int samples_per_batch, float gamma, void *&d_rand_states,
                    void *&d_accum_buffer, CudaScene::Scene *gpu_scene, bool is_moving, bool adaptive_depth,
                    RenderContext &context)
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

      // Call CUDA to render and accumulate samples with progressive depth
      // Note: First parameter (image) is unused by the kernel - it only updates accum_buffer
      unsigned long long cuda_ray_count = ::renderPixelsCUDAAccumulative(
          nullptr, accum_buffer.data(), gpu_scene, frame.image_width, frame.image_height, frame.camera_center.x(),
          frame.camera_center.y(), frame.camera_center.z(), frame.pixel00_loc.x(), frame.pixel00_loc.y(),
          frame.pixel00_loc.z(), frame.pixel_delta_u.x(), frame.pixel_delta_u.y(), frame.pixel_delta_u.z(),
          frame.pixel_delta_v.x(), frame.pixel_delta_v.y(), frame.pixel_delta_v.z(), actual_samples_to_add,
          current_samples, progressive_depth, &d_rand_states, &d_accum_buffer, frame.u.x(), frame.u.y(), frame.u.z(),
          frame.v.x(), frame.v.y(), frame.v.z());

      context.ray_counter.fetch_add(cuda_ray_count, std::memory_order_relaxed);

      render::convertAccumBufferToImage(display_target, accum_buffer, current_samples,
                                        gamma); // Keep display + disk paths identical.
   }

   /**
    * @brief Update the display with current frame and UI
    */
   void displayFrame(SDLGuiHandler &gui, const vector<unsigned char> &display_image, int current_samples,
                     float* samples_per_batch, float* light_intensity, float* background_intensity, float* metal_fuzziness,
                     float* glass_refraction_index, bool* accumulation_enabled, CameraControlHandler& camera_control, bool* dof_enabled,
                     float* dof_aperture, float* dof_focus_distance, float* fov,
                     int image_channels, float sps, float ms_per_sample,
                     std::vector<SceneCategory>& categories, int* active_category_idx,
                     bool force_tab_update,
                     bool* load_scene_request,
                     bool debug_mode, const Vec3& cam_pos, const Vec3& cam_look_at, const Vec3& cam_up,
                     bool has_light, const Vec3& light_pos,
                     int sphere_count, int rect_count, int tri_count)
   {
      bool is_orbiting = camera_control.isAutoOrbitEnabled();

      gui.updateDisplay(display_image, image_channels, sps, ms_per_sample, current_samples,
                        dof_enabled, dof_aperture, dof_focus_distance, fov,
                        light_intensity, background_intensity, metal_fuzziness, glass_refraction_index,
                        samples_per_batch, accumulation_enabled, &is_orbiting,
                        categories, active_category_idx, force_tab_update, load_scene_request,
                        debug_mode, cam_pos, cam_look_at, cam_up, has_light, light_pos,
                        sphere_count, rect_count, tri_count);
      
      if (is_orbiting != camera_control.isAutoOrbitEnabled())
      {
         camera_control.setAutoOrbit(is_orbiting);
      }

      gui.drawLogo();
      gui.present();
   }
};

#endif // SDL2_FOUND