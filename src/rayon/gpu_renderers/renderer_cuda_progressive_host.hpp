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
#include <chrono>
#include <iostream>
#include <vector>

class RendererCUDAProgressive : public IRenderer
{
 public:
   struct Settings
   {
      int samples_per_batch = 8;
      int motion_samples = 10;
      bool auto_accumulate = true;
      int target_fps = 60;
      bool adaptive_depth = false;
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
      int target_fps = settings_.target_fps;
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
      float background_intensity = 1.0f;
      float metal_fuzziness = 1.0f;
      float glass_refraction_index = 1.5f;
      bool dof_enabled = false;
      float dof_aperture = 0.1f;
      float dof_focus_distance = 10.0f;
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
      RenderTargetView display_view{&display_image, image_width, image_height, image_channels};

      void *d_rand_states = nullptr;
      void *d_accum_buffer = nullptr; // Persistent device accumulation buffer

      // Scene selection
      static const char *scene_names[] = {"Default Scene", "Single Object", "Cornell Box (YAML)",
                                          "Simple Scene (YAML)", "BVH Test (YAML)"};
      static const int scene_count = 5;
      int current_scene_index = 0; // Start with whatever was passed in
      Scene::SceneDescription active_scene = scene; // Mutable copy

      // Build initial GPU scene
      CudaScene::Scene *gpu_scene = Scene::CudaSceneBuilder::buildGPUScene(active_scene);

      // Timing for auto-orbit
      auto last_frame_time = std::chrono::high_resolution_clock::now();

      auto total_start = std::chrono::high_resolution_clock::now();

      // Main rendering loop
      while (running)
      {
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
            if (event.type == SDL_KEYDOWN)
            {
               if (event.key.keysym.sym == SDLK_h)
               {
                  gui.toggleControls();
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
                  samples_per_batch_float = static_cast<float>(settings_.samples_per_batch);
                  camera_control.setAutoOrbit(false);
                  camera_changed = true;
                  applySceneSettings();
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

            refreshCameraFrame();
         }

         // Re-display after slider change without re-rendering
         if (needs_rerender && current_samples > 0)
         {
            auto &display_img = *display_view.pixels;
            ::convertAccumToDisplayCUDA(d_accum_buffer, display_img.data(), display_view.width, display_view.height,
                                        display_view.channels, current_samples, gamma);

            if (target.pixels)
               *target.pixels = display_image;
            needs_rerender = false;
         }

         // Render logic
         bool should_render = (current_samples < max_samples && !camera_changed && running) || force_immediate_render;
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

            renderBatch(frame, display_view, current_samples, max_samples, adaptive_samples_per_batch, gamma,
                        d_rand_states, d_accum_buffer, gpu_scene, is_camera_moving, adaptive_depth, context);

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
         int old_scene_index = current_scene_index;

         // Draw ImGui UI — passes pointers so ImGui can modify values directly
         bool auto_orbit = camera_control.isAutoOrbitEnabled();

         float cam_pos[3] = {(float)look_from.x(), (float)look_from.y(), (float)look_from.z()};
         float cam_lookat[3] = {(float)look_at.x(), (float)look_at.y(), (float)look_at.z()};
         gui.updateDisplay(display_image, image_channels, current_sps, current_ms_per_sample, current_samples,
                           &dof_enabled, &dof_aperture, &dof_focus_distance, &light_intensity, &background_intensity,
                           &metal_fuzziness, &glass_refraction_index, &samples_per_batch_float, &accumulation_enabled,
                           &auto_orbit, &current_scene_index, scene_names, scene_count,
                           cam_pos, cam_lookat, (float)camera.vfov);

         if (auto_orbit != camera_control.isAutoOrbitEnabled())
         {
            camera_control.setAutoOrbit(auto_orbit);
         }

         gui.drawLogo();
         gui.present();

         // Handle scene change from UI
         if (current_scene_index != old_scene_index)
         {
            std::cout << "Switching to scene: " << scene_names[current_scene_index] << std::endl;
            switch (current_scene_index)
            {
            case 0:
               active_scene = Scene::SceneFactory::createDefaultScene();
               break;
            case 1:
               active_scene = Scene::SceneFactory::singleObjectScene();
               break;
            case 2:
               active_scene = Scene::SceneFactory::fromYAML("../resources/cornell_box.yaml");
               break;
            case 3:
               active_scene = Scene::SceneFactory::fromYAML("../resources/simple_scene.yaml");
               break;
            case 4:
               active_scene = Scene::SceneFactory::fromYAML("../resources/bvh_test_scene.yaml");
               break;
            }

            // Apply scene camera
            look_from = active_scene.camera_position;
            look_at = active_scene.camera_look_at;
            camera.vup = active_scene.camera_up;
            camera.vfov = active_scene.camera_fov;
            camera_control.initializeCameraControls(look_from, look_at);

            // Apply scene-specific rendering settings
            background_intensity = active_scene.background_intensity;

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
             glass_refraction_index != old_ior)
         {
            camera_changed = true;
            applySceneSettings();
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

      // Cleanup scene
      Scene::CudaSceneBuilder::freeGPUScene(gpu_scene);
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
                    CudaScene::Scene *gpu_scene, bool is_moving, bool adaptive_depth, RenderContext &context)
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
          frame.v.x(), frame.v.y(), frame.v.z());

      context.ray_counter.fetch_add(cuda_ray_count, std::memory_order_relaxed);

      // GPU-side gamma correction -> copy only uint8 display image to host
      auto &display_image = *display_target.pixels;
      ::convertAccumToDisplayCUDA(d_accum_buffer, display_image.data(), display_target.width, display_target.height,
                                  display_target.channels, current_samples, gamma);
   }
};

#endif // SDL2_FOUND
