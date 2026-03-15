/**
 * @class SDLGuiHandler
 * @brief Handles SDL window management and ImGui integration
 *
 * This class manages:
 * - SDL window and renderer creation/cleanup
 * - Main ray-tracing texture display
 * - Logo loading and overlay
 * - ImGui context initialization and frame rendering
 * - Basic SDL event polling
 * - Color theme selection
 */
#pragma once

#include "constants.hpp"
#ifdef SDL2_FOUND

#include <SDL.h>
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "external/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "external/stb_image_resize2.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;

enum class GuiTheme : int
{
   DARK = 0,
   LIGHT,
   CLASSIC,
   NORD,
   DRACULA,
   GRUVBOX,
   CATPPUCCIN,
   COUNT // must be last
};

static const char *themeNames[] = {"Dark", "Light", "Classic", "Nord", "Dracula", "Gruvbox", "Catppuccin Mocha"};

enum class VisualizationMode : int
{
   NORMAL = 0,
   SHOW_NORMALS,
   COUNT // must be last
};

static const char *visualizationModeNames[] = {"Normal Render", "Show Normals"};

class SDLGuiHandler
{
 public:
   SDLGuiHandler(int image_width, int image_height, GuiTheme initial_theme = GuiTheme::NORD)
       : image_width(image_width), image_height(image_height), show_controls(true), collapse_headers(false),
         reset_headers(false), window_collapse_requested(false), window_collapsed(false), current_theme(initial_theme), initialized(false), window(nullptr), renderer(nullptr),
         texture(nullptr), logo_texture(nullptr), perf_write_index(0), last_perf_sample_time_ms(0),
         perf_sample_interval_ms(50), perf_time_window_ms(20000)
   {
   }

   ~SDLGuiHandler() { cleanup(); }

   bool initialize()
   {
      if (SDL_Init(SDL_INIT_VIDEO) < 0)
      {
         cerr << "SDL initialization failed: " << SDL_GetError() << "\n";
         return false;
      }

      std::string window_title =
          "RayON v" + std::string(constants::version) + "";
      window = SDL_CreateWindow(window_title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, image_width,
                                image_height, SDL_WINDOW_SHOWN);
      if (!window)
      {
         cerr << "Window creation failed: " << SDL_GetError() << "\n";
         cleanupSDL();
         return false;
      }

      // Set window icon from ISC logo
      {
         int icon_w, icon_h, icon_ch;
         unsigned char *icon_data = stbi_load("../resources/ISC_logo_rvb.png", &icon_w, &icon_h, &icon_ch, 4);
         if (icon_data)
         {
            SDL_Surface *icon_surface = SDL_CreateRGBSurfaceFrom(
                icon_data, icon_w, icon_h, 32, icon_w * 4,
                0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);
            if (icon_surface)
            {
               SDL_SetWindowIcon(window, icon_surface);
               SDL_FreeSurface(icon_surface);
            }
            stbi_image_free(icon_data);
         }
      }

      renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
      if (!renderer)
      {
         cerr << "Renderer creation failed: " << SDL_GetError() << "\n";
         SDL_DestroyWindow(window);
         cleanupSDL();
         return false;
      }

      texture =
          SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, image_width, image_height);
      if (!texture)
      {
         cerr << "Texture creation failed: " << SDL_GetError() << "\n";
         SDL_DestroyRenderer(renderer);
         SDL_DestroyWindow(window);
         cleanupSDL();
         return false;
      }

      loadLogo();

      // Initialize ImGui
      IMGUI_CHECKVERSION();
      ImGui::CreateContext();
      ImGuiIO &io = ImGui::GetIO();
      (void)io;

      applyTheme(current_theme);

      ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
      ImGui_ImplSDLRenderer2_Init(renderer);
      initialized = true;
      return true;
   }

   void cleanup()
   {
      if (!initialized)
         return;
      initialized = false;

      ImGui_ImplSDLRenderer2_Shutdown();
      ImGui_ImplSDL2_Shutdown();
      ImGui::DestroyContext();

      if (logo_texture)
      {
         SDL_DestroyTexture(logo_texture);
         logo_texture = nullptr;
      }
      if (texture)
      {
         SDL_DestroyTexture(texture);
         texture = nullptr;
      }
      if (renderer)
      {
         SDL_DestroyRenderer(renderer);
         renderer = nullptr;
      }
      if (window)
      {
         SDL_DestroyWindow(window);
         window = nullptr;
      }

      cleanupSDL();
   }

   /**
    * @brief Display the raytraced image and render the ImGui overlay
    *
    * All UI controls are drawn here using ImGui. Parameters are passed as pointers
    * so ImGui can modify them directly.
    */
   void updateDisplay(const vector<unsigned char> &image, int image_channels, float sps, float ms_per_sample, float fps, int spp,
                      bool *dof_enabled, float *aperture, float *focus_dist, float *light_intensity,
                      float *background_intensity, float *metal_fuzziness, float *glass_ior,
                      float *samples_per_batch, bool *auto_accumulate, bool *auto_orbit,
                      int *scene_index, const char *const *scene_names, int scene_count,
                      const float *cam_pos = nullptr, const float *cam_lookat = nullptr, float *cam_fov = nullptr,
                      bool *adaptive_sampling = nullptr, float *adaptive_threshold = nullptr,
                      float convergence_pct = 0.0f, bool *show_heatmap = nullptr,
                      int *visualization_mode = nullptr,
                      bool *show_normal_arrows = nullptr, int *normal_arrow_count = nullptr,
                      float *normal_arrow_scale = nullptr, float *normal_arrow_thickness = nullptr,
                      bool *show_spps_counter = nullptr, int triangle_count = 0,
                      int *target_fps_ptr = nullptr)
   {
      SDL_UpdateTexture(texture, nullptr, image.data(), image_width * image_channels);
      SDL_RenderClear(renderer);
      SDL_RenderCopy(renderer, texture, nullptr, nullptr);

      // Begin ImGui frame
      ImGui_ImplSDLRenderer2_NewFrame();
      ImGui_ImplSDL2_NewFrame();
      ImGui::NewFrame();

      if (show_controls)
      {
         ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
         if (window_collapse_requested)
         {
            window_collapsed = !window_collapsed;
            ImGui::SetNextWindowCollapsed(window_collapsed, ImGuiCond_Always);
            window_collapse_requested = false;
         }
         if (ImGui::Begin("RayON - Interactive UI", nullptr,
                          ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove))
         {
            // --- Performance Monitoring ---
            if (reset_headers)
               ImGui::SetNextItemOpen(!collapse_headers);
            if (ImGui::CollapsingHeader("Performance Monitoring"))
            {
               ImGui::Text("SPP: %d", spp);
               if (sps >= 1e9f)
                  ImGui::Text("Throughput: %.2f GS/s", sps * 1e-9f);
               else if (sps >= 1e6f)
                  ImGui::Text("Throughput: %.2f MS/s", sps * 1e-6f);
               else if (sps >= 1e3f)
                  ImGui::Text("Throughput: %.1f kS/s", sps * 1e-3f);
               else
                  ImGui::Text("Throughput: %.0f S/s", sps);
               ImGui::Text("Time/Pass: %.3f ms", ms_per_sample);

               // Batch size quality ceiling and FPS target
               if (samples_per_batch)
               {
                  ImGui::SliderFloat("Max Samples/Batch", samples_per_batch, 4.0f, 500.0f, "%.0f");
                  ImGui::SetItemTooltip("Quality ceiling for the adaptive scheduler.\nThe batch size auto-scales down to hit the FPS target.");
               }
               if (target_fps_ptr)
               {
                  ImGui::SliderInt("Target FPS", target_fps_ptr, 5, 240);
                  ImGui::SetItemTooltip("Minimum frame rate to maintain.\nThe batch size adapts every frame to stay at or above this.");
               }
               ImGui::SameLine();
               if (ImGui::SmallButton("-##graph_dt"))
               {
                  perf_sample_interval_ms = (perf_sample_interval_ms > 20) ? perf_sample_interval_ms - 20 : 20;
                  last_perf_sample_time_ms = 0;
               }
               ImGui::SameLine();
               if (ImGui::SmallButton("+##graph_dt"))
               {
                  perf_sample_interval_ms = (perf_sample_interval_ms < 1000) ? perf_sample_interval_ms + 20 : 1000;
                  last_perf_sample_time_ms = 0;
               }

               if (sps > 0.0f)
               {
                  const Uint32 now_ms = SDL_GetTicks();
                  if (last_perf_sample_time_ms == 0 || now_ms - last_perf_sample_time_ms >= perf_sample_interval_ms)
                  {
                     const size_t max_points = std::max<size_t>(10, perf_time_window_ms / perf_sample_interval_ms);

                     // Ensure buffers are sized to max_points (filled with 0)
                     if (sps_history.size() != max_points)
                     {
                        sps_history.assign(max_points, 0.0f);
                        ms_history.assign(max_points, 0.0f);
                        perf_write_index = 0;
                     }

                     sps_history[perf_write_index] = sps;
                     ms_history[perf_write_index] = ms_per_sample;
                     perf_write_index = (perf_write_index + 1) % static_cast<int>(max_points);
                     last_perf_sample_time_ms = now_ms;
                  }
               }

               if (!sps_history.empty())
               {
                  const int n = static_cast<int>(sps_history.size());

                  float max_sps = 0.0f;
                  for (float f : sps_history)
                     max_sps = std::max(max_sps, f);

                  ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
                  ImGui::PlotLines("Live SPS", sps_history.data(), n, perf_write_index, nullptr,
                                   0.0f, max_sps * 1.1f, ImVec2(ImGui::CalcItemWidth(), 50));
                  ImGui::PopStyleColor();

                  float max_ms = 0.0f;
                  for (float f : ms_history)
                     max_ms = std::max(max_ms, f);

                  ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(1.0f, 0.7f, 0.0f, 1.0f));
                  ImGui::PlotLines("Time/Sample", ms_history.data(), n, perf_write_index, nullptr,
                                   0.0f, max_ms * 1.1f, ImVec2(ImGui::CalcItemWidth(), 50));
                  ImGui::PopStyleColor();
               }

            }

            // --- Adaptive Sampling ---
            if (adaptive_sampling)
            {
               if (reset_headers)
                  ImGui::SetNextItemOpen(!collapse_headers);
               if (ImGui::CollapsingHeader("Adaptive Sampling"))
               {
                  if (auto_accumulate)
                  {
                     ImGui::Checkbox("Accumulate", auto_accumulate);
                     ImGui::SameLine();
                  }

                  ImGui::Checkbox("Adaptive Sampling##adaptive_toggle", adaptive_sampling);

                  if (*adaptive_sampling)
                  {
                     if (adaptive_threshold)
                     {
                        // Display threshold as power of 10 for readability
                        // Slider works in log10 space: -4.0 = 10^-4 = 0.0001, -1.3 = 10^-1.3 = 0.05
                        float log_val = log10f(*adaptive_threshold);
                        char label[32];
                        snprintf(label, sizeof(label), "10^%.1f", log_val);
                        if (ImGui::SliderFloat("Threshold", &log_val, -6.0f, -1.0f, label))
                        {
                           *adaptive_threshold = powf(10.0f, log_val);
                        }
                     }
                     if (show_heatmap)
                        ImGui::Checkbox("Show Sample Heatmap", show_heatmap);

                     ImGui::Text("Converged: %.1f%%", convergence_pct);

                     // Visual progress bar for convergence
                     ImGui::ProgressBar(convergence_pct / 100.0f, ImVec2(-1, 0), "");
                  }
               }
            }

            // --- Camera Settings ---
            if (reset_headers)
               ImGui::SetNextItemOpen(!collapse_headers);
            if (ImGui::CollapsingHeader("Camera Settings", ImGuiTreeNodeFlags_DefaultOpen))
            {
               if (auto_orbit)
               {
                  ImGui::Checkbox("Auto orbit", auto_orbit);
                  if (dof_enabled && aperture && focus_dist)
                  {
                     ImGui::SameLine();
                     ImGui::Checkbox("Depth of field", dof_enabled);
                  }
               }
               else if (dof_enabled && aperture && focus_dist)
               {
                  ImGui::Checkbox("Depth of field", dof_enabled);
               }

               if (dof_enabled && aperture && focus_dist)
               {
                  ImGui::SeparatorText("Lens Controls");

                  if (!(*dof_enabled))
                     ImGui::BeginDisabled();
                  ImGui::SliderFloat("Aperture", aperture, 0.0f, 1.0f, "%.2f");
                  ImGui::SliderFloat("Focus Dist", focus_dist, 0.1f, 100.0f, "%.1f");
                  if (!(*dof_enabled))
                     ImGui::EndDisabled();
               }

               if (cam_fov)
               {
                  ImGui::SliderFloat("FOV", cam_fov, 10.0f, 140.0f, "%.1f");
               }

               if (cam_pos && cam_lookat)
               {
                  if (ImGui::CollapsingHeader("Camera Info"))
                  {
                     ImGui::Text("Position: %.2f, %.2f, %.2f", cam_pos[0], cam_pos[1], cam_pos[2]);
                     ImGui::Text("Look At:  %.2f, %.2f, %.2f", cam_lookat[0], cam_lookat[1], cam_lookat[2]);
                     ImGui::Text("FOV:      %.1f", cam_fov ? *cam_fov : 0.0f);
                  }
               }
            }

            // --- Environment & Materials ---
            if (reset_headers)
               ImGui::SetNextItemOpen(!collapse_headers);
            if (ImGui::CollapsingHeader("Environment & Materials", ImGuiTreeNodeFlags_DefaultOpen))
            {
               if (light_intensity && background_intensity && metal_fuzziness && glass_ior)
               {
                  ImGui::SliderFloat("Light Intensity", light_intensity, 0.1f, 3.0f, "%.1f");
                  ImGui::SliderFloat("Ambient Light", background_intensity, 0.0f, 5.0f, "%.2f");
                  ImGui::SliderFloat("Metal Fuzz", metal_fuzziness, 0.0f, 5.0f, "%.2f");
                  ImGui::SliderFloat("Glass IOR", glass_ior, 1.0f, 2.5f, "%.2f");
               }
               
               ImGui::Separator();

               if (show_normal_arrows && normal_arrow_count && normal_arrow_scale && normal_arrow_thickness)
               {
                  ImGui::Checkbox("Normal Arrows (CPU)", show_normal_arrows);

                  if (visualization_mode)
                  {
                     bool show_normals = (*visualization_mode == static_cast<int>(VisualizationMode::SHOW_NORMALS));
                     ImGui::SameLine();
                     if (ImGui::Checkbox("Show Normals", &show_normals))
                     {
                        *visualization_mode =
                            show_normals ? static_cast<int>(VisualizationMode::SHOW_NORMALS)
                                         : static_cast<int>(VisualizationMode::NORMAL);
                     }
                  }

                  if (*show_normal_arrows)
                  {
                     ImGui::SliderInt("Arrow Count", normal_arrow_count, 40, 2500);
                     ImGui::SliderFloat("Arrow Scale", normal_arrow_scale, 0.2f, 1.8f, "%.2f");
                     ImGui::SliderFloat("Arrow Thickness", normal_arrow_thickness, 1.0f, 2.5f, "%.1f");
                  }
               }
               else if (visualization_mode)
               {
                  bool show_normals = (*visualization_mode == static_cast<int>(VisualizationMode::SHOW_NORMALS));
                  if (ImGui::Checkbox("Show Normals", &show_normals))
                  {
                     *visualization_mode =
                         show_normals ? static_cast<int>(VisualizationMode::SHOW_NORMALS)
                                      : static_cast<int>(VisualizationMode::NORMAL);
                  }
               }
            }

            // --- Scene ---
            if (scene_index && scene_names && scene_count > 0)
            {
               if (reset_headers)
                  ImGui::SetNextItemOpen(!collapse_headers);
               if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
               {
                  ImGui::Combo("Scene##selector", scene_index, scene_names, scene_count);
                  if (triangle_count > 0)
                     ImGui::Text("Triangles: %d", triangle_count);
               }
            }

            // --- Appearance ---
            if (ImGui::CollapsingHeader("Appearance"))
            {
               int theme_idx = static_cast<int>(current_theme);
               if (ImGui::Combo("Theme", &theme_idx, themeNames, static_cast<int>(GuiTheme::COUNT)))
               {
                  current_theme = static_cast<GuiTheme>(theme_idx);
                  applyTheme(current_theme);
               }

               if (show_spps_counter)
               {
                  ImGui::Checkbox("Show SPP/s Counter (F)", show_spps_counter);
               }
               ImGui::Checkbox("Show Logo (L)", &show_logo);
            }

            // --- Help ---
            if (ImGui::CollapsingHeader("Controls & Help"))
            {
               ImGui::Text("Mouse:");
               ImGui::BulletText("LMB: Rotate");
               ImGui::BulletText("RMB: Pan");
               ImGui::BulletText("Wheel: Zoom");

               ImGui::Separator();
               ImGui::Text("Keys:");
               ImGui::BulletText("SPACE: Toggle Accumulation");
               ImGui::BulletText("A: Toggle Normal Arrows");
               ImGui::BulletText("N: Toggle Show Normals");
               ImGui::BulletText("Left/Right: Previous/Next Scene");
               ImGui::BulletText("O: Auto orbit");
               ImGui::BulletText("F / H: Toggle SPP/s Counter");
               ImGui::BulletText("L: Toggle Logo");
               ImGui::BulletText("Enter: Collapse/Expand Window");
               ImGui::BulletText("H: Hide/Show UI");
               ImGui::BulletText("C: Collapse/Expand All Sections");
               ImGui::BulletText("R: Reset Defaults");
               ImGui::BulletText("ESC: Exit");
            }

            if (reset_headers)
               reset_headers = false;
         }
         ImGui::End();
      }

      const bool draw_spps_counter = (show_spps_counter == nullptr) ? true : *show_spps_counter;
      updateLogoLayout();
      if (draw_spps_counter)
      {
         drawSPPSOverlay(sps, fps);
      }

      // Finalize ImGui frame
      ImGui::Render();
      ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
   }

   void drawLogo()
   {
      if (show_logo && logo_texture)
      {
         SDL_RenderCopy(renderer, logo_texture, nullptr, &logo_rect);
      }
   }

   void present() { SDL_RenderPresent(renderer); }

   // Event handling - routes to ImGui first
   static bool pollEvent(SDL_Event &event)
   {
      bool has_event = SDL_PollEvent(&event);
      if (has_event)
      {
         ImGui_ImplSDL2_ProcessEvent(&event);
      }
      return has_event;
   }

   void toggleControls() { show_controls = !show_controls; }
   bool getShowControls() const { return show_controls; }
   void toggleLogo() { show_logo = !show_logo; }
   void setLogoVisible(bool visible) { show_logo = visible; }

   void toggleWindowCollapse() { window_collapse_requested = true; }

   void toggleHeaderCollapse()
   {
      collapse_headers = !collapse_headers;
      reset_headers = true;
   }

 private:
   int image_width;
   int image_height;
   bool show_controls;
   bool collapse_headers;
   bool reset_headers;
   bool window_collapse_requested;
   bool window_collapsed;
   bool show_logo = true;
   GuiTheme current_theme;
   bool initialized;

   SDL_Window *window;
   SDL_Renderer *renderer;
   SDL_Texture *texture;
   SDL_Texture *logo_texture;
   SDL_Rect logo_rect;
   std::vector<float> sps_history;
   std::vector<float> ms_history;
   int perf_write_index;
   Uint32 last_perf_sample_time_ms;
   Uint32 perf_sample_interval_ms;
   Uint32 perf_time_window_ms;

   // Cached SPP/s + FPS overlay metrics — computed once on first draw, never change.
   bool   spps_metrics_cached = false;
   float  spps_cell_w   = 0.0f;
   float  spps_prefix_w = 0.0f;
   float  fps_prefix_w  = 0.0f;
   float  spps_gap_w    = 0.0f;
   float  spps_unit_w   = 0.0f;
   float  spps_font_h   = 0.0f;
   float  spps_box_w    = 0.0f;
   float  spps_box_h    = 0.0f;

   static void cleanupSDL() { SDL_Quit(); }

   void applyTheme(GuiTheme theme)
   {
      ImGuiStyle &style = ImGui::GetStyle();

      switch (theme)
      {
      case GuiTheme::LIGHT:
         ImGui::StyleColorsLight();
         style.Colors[ImGuiCol_WindowBg].w = 0.60f;
         break;

      case GuiTheme::CLASSIC:
         ImGui::StyleColorsClassic();
         style.Colors[ImGuiCol_WindowBg].w = 0.45f;
         break;

      case GuiTheme::NORD:
      {
         ImGui::StyleColorsDark();
         ImVec4 *c = style.Colors;
         // Nord palette: polar night + snow storm + frost
         c[ImGuiCol_WindowBg] = ImVec4(0.18f, 0.20f, 0.25f, 0.45f);
         c[ImGuiCol_Header] = ImVec4(0.26f, 0.30f, 0.37f, 0.80f);
         c[ImGuiCol_HeaderHovered] = ImVec4(0.33f, 0.37f, 0.44f, 0.80f);
         c[ImGuiCol_HeaderActive] = ImVec4(0.37f, 0.42f, 0.50f, 0.80f);
         c[ImGuiCol_FrameBg] = ImVec4(0.22f, 0.25f, 0.31f, 0.70f);
         c[ImGuiCol_FrameBgHovered] = ImVec4(0.26f, 0.30f, 0.37f, 0.70f);
         c[ImGuiCol_FrameBgActive] = ImVec4(0.33f, 0.37f, 0.44f, 0.70f);
         c[ImGuiCol_SliderGrab] = ImVec4(0.53f, 0.75f, 0.82f, 1.00f);
         c[ImGuiCol_SliderGrabActive] = ImVec4(0.56f, 0.74f, 0.73f, 1.00f);
         c[ImGuiCol_CheckMark] = ImVec4(0.53f, 0.75f, 0.82f, 1.00f);
         c[ImGuiCol_Button] = ImVec4(0.26f, 0.30f, 0.37f, 0.80f);
         c[ImGuiCol_ButtonHovered] = ImVec4(0.33f, 0.37f, 0.44f, 0.80f);
         c[ImGuiCol_ButtonActive] = ImVec4(0.37f, 0.42f, 0.50f, 0.80f);
         c[ImGuiCol_TitleBg] = ImVec4(0.18f, 0.20f, 0.25f, 1.00f);
         c[ImGuiCol_TitleBgActive] = ImVec4(0.22f, 0.25f, 0.31f, 1.00f);
         c[ImGuiCol_Text] = ImVec4(0.85f, 0.87f, 0.91f, 1.00f);
         break;
      }

      case GuiTheme::DRACULA:
      {
         ImGui::StyleColorsDark();
         ImVec4 *c = style.Colors;
         c[ImGuiCol_WindowBg] = ImVec4(0.16f, 0.16f, 0.21f, 0.45f);
         c[ImGuiCol_Header] = ImVec4(0.27f, 0.22f, 0.39f, 0.80f);
         c[ImGuiCol_HeaderHovered] = ImVec4(0.39f, 0.30f, 0.55f, 0.80f);
         c[ImGuiCol_HeaderActive] = ImVec4(0.44f, 0.35f, 0.60f, 0.80f);
         c[ImGuiCol_FrameBg] = ImVec4(0.21f, 0.20f, 0.28f, 0.70f);
         c[ImGuiCol_FrameBgHovered] = ImVec4(0.27f, 0.22f, 0.39f, 0.70f);
         c[ImGuiCol_FrameBgActive] = ImVec4(0.39f, 0.30f, 0.55f, 0.70f);
         c[ImGuiCol_SliderGrab] = ImVec4(0.74f, 0.58f, 0.98f, 1.00f);
         c[ImGuiCol_SliderGrabActive] = ImVec4(1.00f, 0.47f, 0.66f, 1.00f);
         c[ImGuiCol_CheckMark] = ImVec4(0.74f, 0.58f, 0.98f, 1.00f);
         c[ImGuiCol_Button] = ImVec4(0.27f, 0.22f, 0.39f, 0.80f);
         c[ImGuiCol_ButtonHovered] = ImVec4(0.39f, 0.30f, 0.55f, 0.80f);
         c[ImGuiCol_ButtonActive] = ImVec4(0.44f, 0.35f, 0.60f, 0.80f);
         c[ImGuiCol_TitleBg] = ImVec4(0.16f, 0.16f, 0.21f, 1.00f);
         c[ImGuiCol_TitleBgActive] = ImVec4(0.21f, 0.20f, 0.28f, 1.00f);
         c[ImGuiCol_Text] = ImVec4(0.97f, 0.97f, 0.95f, 1.00f);
         break;
      }

      case GuiTheme::GRUVBOX:
      {
         ImGui::StyleColorsDark();
         ImVec4 *c = style.Colors;
         c[ImGuiCol_WindowBg] = ImVec4(0.16f, 0.15f, 0.13f, 0.45f);
         c[ImGuiCol_Header] = ImVec4(0.31f, 0.24f, 0.16f, 0.80f);
         c[ImGuiCol_HeaderHovered] = ImVec4(0.40f, 0.31f, 0.20f, 0.80f);
         c[ImGuiCol_HeaderActive] = ImVec4(0.50f, 0.38f, 0.24f, 0.80f);
         c[ImGuiCol_FrameBg] = ImVec4(0.20f, 0.19f, 0.17f, 0.70f);
         c[ImGuiCol_FrameBgHovered] = ImVec4(0.31f, 0.24f, 0.16f, 0.70f);
         c[ImGuiCol_FrameBgActive] = ImVec4(0.40f, 0.31f, 0.20f, 0.70f);
         c[ImGuiCol_SliderGrab] = ImVec4(0.98f, 0.72f, 0.20f, 1.00f);
         c[ImGuiCol_SliderGrabActive] = ImVec4(0.84f, 0.60f, 0.13f, 1.00f);
         c[ImGuiCol_CheckMark] = ImVec4(0.72f, 0.84f, 0.15f, 1.00f);
         c[ImGuiCol_Button] = ImVec4(0.31f, 0.24f, 0.16f, 0.80f);
         c[ImGuiCol_ButtonHovered] = ImVec4(0.40f, 0.31f, 0.20f, 0.80f);
         c[ImGuiCol_ButtonActive] = ImVec4(0.50f, 0.38f, 0.24f, 0.80f);
         c[ImGuiCol_TitleBg] = ImVec4(0.16f, 0.15f, 0.13f, 1.00f);
         c[ImGuiCol_TitleBgActive] = ImVec4(0.20f, 0.19f, 0.17f, 1.00f);
         c[ImGuiCol_Text] = ImVec4(0.92f, 0.86f, 0.70f, 1.00f);
         break;
      }

      case GuiTheme::CATPPUCCIN:
      {
         ImGui::StyleColorsDark();
         ImVec4 *c = style.Colors;
         // Catppuccin Mocha palette
         c[ImGuiCol_WindowBg] = ImVec4(0.12f, 0.12f, 0.18f, 0.45f);
         c[ImGuiCol_Header] = ImVec4(0.18f, 0.19f, 0.29f, 0.80f);
         c[ImGuiCol_HeaderHovered] = ImVec4(0.24f, 0.25f, 0.38f, 0.80f);
         c[ImGuiCol_HeaderActive] = ImVec4(0.29f, 0.30f, 0.45f, 0.80f);
         c[ImGuiCol_FrameBg] = ImVec4(0.18f, 0.19f, 0.29f, 0.70f);
         c[ImGuiCol_FrameBgHovered] = ImVec4(0.24f, 0.25f, 0.38f, 0.70f);
         c[ImGuiCol_FrameBgActive] = ImVec4(0.29f, 0.30f, 0.45f, 0.70f);
         c[ImGuiCol_SliderGrab] = ImVec4(0.52f, 0.65f, 0.99f, 1.00f);
         c[ImGuiCol_SliderGrabActive] = ImVec4(0.95f, 0.55f, 0.66f, 1.00f);
         c[ImGuiCol_CheckMark] = ImVec4(0.65f, 0.89f, 0.63f, 1.00f);
         c[ImGuiCol_Button] = ImVec4(0.18f, 0.19f, 0.29f, 0.80f);
         c[ImGuiCol_ButtonHovered] = ImVec4(0.24f, 0.25f, 0.38f, 0.80f);
         c[ImGuiCol_ButtonActive] = ImVec4(0.29f, 0.30f, 0.45f, 0.80f);
         c[ImGuiCol_TitleBg] = ImVec4(0.12f, 0.12f, 0.18f, 1.00f);
         c[ImGuiCol_TitleBgActive] = ImVec4(0.18f, 0.19f, 0.29f, 1.00f);
         c[ImGuiCol_Text] = ImVec4(0.80f, 0.84f, 0.96f, 1.00f);
         break;
      }

      case GuiTheme::DARK:
      default:
         ImGui::StyleColorsDark();
         style.Colors[ImGuiCol_WindowBg].w = 0.35f;
         break;
      }

      // Common style tweaks
      style.WindowRounding = 4.0f;
      style.FrameRounding = 2.0f;
      style.GrabRounding = 2.0f;
   }

   void loadLogo()
   {
      const auto relative_width = 0.3F;
      int logo_img_width, logo_img_height, logo_img_channels;
      unsigned char *logo_data = stbi_load("../resources/ISC Logo inline white v3 - 1500px.png", &logo_img_width,
                                           &logo_img_height, &logo_img_channels, 4);

      if (logo_data != nullptr)
      {
         int target_logo_width = static_cast<int>(image_width * relative_width);
         int target_logo_height = (logo_img_height * target_logo_width) / logo_img_width;

         auto *resized_logo = new unsigned char[target_logo_width * target_logo_height * 4];
         stbir_resize_uint8_srgb(logo_data, logo_img_width, logo_img_height, 0, resized_logo, target_logo_width,
                                 target_logo_height, 0, STBIR_RGBA);

         SDL_Surface *logo_surface =
             SDL_CreateRGBSurfaceFrom(resized_logo, target_logo_width, target_logo_height, 32, target_logo_width * 4,
                                      0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);

         if (logo_surface != nullptr)
         {
            logo_texture = SDL_CreateTextureFromSurface(renderer, logo_surface);
            SDL_SetTextureBlendMode(logo_texture, SDL_BLENDMODE_BLEND);

            logo_rect.w = target_logo_width;
            logo_rect.h = target_logo_height;
            logo_rect.x = image_width - target_logo_width - 10;
            logo_rect.y = image_height - target_logo_height - 10;

            SDL_FreeSurface(logo_surface);
         }

         delete[] resized_logo;
         stbi_image_free(logo_data);
      }
   }

   void updateLogoLayout()
   {
      if (!logo_texture)
         return;

      const int margin = 10;
      logo_rect.x = image_width - logo_rect.w - margin;
      logo_rect.y = image_height - logo_rect.h - margin;
   }

   // Draw a stable SPP/s counter in the bottom-left using the foreground draw list.
   // GetForegroundDrawList() is unclipped and independent of any ImGui window, so
   // positioning is exact screen-space with no window padding or clip-rect surprises.
   void drawSPPSOverlay(float sps, float fps)
   {
      ImFont      *font = ImGui::GetFont();
      const float  fs   = ImGui::GetFontSize();

      // Compute metrics only once — font never changes at runtime.
      if (!spps_metrics_cached)
      {
         auto gw = [&](const char *s) { return font->CalcTextSizeA(fs, FLT_MAX, 0.0f, s).x; };
         for (const char *c = "0123456789."; *c; ++c)
         {
            char g[2] = {*c, '\0'};
            spps_cell_w = std::max(spps_cell_w, gw(g));
         }
         spps_prefix_w = gw("SPP/s: ");
         fps_prefix_w  = gw("FPS:  ");
         spps_gap_w    = gw(" ");
         spps_unit_w   = gw("M");  // M is widest unit
         spps_font_h   = font->CalcTextSizeA(fs, FLT_MAX, 0.0f, "A").y;

         constexpr int num_digit_chars = 4; // "%4.0f" always produces 4 chars
         const float   sps_content_w = spps_prefix_w + spps_cell_w * num_digit_chars
                                        + spps_gap_w + spps_unit_w;
         const float   fps_content_w = fps_prefix_w + spps_cell_w * num_digit_chars;
         spps_box_w = std::max(sps_content_w, fps_content_w) + 8.0f * 2.0f;
         spps_box_h = 2.0f * spps_font_h + 3.0f + 5.0f * 2.0f; // two rows + 3px gap
         spps_metrics_cached = true;
      }

      // --- scale value and choose unit; overflow if scaled value exceeds 4 digits ---
      float scaled = sps;
      const char *unit = " ";
      if      (sps >= 1e6f) { scaled = sps * 1e-6f; unit = "M"; }
      else if (sps >= 1e3f) { scaled = sps * 1e-3f; unit = "k"; }

      // Guard against values that won't fit in 4 integer digits after scaling
      const bool overflow = (scaled >= 10000.0f);

      // 4-char integer field: "NNNN" (no decimal point)
      char number_str[8];
      if (!overflow)
         std::snprintf(number_str, sizeof(number_str), "%4.0f", static_cast<double>(scaled));

      const float pad_x  = 8.0f;
      const float pad_y  = 5.0f;
      const float margin = 10.0f;
      const float box_w  = spps_box_w;
      const float box_h  = spps_box_h;
      const float box_x     = margin;
      const float box_y     = static_cast<float>(image_height) - box_h - margin;

      // --- hover tooltip: show description when mouse is over the box ---
      const ImVec2 mouse = ImGui::GetIO().MousePos;
      if (mouse.x >= box_x && mouse.x <= box_x + box_w &&
          mouse.y >= box_y && mouse.y <= box_y + box_h)
      {
         ImGui::SetTooltip("SPP/s: Samples Per Pixel per second — rendering throughput.\n"
                           "FPS: Display frame rate.\n"
                           "Toggle with F.");
      }

      // --- draw background rect then text directly onto the foreground layer ---
      ImDrawList *draw = ImGui::GetForegroundDrawList();

      draw->AddRectFilled(ImVec2(box_x, box_y),
                          ImVec2(box_x + box_w, box_y + box_h),
                          IM_COL32(0, 0, 0, 89), 4.0f);

      const ImU32  text_col = IM_COL32(220, 220, 220, 220);
      const float  text_y   = box_y + pad_y;
      float        cx       = box_x + pad_x;

      // ── SPP/s row ──
      // prefix as normal text
      draw->AddText(font, fs, ImVec2(cx, text_y), text_col, "SPP/s: ");
      cx += spps_prefix_w;

      // digits in fixed-width cells so varying widths never shift the unit
      if (overflow)
      {
         // Right-align ">= 10G" within the number+gap+unit region
         constexpr const char *ovf_str = ">= 10G";
         const float ovf_w   = font->CalcTextSizeA(fs, FLT_MAX, 0.0f, ovf_str).x;
         const float region_w = spps_cell_w * 4.0f + spps_gap_w + spps_unit_w;
         const float ovf_x   = cx + region_w - ovf_w;
         draw->AddText(font, fs, ImVec2(ovf_x, text_y), text_col, ovf_str);
      }
      else
      {
         const int num_chars = 4; // always render exactly 4 cells
         for (int i = 0; i < num_chars; ++i)
         {
            if (number_str[i] != ' ')
            {
               char g[2] = {number_str[i], '\0'};
               const float gw = font->CalcTextSizeA(fs, FLT_MAX, 0.0f, g).x;
               draw->AddText(font, fs, ImVec2(cx + (spps_cell_w - gw) * 0.5f, text_y), text_col, g);
            }
            cx += spps_cell_w;
         }

         // fixed-width gap then unit
         cx += spps_gap_w;
         draw->AddText(font, fs, ImVec2(cx, text_y), text_col, unit);
      }

      // ── FPS row ──
      const float  text_y2     = text_y + spps_font_h + 3.0f;
      const bool   fps_overflow = (fps >= 10000.0f);
      char fps_str[8];
      if (!fps_overflow)
         std::snprintf(fps_str, sizeof(fps_str), "%4.0f", static_cast<double>(fps));

      float cx2 = box_x + pad_x;
      draw->AddText(font, fs, ImVec2(cx2, text_y2), text_col, "FPS:  ");
      cx2 += fps_prefix_w;

      if (fps_overflow)
      {
         constexpr const char *ovf_str = ">9999";
         const float ovf_w    = font->CalcTextSizeA(fs, FLT_MAX, 0.0f, ovf_str).x;
         const float region_w = spps_cell_w * 4.0f;
         draw->AddText(font, fs, ImVec2(cx2 + region_w - ovf_w, text_y2), text_col, ovf_str);
      }
      else
      {
         for (int i = 0; i < 4; ++i)
         {
            if (fps_str[i] != ' ')
            {
               char g[2] = {fps_str[i], '\0'};
               const float gw = font->CalcTextSizeA(fs, FLT_MAX, 0.0f, g).x;
               draw->AddText(font, fs, ImVec2(cx2 + (spps_cell_w - gw) * 0.5f, text_y2), text_col, g);
            }
            cx2 += spps_cell_w;
         }
      }
   }
};
#endif // SDL2_FOUND
