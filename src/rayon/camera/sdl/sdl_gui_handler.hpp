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

#include <algorithm>
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

/// Quality reductions applied while the camera is moving (interactive OptiX renderer).
/// Accumulation restarts on every camera change, so none of these affect the converged image.
struct MotionQualitySettings
{
   float motion_fraction  = 0.75f; ///< Fraction of the width and height rendered while moving (1, 3/4, 1/2, 1/4)
   // Frame-rate regulation while moving: samples per frame first, then the render fraction, are
   // lowered to hold target_fps and raised back when there is headroom (see the OptiX host).
   bool  hold_target_fps  = true;  ///< Regulate samples and resolution to hold target_fps while moving
   float target_fps       = 60.0f; ///< Frame rate to hold while moving
   float min_fraction     = 0.25f;  ///< Lowest render fraction the regulator may use (1/4: 1/16 of the pixels)
   int   min_spp          = 4;     ///< Lowest samples per pixel per motion frame the regulator may use
   float current_fraction = 0.75f; ///< Readout: fraction used by the last motion frame
   int   current_spp      = 0;     ///< Readout: samples per pixel of the last motion frame
   bool  refine_stages    = false; ///< After the motion, climb through 1/2 and 3/4 resolution before full (off by default)
   float stage_seconds    = 0.1f;  ///< Duration of each intermediate resolution stage
   float fade_seconds     = 0.1f;  ///< Duration of the fade from one stage's image into the next
   float cooldown_seconds = 0.05f; ///< Time after the last camera event before the camera counts as still
   // OptiX AI denoiser: applied to every motion frame, then faded out over the first still frames
   bool  denoise               = false; ///< Denoise motion frames (OptiX backend only), off by default
   int   denoise_until_samples = 64;    ///< Once still, keep denoising with a weight fading to 0 at this many samples
   float denoise_ms            = 0.0f;  ///< Readout: denoiser time of the last denoised frame (ms)
   bool single_sample    = false; ///< One sample per frame instead of the batch size
   bool fast_sampler     = false; ///< PCG instead of Sobol (cheaper per sample)
   bool limit_depth      = false; ///< Cap the bounce depth at motion_max_depth (slightly darkens indirect light)
   int  motion_max_depth = 4;
};

/// Live controls of the scene's volume (cloud / smoke) material. The host fills it from the
/// material, the GUI edits it and sets `changed`, the host writes it back to the material.
struct VolumeMaterialSettings
{
   float albedo[3]      = {0.99f, 0.99f, 0.99f}; ///< Single-scattering albedo
   float density        = 2.0f;                  ///< Extinction per scene unit
   float g              = 0.0f;                  ///< Henyey-Greenstein asymmetry
   float noise_scale    = 2.0f;
   float noise_strength = 0.3f;
   float edge           = 0.1f;                  ///< Edge fall-off depth (scene units)
   float lumps          = 0.3f;                  ///< Surface displacement amplitude (scene units)
   float billow         = 0.7f;                  ///< Worley share of the shape noise
   bool  changed        = false;                 ///< Set by the GUI, cleared by the host once applied
};

class SDLGuiHandler
{
 public:
   SDLGuiHandler(int image_width, int image_height, GuiTheme initial_theme = GuiTheme::NORD, std::string backend_name = "")
       : image_width(image_width), image_height(image_height), backend_name(std::move(backend_name)), show_controls(true), collapse_headers(false),
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
          "RayON v" + std::string(constants::version) + (backend_name.empty() ? "" : " [" + backend_name + "]");
      // Resizable: the render keeps its pixel count, the image is scaled (letterboxed) to the window
      window = SDL_CreateWindow(window_title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, image_width,
                                image_height, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
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
   /// Despeckle controls for motion frames (toggle + ratio). Either pointer may be null.
   void drawMotionDespeckle(bool *enabled, float *ratio)
   {
      if (!enabled)
         return;
      ImGui::Checkbox("Despeckle motion frames##mq_despeckle", enabled);
      ImGui::SetItemTooltip("While moving, a pixel far brighter than its 8 neighbours is scaled\n"
                            "down at display time. Isolated bright dots disappear; bright regions\n"
                            "and the accumulation itself are untouched.");
      if (*enabled && ratio)
      {
         // Own line: next to the checkbox label the slider would run past the panel edge
         ImGui::PushItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.67f, 120.0f, 240.0f));
         ImGui::SliderFloat("Ratio##mq_despeckle_ratio", ratio, 1.0f, 8.0f, "x%.1f");
         ImGui::PopItemWidth();
         ImGui::SetItemTooltip("A pixel is scaled down when it exceeds this multiple of the mean of\n"
                               "its neighbours. Lower = more aggressive.");
      }
   }

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
                      int *target_fps_ptr = nullptr,
                      int *golf_dimple_count = nullptr, float *golf_dimple_radius = nullptr,
                      float *golf_dimple_depth = nullptr,
                      int *hdr_env_index = nullptr, const char *const *hdr_env_names = nullptr,
                      int hdr_env_count = 0,
                      bool *mis_enabled = nullptr, bool *motion_gate_mis = nullptr,
                      int *nee_stride = nullptr, bool *use_sobol = nullptr,
                      MotionQualitySettings *motion_quality = nullptr,
                      bool *motion_despeckle = nullptr, float *motion_despeckle_ratio = nullptr,
                      VolumeMaterialSettings *volume = nullptr,
                      bool *save_camera_request = nullptr, const char *scene_yaml_path = nullptr,
                      const char *camera_save_status = nullptr, float *light_falloff = nullptr)
   {
      SDL_UpdateTexture(texture, nullptr, image.data(), image_width * image_channels);
      SDL_RenderClear(renderer);
      const SDL_Rect image_rect = imageDestRect();
      SDL_RenderCopy(renderer, texture, nullptr, &image_rect);

      // Begin ImGui frame
      ImGui_ImplSDLRenderer2_NewFrame();
      ImGui_ImplSDL2_NewFrame();
      ImGui::NewFrame();

      if (show_controls)
      {
         ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
         // Keep a stable panel width so opening sections doesn't cause horizontal jitter.
         const float panel_width = 300.0f;
         // Cap height so the panel never overlaps the SPP/s counter at page bottom.
         const float max_panel_height = ImGui::GetIO().DisplaySize.y - 80.0f;
         ImGui::SetNextWindowSizeConstraints(ImVec2(panel_width, 0.0f), ImVec2(panel_width, max_panel_height));
         if (window_collapse_requested)
         {
            window_collapsed = !window_collapsed;
            ImGui::SetNextWindowCollapsed(window_collapsed, ImGuiCond_Always);
            window_collapse_requested = false;
         }
         if (ImGui::Begin("RayON - Interactive UI", nullptr,
                          ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove))
         {
            auto pushSliderWidth = []()
            {
               const float width_from_layout = ImGui::GetContentRegionAvail().x * 0.67f;
               const float clamped_width = std::clamp(width_from_layout, 120.0f, 240.0f);
               ImGui::PushItemWidth(clamped_width);
            };

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

               // Fixed batch size and motion batch cap
               if (samples_per_batch)
               {
                  pushSliderWidth();
                  ImGui::SliderFloat("Samples/Batch", samples_per_batch, 2.0f, 120.0f, "%.0f");
                  ImGui::PopItemWidth();
                  ImGui::SetItemTooltip("Fixed samples per batch while the camera is still.");
               }
               if (sps > 0.0f || ms_per_sample > 0.0f)
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

                  auto maxHistoryValue = [](const std::vector<float> &history)
                  {
                     float max_value = 0.0f;
                     for (float value : history)
                        max_value = std::max(max_value, value);
                     return max_value;
                  };

                  const float max_sps = std::max(maxHistoryValue(sps_history), 1.0f);
                  const float max_ms = std::max(maxHistoryValue(ms_history), 1.0f);
                  const int latest_index = (perf_write_index + n - 1) % n;

                  auto formatSpsValue = [](float value)
                  {
                     char buffer[32];
                     if (value >= 1e9f)
                        std::snprintf(buffer, sizeof(buffer), "%.2f GS/s", static_cast<double>(value * 1e-9f));
                     else if (value >= 1e6f)
                        std::snprintf(buffer, sizeof(buffer), "%.2f MS/s", static_cast<double>(value * 1e-6f));
                     else if (value >= 1e3f)
                        std::snprintf(buffer, sizeof(buffer), "%.1f kS/s", static_cast<double>(value * 1e-3f));
                     else
                        std::snprintf(buffer, sizeof(buffer), "%.0f S/s", static_cast<double>(value));
                     return std::string(buffer);
                  };

                  ImGui::TextUnformatted("Performance History (normalized)");
                  ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "SPS");
                  ImGui::SameLine();
                  ImGui::TextUnformatted("/");
                  ImGui::SameLine();
                  ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.0f, 1.0f), "Time/Pass");

                  const ImVec2 graph_size(std::max(ImGui::GetContentRegionAvail().x, 1.0f), 80.0f);
                  const ImVec2 graph_pos = ImGui::GetCursorScreenPos();
                  ImGui::InvisibleButton("##perf_history_graph", graph_size);

                  ImDrawList *draw = ImGui::GetWindowDrawList();
                  const ImU32 bg_col = IM_COL32(18, 18, 18, 160);
                  const ImU32 border_col = IM_COL32(110, 110, 110, 180);
                  const ImU32 grid_col = IM_COL32(90, 90, 90, 70);
                  const ImVec2 graph_min = graph_pos;
                  const ImVec2 graph_max(graph_pos.x + graph_size.x, graph_pos.y + graph_size.y);

                  draw->AddRectFilled(graph_min, graph_max, bg_col, 4.0f);
                  draw->AddRect(graph_min, graph_max, border_col, 4.0f);

                  for (int grid = 1; grid < 4; ++grid)
                  {
                     const float y = graph_min.y + (graph_size.y * static_cast<float>(grid) / 4.0f);
                     draw->AddLine(ImVec2(graph_min.x, y), ImVec2(graph_max.x, y), grid_col);
                  }

                  auto drawHistoryLine = [&](const std::vector<float> &history, float max_value, ImU32 color)
                  {
                     if (history.size() < 2)
                        return;

                     for (int i = 1; i < n; ++i)
                     {
                        const int prev_index = (perf_write_index + i - 1) % n;
                        const int curr_index = (perf_write_index + i) % n;
                        const float prev_value = history[prev_index] / max_value;
                        const float curr_value = history[curr_index] / max_value;

                        const float x0 = graph_min.x + (graph_size.x * static_cast<float>(i - 1) / static_cast<float>(n - 1));
                        const float x1 = graph_min.x + (graph_size.x * static_cast<float>(i) / static_cast<float>(n - 1));
                        const float y0 = graph_max.y - prev_value * graph_size.y;
                        const float y1 = graph_max.y - curr_value * graph_size.y;
                        draw->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), color, 1.0f);
                     }
                  };

                  drawHistoryLine(sps_history, max_sps, IM_COL32(0, 255, 0, 255));
                  drawHistoryLine(ms_history, max_ms, IM_COL32(255, 179, 0, 255));

                  if (ImGui::IsItemHovered())
                  {
                     const ImVec2 mouse = ImGui::GetIO().MousePos;
                     const float graph_width = std::max(graph_size.x, 1.0f);
                     const float t = std::clamp((mouse.x - graph_min.x) / graph_width, 0.0f, 1.0f);
                     const int sample_offset = (n <= 1) ? 0 : static_cast<int>(std::lround(t * static_cast<float>(n - 1)));
                     const int history_index = (perf_write_index + sample_offset) % n;
                     const int age = n - 1 - sample_offset;
                     const Uint32 age_ms = static_cast<Uint32>(age) * perf_sample_interval_ms;

                     ImGui::BeginTooltip();
                     ImGui::Text("History sample: -%u ms", age_ms);
                     ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "SPS: %s", formatSpsValue(sps_history[history_index]).c_str());
                     ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.0f, 1.0f), "Time/Pass: %.3f ms", ms_history[history_index]);
                     ImGui::EndTooltip();
                  }

                  ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "SPS  %s", formatSpsValue(sps_history[latest_index]).c_str());
                  ImGui::SameLine();
                  ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.0f, 1.0f), "Time  %.3f ms", ms_history[latest_index]);
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
                        // Threshold = standard error of the pixel mean, in display space (fraction
                        // of full scale), at which a pixel stops sampling. Slider works in log10
                        // space: -2.0 = 10^-2 = 0.01 ≈ 2.5 levels of 255.
                        float log_val = log10f(*adaptive_threshold);
                        char label[32];
                        snprintf(label, sizeof(label), "10^%.1f (%.1f/255)", log_val, 255.0f * *adaptive_threshold);
                        pushSliderWidth();
                        if (ImGui::SliderFloat("Threshold", &log_val, -3.0f, -1.0f, label))
                        {
                           *adaptive_threshold = powf(10.0f, log_val);
                        }
                        ImGui::PopItemWidth();
                        ImGui::SetItemTooltip("Standard error of a pixel's displayed value (in levels of 255)\n"
                                              "below which it stops taking samples. Lower = cleaner, slower.");
                     }
                     if (show_heatmap)
                        ImGui::Checkbox("Show Sample Heatmap", show_heatmap);

                     ImGui::Text("Converged: %.1f%%", convergence_pct);

                     // Visual progress bar for convergence
                     ImGui::ProgressBar(convergence_pct / 100.0f, ImVec2(-1, 0), "");
                  }

                  if (use_sobol)
                  {
                     ImGui::Separator();
                     ImGui::Checkbox("Sobol sampler##sobol_toggle", use_sobol);
                     ImGui::SetItemTooltip("Use Sobol low-discrepancy sampler (better convergence).\n"
                                          "Uncheck for classic PCG pseudo-random sampling.");
                  }

                  // --- MIS / NEE (nested under Adaptive Sampling) ---
                  if (mis_enabled)
                  {
                     ImGui::Separator();
                     ImGui::SetNextItemOpen(true, ImGuiCond_Once);
                     if (ImGui::TreeNode("MIS / NEE"))
                     {
                        ImGui::Checkbox("Enable MIS##mis_toggle", mis_enabled);
                        ImGui::SetItemTooltip("Master toggle for Multiple-Importance Sampling.\n"
                                             "Disabling removes shadow rays and emissive MIS weight\n"
                                             "for maximum throughput (faster, noisier).");

                        bool mis_ok = *mis_enabled;


                        if (nee_stride)
                        {
                           if (!mis_ok) ImGui::BeginDisabled();
                           pushSliderWidth();
                           ImGui::SliderInt("NEE stride##nee_stride", nee_stride, 1, 8);
                           ImGui::SetItemTooltip("The first bounce always traces a shadow ray; deeper bounces\n"
                                                "do it on 1 out of every N samples (contribution \xc3\x97 N, unbiased).\n"
                                                "1 = every sample (lowest noise), 3 = default, higher = faster.");
                           ImGui::PopItemWidth();
                           if (!mis_ok) ImGui::EndDisabled();
                        }
                        ImGui::TreePop();
                     }
                  }
               }
            }

            // --- Camera Motion ---
            // Everything that only applies while the camera moves (and the steps back to full
            // quality once it stops). The accumulation restarts on every move, so none of this
            // ever affects the final image.
            if (motion_quality || motion_despeckle || motion_gate_mis)
            {
               if (reset_headers)
                  ImGui::SetNextItemOpen(!collapse_headers);
               const bool motion_open = ImGui::CollapsingHeader("Camera Motion");
               ImGui::SetItemTooltip("Quality trade-offs that only apply while the camera moves.\n"
                                     "The accumulation restarts on every move, so none of this\n"
                                     "affects the final image.");
               if (motion_open)
               {
                  if (motion_quality)
                  {
                     ImGui::SeparatorText("Resolution");
                     static const float fractions[] = {1.0f, 0.75f, 0.5f, 0.25f};
                     int res_idx = 2;
                     for (int i = 0; i < 4; i++)
                        if (std::fabs(motion_quality->motion_fraction - fractions[i]) < 0.01f)
                           res_idx = i;
                     pushSliderWidth();
                     if (ImGui::Combo(motion_quality->hold_target_fps ? "Max while moving##mq_res" : "While moving##mq_res",
                                      &res_idx, "Full\0Three quarters\0Half\0Quarter\0"))
                        motion_quality->motion_fraction = fractions[res_idx];
                     ImGui::PopItemWidth();
                     // The floor can never exceed the ceiling: keep the slider range and the value inside it
                     motion_quality->min_fraction = std::min(motion_quality->min_fraction, motion_quality->motion_fraction);
                     ImGui::SetItemTooltip(motion_quality->hold_target_fps
                                               ? "Highest fraction of the width and height the regulator may use\n"
                                                 "while moving."
                                               : "Render at this fraction of the width and height while moving\n"
                                                 "(quarter = 16x fewer paths per frame), upscaled for display.");
                     ImGui::Checkbox("Hold a target frame rate##mq_hold", &motion_quality->hold_target_fps);
                     ImGui::SetItemTooltip("While moving, lower the render fraction first, then the samples per\n"
                                           "frame, until the frame rate holds; raise them back when there is\n"
                                           "headroom. The fraction above is the ceiling, the one below the floor.");
                     if (motion_quality->hold_target_fps)
                     {
                        pushSliderWidth();
                        ImGui::SliderFloat("Target##mq_fps", &motion_quality->target_fps, 20.0f, 144.0f, "%.0f fps");
                        ImGui::SliderFloat("Min while moving##mq_minfrac", &motion_quality->min_fraction, 0.0625f,
                                           motion_quality->motion_fraction, "%.3f", ImGuiSliderFlags_AlwaysClamp);
                        ImGui::SetItemTooltip("Lowest render fraction the regulator may go down to (the\n"
                                              "\"Max while moving\" choice above is its ceiling).");
                        // The floor of the samples per motion frame: the ceiling is the batch size
                        const int spp_ceiling = samples_per_batch ? std::max(1, (int)*samples_per_batch) : 120;
                        motion_quality->min_spp = std::clamp(motion_quality->min_spp, 1, spp_ceiling);
                        ImGui::SliderInt("Min samples##mq_minspp", &motion_quality->min_spp, 1, spp_ceiling, "%d spp",
                                         ImGuiSliderFlags_AlwaysClamp);
                        ImGui::PopItemWidth();
                        ImGui::SetItemTooltip("Lowest samples per pixel per frame the regulator may go down to\n"
                                              "while moving (the Samples/Batch value is its ceiling).");
                        if (motion_quality->current_spp > 0)
                           ImGui::TextDisabled("Last motion frame: %.0f%% x %d spp", 100.0f * motion_quality->current_fraction,
                                               motion_quality->current_spp);
                     }
                     // The steps back to full quality once the camera stops, in their own sub-menu
                     // (off by default: the image goes straight to full resolution)
                     if (ImGui::TreeNode("Intermediate stages##mq_stages_menu"))
                     {
                        ImGui::Checkbox("Intermediate stages (1/2, 3/4)##mq_refine", &motion_quality->refine_stages);
                        ImGui::SetItemTooltip("When the camera stops, climb through the standard resolutions\n"
                                              "above the motion one (1/2 and 3/4) before going full, each\n"
                                              "stage fading into the previous image.");
                        pushSliderWidth();
                        ImGui::SliderFloat("Stage length##mq_stage", &motion_quality->stage_seconds, 0.1f, 1.5f, "%.2f s");
                        ImGui::SetItemTooltip("How long each intermediate resolution stage lasts.");
                        ImGui::SliderFloat("Fade##mq_fade", &motion_quality->fade_seconds, 0.1f, 2.0f, "%.2f s");
                        ImGui::SetItemTooltip("How long the previous image stays blended in after a stage change.");
                        ImGui::SliderFloat("Motion cooldown##mq_cool", &motion_quality->cooldown_seconds, 0.05f, 0.5f, "%.2f s");
                        ImGui::SetItemTooltip("Time after the last camera event before the camera counts as still.\n"
                                              "Bridges the gaps between mouse or key-repeat events.");
                        ImGui::PopItemWidth();
                        ImGui::TreePop();
                     }

                     ImGui::SeparatorText("Sampling while moving");
                     ImGui::Checkbox("One sample per frame##mq_spp", &motion_quality->single_sample);
                     ImGui::SetItemTooltip("Trace a single sample per pixel per frame instead of the\n"
                                           "batch size while moving.");
                     ImGui::Checkbox("Fast sampler (PCG)##mq_pcg", &motion_quality->fast_sampler);
                     ImGui::SetItemTooltip("Use the cheaper PCG random sampler instead of Sobol while moving.");
                     ImGui::Checkbox("Limit bounce depth##mq_depth", &motion_quality->limit_depth);
                     ImGui::SetItemTooltip("Cap the number of bounces while moving (~10%% faster).\n"
                                           "Indirect light is slightly darker until the camera stops,\n"
                                           "which shows as a small brightness step.");
                     if (motion_quality->limit_depth)
                     {
                        ImGui::SameLine();
                        pushSliderWidth();
                        ImGui::SliderInt("##mq_depth_val", &motion_quality->motion_max_depth, 1, 8);
                        ImGui::PopItemWidth();
                     }
                  }
                  if (motion_gate_mis)
                  {
                     if (!motion_quality)
                        ImGui::SeparatorText("Sampling while moving");
                     const bool mis_ok = mis_enabled ? *mis_enabled : true;
                     if (!mis_ok) ImGui::BeginDisabled();
                     ImGui::Checkbox("Disable MIS / NEE while moving##mgm", motion_gate_mis);
                     ImGui::SetItemTooltip("Skip the shadow rays of next event estimation while the camera\n"
                                           "moves, to spend the budget on throughput. MIS resumes as soon\n"
                                           "as the camera stops.");
                     if (!mis_ok) ImGui::EndDisabled();
                  }
                  if (motion_despeckle)
                  {
                     ImGui::SeparatorText("Display while moving");
                     drawMotionDespeckle(motion_despeckle, motion_despeckle_ratio);
                  }
                  if (motion_quality)
                  {
                     ImGui::Checkbox("AI denoiser (OptiX)##mq_denoise", &motion_quality->denoise);
                     ImGui::SetItemTooltip("Run the OptiX neural denoiser on each motion frame, guided by the\n"
                                           "albedo and normal of the first surface each sample sees. Display\n"
                                           "only: the accumulation stays unbiased.");
                     if (motion_quality->denoise)
                     {
                        ImGui::PushItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.67f, 120.0f, 240.0f));
                        ImGui::SliderInt("Fade out until##mq_denoise_until", &motion_quality->denoise_until_samples, 0,
                                         512, "%d spp");
                        ImGui::PopItemWidth();
                        ImGui::SetItemTooltip("Once the camera is still, keep denoising with a weight that fades\n"
                                              "to zero when the accumulation reaches this many samples, so the\n"
                                              "image does not snap from smooth to noisy. 0 = motion frames only.");
                        if (motion_quality->denoise_ms > 0.0f)
                           ImGui::Text("Denoiser: %.1f ms", motion_quality->denoise_ms);
                     }
                  }
               }
            }

            // --- Camera Settings ---
            if (reset_headers)
               ImGui::SetNextItemOpen(!collapse_headers);
            if (ImGui::CollapsingHeader("Camera Settings"))
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
                  pushSliderWidth();
                  ImGui::SliderFloat("Aperture", aperture, 0.0f, 1.0f, "%.2f");
                  ImGui::SliderFloat("Focus Dist", focus_dist, 0.1f, 100.0f, "%.1f");
                  ImGui::PopItemWidth();
                  if (!(*dof_enabled))
                     ImGui::EndDisabled();
               }

               if (cam_fov)
               {
                  pushSliderWidth();
                  ImGui::SliderFloat("FOV", cam_fov, 10.0f, 140.0f, "%.1f");
                  ImGui::PopItemWidth();
               }

               if (cam_pos && cam_lookat)
               {
                  ImGui::Separator();
                  ImGui::Text("Position: %.2f, %.2f, %.2f", cam_pos[0], cam_pos[1], cam_pos[2]);
                  ImGui::Text("Look at:  %.2f, %.2f, %.2f", cam_lookat[0], cam_lookat[1], cam_lookat[2]);
                  if (save_camera_request)
                  {
                     const bool can_save = scene_yaml_path && scene_yaml_path[0] != '\0';
                     if (!can_save)
                        ImGui::BeginDisabled();
                     if (ImGui::Button("Save camera & lights to scene YAML"))
                        *save_camera_request = true;
                     if (!can_save)
                        ImGui::EndDisabled();
                     if (can_save)
                        ImGui::SetItemTooltip("Writes the camera (position, look_at, fov) and the light settings\n"
                                              "(Lights inten., Ambient Light) into\n%s\n(only those lines are touched).",
                                              scene_yaml_path);
                     else
                        ImGui::SetItemTooltip("The current scene was not loaded from a YAML file.");
                     if (camera_save_status && camera_save_status[0] != '\0')
                        ImGui::TextWrapped("%s", camera_save_status);
                  }
               }
            }

            // --- Environment & Materials ---
            if (reset_headers)
               ImGui::SetNextItemOpen(!collapse_headers);
            if (ImGui::CollapsingHeader("Environment & Materials"))
            {
               // HDR sky selector
               if (hdr_env_index && hdr_env_names && hdr_env_count > 0)
               {
                  ImGui::Combo("Sky##hdr", hdr_env_index, hdr_env_names, hdr_env_count);
                  ImGui::SetItemTooltip("Background sky dome.\nNumpad +/- to cycle.");
               }

               if (light_intensity && background_intensity && metal_fuzziness && glass_ior)
               {
                  pushSliderWidth();
                  ImGui::SliderFloat("Lights inten.", light_intensity, 0.1f, 3.0f, "%.1f");
                  if (light_falloff)
                  {
                     ImGui::SliderFloat("Lights falloff", light_falloff, 0.5f, 4.0f, "%.2f");
                     ImGui::SetItemTooltip("Distance exponent of the lights' reach. 2 = physical inverse\n"
                                           "square. Lower: the light carries farther (open room). Higher: it\n"
                                           "dies out around the source (lamp in the dark). Unchanged at one\n"
                                           "scene unit from the light; the emitters themselves look the same.\n"
                                           "Saved per scene with the camera (settings.light_falloff).");
                  }
                  ImGui::SliderFloat("Ambient Light", background_intensity, 0.0f, 5.0f, "%.2f");
                  ImGui::SliderFloat("Metal Fuzz", metal_fuzziness, 0.0f, 5.0f, "%.2f");
                  ImGui::SliderFloat("Glass IOR", glass_ior, 1.0f, 2.5f, "%.2f");
                  ImGui::PopItemWidth();
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
                     pushSliderWidth();
                     ImGui::SliderInt("Arrow Count", normal_arrow_count, 40, 2500);
                     ImGui::SliderFloat("Arrow Scale", normal_arrow_scale, 0.2f, 1.8f, "%.2f");
                     ImGui::SliderFloat("Arrow Thickness", normal_arrow_thickness, 1.0f, 2.5f, "%.1f");
                     ImGui::PopItemWidth();
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

            // --- Procedural Patterns ---
            if (golf_dimple_count)
            {
               if (reset_headers)
                  ImGui::SetNextItemOpen(!collapse_headers);
               if (ImGui::CollapsingHeader("Procedural Patterns"))
               {
                  if (golf_dimple_count && golf_dimple_radius && golf_dimple_depth)
                  {
                     ImGui::SeparatorText("Golf Ball Dimples");
                     pushSliderWidth();
                     ImGui::SliderInt("Dimple Count##golf", golf_dimple_count, 50, 400);
                     ImGui::SetItemTooltip("Number of dimples.\nHigher values are more expensive per hit.");
                     ImGui::SliderFloat("Dimple Radius##golf", golf_dimple_radius, 0.05f, 0.6f, "%.3f");
                     ImGui::SetItemTooltip("Angular radius of each dimple (radians).");
                     ImGui::SliderFloat("Dimple Depth##golf", golf_dimple_depth, 0.0f, 1.0f, "%.3f");
                     ImGui::SetItemTooltip("Depth of each dimple (displacement scale).");
                     ImGui::PopItemWidth();
                  }
               }
            }

            // --- Cloud / Smoke: shown when the scene holds a volume material ---
            if (volume)
            {
               if (reset_headers)
                  ImGui::SetNextItemOpen(!collapse_headers);
               if (ImGui::CollapsingHeader("Cloud / Smoke", ImGuiTreeNodeFlags_DefaultOpen))
               {
                  pushSliderWidth();
                  bool moved = false;
                  moved |= ImGui::ColorEdit3("Albedo##vol", volume->albedo, ImGuiColorEditFlags_Float);
                  ImGui::SetItemTooltip("Single-scattering albedo: the share of light that survives each\n"
                                        "scattering event. Near white for a cloud, grey for smoke.");
                  moved |= ImGui::SliderFloat("Density##vol", &volume->density, 0.1f, 30.0f, "%.2f",
                                              ImGuiSliderFlags_Logarithmic);
                  ImGui::SetItemTooltip("Extinction per scene unit: the mean free path is 1 / density.");
                  moved |= ImGui::SliderFloat("Anisotropy g##vol", &volume->g, -0.9f, 0.9f, "%.2f");
                  ImGui::SetItemTooltip("Henyey-Greenstein asymmetry: 0 isotropic, positive forward\n"
                                        "scattering (silver lining when backlit), negative backward.");
                  moved |= ImGui::SliderFloat("Noise scale##vol", &volume->noise_scale, 0.2f, 8.0f, "%.2f",
                                              ImGuiSliderFlags_Logarithmic);
                  ImGui::SetItemTooltip("Frequency of the density noise: features of about 1 / scale units.");
                  moved |= ImGui::SliderFloat("Noise strength##vol", &volume->noise_strength, 0.0f, 2.0f, "%.2f");
                  ImGui::SetItemTooltip("Amplitude of the interior density noise: 0 homogeneous, 1 full\n"
                                        "contrast, above 1 the field has holes.");
                  ImGui::SeparatorText("Shape");
                  moved |= ImGui::SliderFloat("Edge##vol", &volume->edge, 0.0f, 1.0f, "%.2f");
                  ImGui::SetItemTooltip("Depth over which the density ramps up at the surface (scene\n"
                                        "units): small = crisp cloud edge, large = soft haze.\n"
                                        "0 = hard-edged fog filling the mesh (no shape noise).");
                  moved |= ImGui::SliderFloat("Lumps##vol", &volume->lumps, 0.0f, 0.75f, "%.2f"); // max = VOLUME_BOX_PAD
                  ImGui::SetItemTooltip("Amplitude of the surface displacement by the shape noise (scene\n"
                                        "units): bulges where the noise is high, crevices where it is low.\n"
                                        "Needs a reload of the scene to take effect when raised from 0.");
                  moved |= ImGui::SliderFloat("Billow##vol", &volume->billow, 0.0f, 1.0f, "%.2f");
                  ImGui::SetItemTooltip("Share of inverted Worley noise in the shape: rounded cauliflower\n"
                                        "lumps (1) versus soft fractal wisps (0).");
                  ImGui::PopItemWidth();
                  if (moved)
                     volume->changed = true;
               }
            }

            // --- Scene ---
            if (scene_index && scene_names && scene_count > 0)
            {
               if (reset_headers)
                  ImGui::SetNextItemOpen(!collapse_headers);
               if (ImGui::CollapsingHeader("Scene"))
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
               ImGui::BulletText("Numpad +/-: Next/Prev HDR Sky");
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
         // Refresh the displayed values at most 10 times per second (every 100 ms).
         const Uint32 now_ms = SDL_GetTicks();
         if (now_ms - last_spps_refresh_time_ms >= 100u)
         {
            display_sps               = sps;
            display_fps               = fps;
            last_spps_refresh_time_ms = now_ms;
         }
         drawSPPSOverlay(display_sps, display_fps);
      }

      // Finalize ImGui frame
      ImGui::Render();
      ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
   }

   void drawLogo()
   {
      if (show_logo && logo_texture)
      {
         updateLogoLayout();
         SDL_RenderCopy(renderer, logo_texture, nullptr, &logo_rect);
      }
   }

   /// Current size of the drawable area (follows window resizes)
   void outputSize(int &w, int &h) const
   {
      w = image_width;
      h = image_height;
      if (renderer)
         SDL_GetRendererOutputSize(renderer, &w, &h);
   }

   /// Largest centred rectangle with the image's aspect ratio that fits in the window
   SDL_Rect imageDestRect() const
   {
      int out_w, out_h;
      outputSize(out_w, out_h);
      SDL_Rect r;
      if ((long long)out_w * image_height <= (long long)out_h * image_width)
      {
         r.w = out_w;
         r.h = std::max(1, (int)((long long)out_w * image_height / image_width));
      }
      else
      {
         r.h = out_h;
         r.w = std::max(1, (int)((long long)out_h * image_width / image_height));
      }
      r.x = (out_w - r.w) / 2;
      r.y = (out_h - r.h) / 2;
      return r;
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
   std::string backend_name;
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

   // Throttled display values for SPP/s and FPS — updated at most 10×/s.
   float  display_sps               = 0.0f;
   float  display_fps               = 0.0f;
   Uint32 last_spps_refresh_time_ms = 0;

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
      int out_w, out_h;
      outputSize(out_w, out_h);
      logo_rect.x = out_w - logo_rect.w - margin;
      logo_rect.y = out_h - logo_rect.h - margin;
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
      const float box_y     = ImGui::GetIO().DisplaySize.y - box_h - margin;

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
