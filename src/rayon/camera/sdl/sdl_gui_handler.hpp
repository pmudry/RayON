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

#include <iostream>
#include <string>
#include <vector>

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;

class SDLGuiHandler
{
 public:
   SDLGuiHandler(int image_width, int image_height)
       : image_width(image_width), image_height(image_height), show_controls(true), collapse_headers(false),
         reset_headers(false), window(nullptr), renderer(nullptr), texture(nullptr), logo_texture(nullptr)
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
          "RayON v" + std::string(constants::version) + " - Interactive mode (LMB:Rotate RMB:Pan Wheel:Zoom)";
      window = SDL_CreateWindow(window_title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, image_width,
                                image_height, SDL_WINDOW_SHOWN);
      if (!window)
      {
         cerr << "Window creation failed: " << SDL_GetError() << "\n";
         cleanupSDL();
         return false;
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
      io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
      ImGui::StyleColorsDark();

      // Semi-transparent window backgrounds
      ImGuiStyle &style = ImGui::GetStyle();
      style.Colors[ImGuiCol_WindowBg].w = 0.35f;

      ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
      ImGui_ImplSDLRenderer2_Init(renderer);
      return true;
   }

   void cleanup()
   {
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
   void updateDisplay(const vector<unsigned char> &image, int image_channels, float sps, float ms_per_sample, int spp,
                      bool *dof_enabled, float *aperture, float *focus_dist, float *light_intensity,
                      float *background_intensity, float *metal_fuzziness, float *glass_ior,
                      float *samples_per_batch, bool *auto_accumulate, bool *auto_orbit)
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
         ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
         if (ImGui::Begin("RayON - Interactive UI", nullptr,
                          ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove))
         {
            // --- Performance Monitoring ---
            if (reset_headers)
               ImGui::SetNextItemOpen(!collapse_headers);
            if (ImGui::CollapsingHeader("Performance Monitoring", ImGuiTreeNodeFlags_DefaultOpen))
            {
               ImGui::Text("SPP: %d", spp);
               ImGui::Text("Throughput: %.0f SPS", sps);
               ImGui::Text("Time/Sample: %.3f ms", ms_per_sample);

               sps_history.push_back(sps);
               ms_history.push_back(ms_per_sample);

               if (sps_history.size() > 500)
               {
                  sps_history.erase(sps_history.begin());
                  ms_history.erase(ms_history.begin());
               }

               if (!sps_history.empty())
               {
                  float max_sps = 0.0f;
                  for (float f : sps_history)
                     max_sps = std::max(max_sps, f);

                  ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
                  ImGui::PlotLines("Live SPS", sps_history.data(), static_cast<int>(sps_history.size()), 0, nullptr,
                                   0.0f, max_sps * 1.1f, ImVec2(ImGui::CalcItemWidth(), 50));
                  ImGui::PopStyleColor();

                  float max_ms = 0.0f;
                  for (float f : ms_history)
                     max_ms = std::max(max_ms, f);

                  ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(1.0f, 0.7f, 0.0f, 1.0f));
                  ImGui::PlotLines("Time/Sample", ms_history.data(), static_cast<int>(ms_history.size()), 0, nullptr,
                                   0.0f, max_ms * 1.1f, ImVec2(ImGui::CalcItemWidth(), 50));
                  ImGui::PopStyleColor();
               }

               if (samples_per_batch && auto_accumulate)
               {
                  ImGui::Separator();
                  ImGui::SliderFloat("Samples/Batch", samples_per_batch, 1.0f, 256.0f, "%.0f");
                  ImGui::Checkbox("Auto-Accumulate (Space)", auto_accumulate);
               }
            }

            // --- Camera Settings ---
            if (reset_headers)
               ImGui::SetNextItemOpen(!collapse_headers);
            if (ImGui::CollapsingHeader("Camera Settings", ImGuiTreeNodeFlags_DefaultOpen))
            {
               if (auto_orbit)
               {
                  ImGui::Checkbox("Auto-Orbit (O)", auto_orbit);
               }

               if (dof_enabled && aperture && focus_dist)
               {
                  ImGui::Checkbox("Enable Depth of Field", dof_enabled);
                  ImGui::SeparatorText("Lens Controls");

                  if (!(*dof_enabled))
                     ImGui::BeginDisabled();
                  ImGui::SliderFloat("Aperture", aperture, 0.0f, 1.0f, "%.2f");
                  ImGui::SliderFloat("Focus Dist", focus_dist, 0.1f, 100.0f, "%.1f");
                  if (!(*dof_enabled))
                     ImGui::EndDisabled();
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
                  ImGui::SliderFloat("Background", background_intensity, 0.0f, 5.0f, "%.2f");
                  ImGui::SliderFloat("Metal Fuzz", metal_fuzziness, 0.0f, 5.0f, "%.2f");
                  ImGui::SliderFloat("Glass IOR", glass_ior, 1.0f, 2.5f, "%.2f");
               }
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
               ImGui::BulletText("O: Auto-Orbit");
               ImGui::BulletText("H: Toggle UI");
               ImGui::BulletText("C: Collapse/Expand All");
               ImGui::BulletText("R: Reset Defaults");
               ImGui::BulletText("ESC: Exit");
            }

            if (reset_headers)
               reset_headers = false;
         }
         ImGui::End();
      }

      // Finalize ImGui frame
      ImGui::Render();
      ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
   }

   void drawLogo()
   {
      if (logo_texture)
      {
         SDL_RenderCopy(renderer, logo_texture, nullptr, &logo_rect);
      }
   }

   void present() { SDL_RenderPresent(renderer); }

   // Event handling — routes to ImGui first
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

   SDL_Window *window;
   SDL_Renderer *renderer;
   SDL_Texture *texture;
   SDL_Texture *logo_texture;
   SDL_Rect logo_rect;
   std::vector<float> sps_history;
   std::vector<float> ms_history;

   static void cleanupSDL() { SDL_Quit(); }

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
};
#endif // SDL2_FOUND
