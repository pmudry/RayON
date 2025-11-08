/**
 * @class SDLGuiHandler
 * @brief Handles all SDL GUI components for the interactive renderer
 *
 * This class manages:
 * - SDL window and renderer creation/cleanup
 * - Logo loading and display
 * - Font loading and text rendering
 * - UI controls (sliders, buttons)
 * - Mouse and keyboard event handling
 * - Camera control state
 */
#pragma once

#ifdef SDL2_FOUND

#include "vec3.h"

#include <SDL.h>
#ifdef SDL2_TTF_FOUND
#include <SDL_ttf.h>
#endif
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

// Structure to hold slider interaction bounds
struct SliderBounds
{
   int x, y, width, height;
   float min_val, max_val;
   float *value_ptr;
};

class SDLGuiHandler
{
 public:
   SDLGuiHandler(int image_width, int image_height)
       : image_width(image_width), image_height(image_height), show_controls(true), window(nullptr), renderer(nullptr),
         texture(nullptr), logo_texture(nullptr), font(nullptr)
   {
   }

   ~SDLGuiHandler() { cleanup(); }

   bool initialize()
   {
      if (SDL_Init(SDL_INIT_VIDEO) < 0)
      {
         cerr << "SDL initialization failed: " << SDL_GetError() << endl;
         return false;
      }

#ifdef SDL2_TTF_FOUND
      if (TTF_Init() < 0)
      {
         cerr << "SDL_ttf initialization failed: " << TTF_GetError() << endl;
         SDL_Quit();
         return false;
      }
#endif

      window =
          SDL_CreateWindow("ISC - 302 ray tracer (mui) / Continuous mode (LMB:Rotate RMB:Pan Wheel:Zoom)",
                           SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, image_width, image_height, SDL_WINDOW_SHOWN);
      if (!window)
      {
         cerr << "Window creation failed: " << SDL_GetError() << endl;
         cleanupSDL();
         return false;
      }

      renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
      if (!renderer)
      {
         cerr << "Renderer creation failed: " << SDL_GetError() << endl;
         SDL_DestroyWindow(window);
         cleanupSDL();
         return false;
      }

      texture =
          SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, image_width, image_height);
      if (!texture)
      {
         cerr << "Texture creation failed: " << SDL_GetError() << endl;
         SDL_DestroyRenderer(renderer);
         SDL_DestroyWindow(window);
         cleanupSDL();
         return false;
      }

      loadLogo();
      loadFont();

      return true;
   }

   void cleanup()
   {
#ifdef SDL2_TTF_FOUND
      if (font)
      {
         TTF_CloseFont(static_cast<TTF_Font *>(font));
         font = nullptr;
      }
#endif

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

   void updateDisplay(const vector<unsigned char> &image, int image_channels)
   {
      SDL_UpdateTexture(texture, nullptr, image.data(), image_width * image_channels);
      SDL_RenderClear(renderer);
      SDL_RenderCopy(renderer, texture, nullptr, nullptr);
   }

   void drawLogo()
   {
      if (logo_texture)
      {
         SDL_RenderCopy(renderer, logo_texture, nullptr, &logo_rect);
      }
   }

   void present() { SDL_RenderPresent(renderer); }

   void drawSampleCountText(int sample_count)
   {
      // Don't draw sample count if controls are hidden
      if (!show_controls)
         return;

#ifdef SDL2_TTF_FOUND
      TTF_Font *ttf_font = static_cast<TTF_Font *>(font);
      if (!ttf_font)
         return;

      std::string text = std::to_string(sample_count) + " SPP";

      SDL_Color white = {255, 255, 255, 255};
      SDL_Surface *text_surface = TTF_RenderText_Blended(ttf_font, text.c_str(), white);
      if (!text_surface)
         return;

      SDL_Texture *text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);

      if (text_texture)
      {
         int text_width = text_surface->w;
         int text_height = text_surface->h;
         int padding = 15;
         int box_width = 90;

         SDL_Rect bg_rect = {image_width - box_width - padding, padding - 5, box_width, text_height + 10};
         SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
         SDL_SetRenderDrawColor(renderer, 0, 0, 0, 200);
         SDL_RenderFillRect(renderer, &bg_rect);

         SDL_Rect text_rect = {image_width - padding - box_width + (box_width - text_width) / 2, padding, text_width,
                               text_height};
         SDL_RenderCopy(renderer, text_texture, nullptr, &text_rect);

         SDL_DestroyTexture(text_texture);
      }

      SDL_FreeSurface(text_surface);
#endif
   }

   void drawUIControls(int samples_per_batch, float light_intensity, float background_intensity, float metal_fuzziness,
                       float glass_refraction_index,
                       bool accumulation_enabled, bool auto_orbit_enabled,
                       SliderBounds &samples_slider_bounds,
                       SliderBounds &intensity_slider_bounds, SliderBounds &background_slider_bounds,
                       SliderBounds &fuzziness_slider_bounds, SliderBounds &glass_ior_slider_bounds,
                       SDL_Rect &toggle_button_rect, SDL_Rect &orbit_button_rect)
   {
      // Don't draw controls if they're hidden
      if (!show_controls)
         return;

#ifdef SDL2_TTF_FOUND
      TTF_Font *ttf_font = static_cast<TTF_Font *>(font);
      if (!ttf_font)
         return;

      TTF_Font *small_font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12);
      if (!small_font)
         small_font = ttf_font;

      int padding = 15;
      int control_width = 280;
      int label_width = 120;
      int slider_height = 25;
      int spacing = 8;
      int button_row_height = slider_height; // Height for row with two buttons
      int start_y = image_height - (6 * slider_height + button_row_height + 6 * spacing + padding);

      SDL_Color white = {255, 255, 255, 255};

      SDL_Rect bg_rect = {padding - 5, start_y - 5, control_width + 10,
                          5 * slider_height + button_row_height + 6 * spacing + 10};
      SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
      SDL_SetRenderDrawColor(renderer, 0, 0, 0, 200);
      SDL_RenderFillRect(renderer, &bg_rect);

      // Draw two toggle buttons side by side
      int button_width = (control_width - spacing) / 2;
      drawToggleButton(small_font, padding, start_y, accumulation_enabled, white, toggle_button_rect, "Auto-Accum",
                       button_width);
      drawOrbitButton(small_font, padding + button_width + spacing, start_y, auto_orbit_enabled, white,
                      orbit_button_rect, "Auto-Orbit", button_width);

      drawSamplesSlider(small_font, padding, start_y + button_row_height + spacing, control_width, samples_per_batch,
                        white, samples_slider_bounds, label_width);
      drawLightSlider(small_font, padding, start_y + button_row_height + slider_height + 2 * spacing, control_width,
                      light_intensity, white, intensity_slider_bounds, label_width);
      drawBackgroundSlider(small_font, padding, start_y + button_row_height + 2 * slider_height + 3 * spacing,
                           control_width, background_intensity, white, background_slider_bounds, label_width);
      drawFuzzinessSlider(small_font, padding, start_y + button_row_height + 3 * slider_height + 4 * spacing,
                          control_width, metal_fuzziness, white, fuzziness_slider_bounds, label_width);
      drawGlassIORSlider(small_font, padding, start_y + button_row_height + 4 * slider_height + 5 * spacing,
                         control_width, glass_refraction_index, white, glass_ior_slider_bounds, label_width);

      if (small_font != ttf_font)
      {
         TTF_CloseFont(small_font);
      }
#endif
   }

   void drawEffectsPanel(bool dof_enabled, float dof_aperture, float dof_focus_distance,
                         SliderBounds &dof_aperture_slider_bounds, SliderBounds &dof_focus_slider_bounds,
                         SDL_Rect &dof_button_rect)
   {
      // Don't draw effects panel if controls are hidden
      if (!show_controls)
         return;

#ifdef SDL2_TTF_FOUND
      TTF_Font *ttf_font = static_cast<TTF_Font *>(font);
      if (!ttf_font)
         return;

      TTF_Font *small_font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12);
      if (!small_font)
      {
         small_font = TTF_OpenFont("/usr/share/fonts/TTF/DejaVuSans.ttf", 12);
      }
      if (!small_font)
      {
         small_font = ttf_font;
      }

      int padding = 15;
      int control_width = 220;
      int label_width = 130;
      int slider_height = 25;
      int spacing = 8;
      int button_row_height = slider_height;
      
      // Position at top-right corner
      int start_x = image_width - control_width - padding;
      int start_y = padding + 40; // Below the sample count

      SDL_Color white = {255, 255, 255, 255};

      // Background for effects panel
      SDL_Rect bg_rect = {start_x - 5, start_y - 5, control_width + 10,
                          button_row_height + 2 * slider_height + 3 * spacing + 10};
      SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
      SDL_SetRenderDrawColor(renderer, 0, 0, 0, 200);
      SDL_RenderFillRect(renderer, &bg_rect);

      // DOF toggle button (full width)
      drawToggleButton(small_font, start_x, start_y, dof_enabled, white, dof_button_rect, "Depth of Field",
                       control_width);

      // DOF sliders
      drawDOFApertureSlider(small_font, start_x, start_y + button_row_height + spacing, control_width, dof_aperture,
                           white, dof_aperture_slider_bounds, label_width);
      drawDOFFocusSlider(small_font, start_x, start_y + button_row_height + slider_height + 2 * spacing,
                        control_width, dof_focus_distance, white, dof_focus_slider_bounds, label_width);

      if (small_font != ttf_font)
      {
         TTF_CloseFont(small_font);
      }
#endif
   }

   void printControls(int samples_per_batch, int max_samples, bool auto_accumulate)
   {
      cout << "\n=== Interactive Ray Tracing with Real-time Display ===" << endl;
      cout << "Controls:" << endl;
      cout << "  Left Mouse Button:   Rotate camera (orbit)" << endl;
      cout << "  Right Mouse Button:  Pan camera" << endl;
      cout << "  Mouse Wheel:         Zoom in/out" << endl;
      cout << "  SPACEBAR:            Toggle automatic accumulation" << endl;
      cout << "  O:                   Toggle auto-orbit camera" << endl;
      cout << "  H:                   Hide/show GUI controls" << endl;
      cout << "  Up/Down Arrows:      Adjust samples per batch (1-256)" << endl;
      cout << "  Left/Right Arrows:   Adjust light intensity (0.1-3.0)" << endl;
      cout << "  ESC:                 Exit" << endl << endl;
      cout << "Sample accumulation: " << samples_per_batch << " samples per batch, up to " << max_samples
           << " total samples" << endl;
      cout << "Auto-accumulation: " << (auto_accumulate ? "ON" : "OFF") << endl;
   }

   // Event handling
   bool pollEvent(SDL_Event &event) { return SDL_PollEvent(&event); }

   // Getters
   SDL_Window *getWindow() { return window; }
   SDL_Renderer *getRenderer() { return renderer; }
   SDL_Texture *getTexture() { return texture; }
   SDL_Texture *getLogoTexture() { return logo_texture; }
   const SDL_Rect &getLogoRect() const { return logo_rect; }
   void *getFont() { return font; }
   bool getShowControls() const { return show_controls; }

   // Control visibility toggle
   void toggleControls() { show_controls = !show_controls; }

 private:
   int image_width;
   int image_height;
   bool show_controls; // Flag to show/hide GUI controls

   SDL_Window *window;
   SDL_Renderer *renderer;
   SDL_Texture *texture;
   SDL_Texture *logo_texture;
   SDL_Rect logo_rect;
   void *font;

   void cleanupSDL()
   {
#ifdef SDL2_TTF_FOUND
      TTF_Quit();
#endif
      SDL_Quit();
   }

   void loadLogo()
   {
      int logo_img_width, logo_img_height, logo_img_channels;
      unsigned char *logo_data = stbi_load("../resources/ISC Logo inline white v3 - 1500px.png", &logo_img_width,
                                           &logo_img_height, &logo_img_channels, 4);

      if (logo_data)
      {
         int target_logo_width = image_width / 5;
         int target_logo_height = (logo_img_height * target_logo_width) / logo_img_width;

         unsigned char *resized_logo = new unsigned char[target_logo_width * target_logo_height * 4];
         stbir_resize_uint8_srgb(logo_data, logo_img_width, logo_img_height, 0, resized_logo, target_logo_width,
                                 target_logo_height, 0, STBIR_RGBA);

         SDL_Surface *logo_surface =
             SDL_CreateRGBSurfaceFrom(resized_logo, target_logo_width, target_logo_height, 32, target_logo_width * 4,
                                      0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);

         if (logo_surface)
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

   void loadFont()
   {
#ifdef SDL2_TTF_FOUND
      font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 16);
      if (!font)
      {
         font = TTF_OpenFont("/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf", 16);
      }
      if (!font)
      {
         cerr << "Warning: Could not load font: " << TTF_GetError() << endl;
      }
#endif
   }

#ifdef SDL2_TTF_FOUND
   void drawToggleButton(TTF_Font *ttf_font, int x, int y, bool enabled, SDL_Color white, SDL_Rect &button_rect,
                         const char *label, int max_width)
   {
      int box_size = 14;
      int box_y = y + 5;
      SDL_Rect checkbox_bg = {x, box_y, box_size, box_size};

      SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
      SDL_RenderDrawRect(renderer, &checkbox_bg);

      SDL_Rect checkbox_fill = {x + 2, box_y + 2, box_size - 4, box_size - 4};
      if (enabled)
      {
         SDL_SetRenderDrawColor(renderer, 0, 200, 0, 255);
      }
      else
      {
         SDL_SetRenderDrawColor(renderer, 200, 0, 0, 255);
      }
      SDL_RenderFillRect(renderer, &checkbox_fill);

      SDL_Surface *text_surface = TTF_RenderText_Blended(ttf_font, label, white);
      if (text_surface)
      {
         SDL_Texture *text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
         if (text_texture)
         {
            SDL_Rect text_rect = {x + box_size + 8, y + 5, text_surface->w, text_surface->h};
            SDL_RenderCopy(renderer, text_texture, nullptr, &text_rect);
            SDL_DestroyTexture(text_texture);
         }
         SDL_FreeSurface(text_surface);
      }

      button_rect.x = x;
      button_rect.y = box_y;
      button_rect.w = std::min(max_width, box_size + 8 + 80);
      button_rect.h = box_size;
   }

   void drawOrbitButton(TTF_Font *ttf_font, int x, int y, bool enabled, SDL_Color white, SDL_Rect &button_rect,
                        const char *label, int max_width)
   {
      drawToggleButton(ttf_font, x, y, enabled, white, button_rect, label, max_width);
   }

   void drawSamplesSlider(TTF_Font *ttf_font, int padding, int y, int control_width, int samples_per_batch,
                          SDL_Color white, SliderBounds &samples_slider, int label_width)
   {
      char label[32];
      snprintf(label, sizeof(label), "Samples: %d", samples_per_batch);
      SDL_Surface *text_surface = TTF_RenderText_Blended(ttf_font, label, white);
      if (text_surface)
      {
         SDL_Texture *text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
         if (text_texture)
         {
            SDL_Rect text_rect = {padding, y + 5, text_surface->w, text_surface->h};
            SDL_RenderCopy(renderer, text_texture, nullptr, &text_rect);
            SDL_DestroyTexture(text_texture);
         }
         SDL_FreeSurface(text_surface);
      }

      int slider_x = padding + label_width;
      int slider_w = control_width - label_width;
      SDL_Rect slider_bg = {slider_x, y + 8, slider_w, 8};
      SDL_SetRenderDrawColor(renderer, 60, 60, 60, 255);
      SDL_RenderFillRect(renderer, &slider_bg);

      samples_slider.x = slider_x;
      samples_slider.y = y + 8;
      samples_slider.width = slider_w;
      samples_slider.height = 8;
      samples_slider.min_val = 1.0f;
      samples_slider.max_val = 256.0f;

      float samples_ratio = (float(samples_per_batch) - 1.0f) / (256.0f - 1.0f);
      int fill_w = static_cast<int>(slider_w * samples_ratio);
      SDL_Rect slider_fill = {slider_x, y + 8, fill_w, 8};
      SDL_SetRenderDrawColor(renderer, 100, 150, 255, 255);
      SDL_RenderFillRect(renderer, &slider_fill);

      int handle_x = slider_x + fill_w - 3;
      SDL_Rect handle = {handle_x, y + 4, 6, 16};
      SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
      SDL_RenderFillRect(renderer, &handle);
   }

   void drawLightSlider(TTF_Font *ttf_font, int padding, int y, int control_width, float light_intensity,
                        SDL_Color white, SliderBounds &intensity_slider, int label_width)
   {
      char label[32];
      snprintf(label, sizeof(label), "Light: %.1f", light_intensity);
      SDL_Surface *text_surface = TTF_RenderText_Blended(ttf_font, label, white);
      if (text_surface)
      {
         SDL_Texture *text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
         if (text_texture)
         {
            SDL_Rect text_rect = {padding, y + 5, text_surface->w, text_surface->h};
            SDL_RenderCopy(renderer, text_texture, nullptr, &text_rect);
            SDL_DestroyTexture(text_texture);
         }
         SDL_FreeSurface(text_surface);
      }

      int slider_x = padding + label_width;
      int slider_w = control_width - label_width;
      SDL_Rect slider_bg = {slider_x, y + 8, slider_w, 8};
      SDL_SetRenderDrawColor(renderer, 60, 60, 60, 255);
      SDL_RenderFillRect(renderer, &slider_bg);

      intensity_slider.x = slider_x;
      intensity_slider.y = y + 8;
      intensity_slider.width = slider_w;
      intensity_slider.height = 8;
      intensity_slider.min_val = 0.1f;
      intensity_slider.max_val = 3.0f;

      float intensity_ratio = (light_intensity - 0.1f) / (3.0f - 0.1f);
      int fill_w = static_cast<int>(slider_w * intensity_ratio);
      SDL_Rect slider_fill = {slider_x, y + 8, fill_w, 8};
      SDL_SetRenderDrawColor(renderer, 255, 200, 100, 255);
      SDL_RenderFillRect(renderer, &slider_fill);

      int handle_x = slider_x + fill_w - 3;
      SDL_Rect handle = {handle_x, y + 4, 6, 16};
      SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
      SDL_RenderFillRect(renderer, &handle);
   }

   void drawBackgroundSlider(TTF_Font *ttf_font, int padding, int y, int control_width, float background_intensity,
                             SDL_Color white, SliderBounds &background_slider, int label_width)
   {
      char label[32];
      snprintf(label, sizeof(label), "Background: %.2f", background_intensity);
      SDL_Surface *text_surface = TTF_RenderText_Blended(ttf_font, label, white);
      if (text_surface)
      {
         SDL_Texture *text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
         if (text_texture)
         {
            SDL_Rect text_rect = {padding, y + 5, text_surface->w, text_surface->h};
            SDL_RenderCopy(renderer, text_texture, nullptr, &text_rect);
            SDL_DestroyTexture(text_texture);
         }
         SDL_FreeSurface(text_surface);
      }

      int slider_x = padding + label_width;
      int slider_w = control_width - label_width;
      SDL_Rect slider_bg = {slider_x, y + 8, slider_w, 8};
      SDL_SetRenderDrawColor(renderer, 60, 60, 60, 255);
      SDL_RenderFillRect(renderer, &slider_bg);

      background_slider.x = slider_x;
      background_slider.y = y + 8;
      background_slider.width = slider_w;
      background_slider.height = 8;
      background_slider.min_val = 0.0f;
      background_slider.max_val = 3.0f;

      float bg_ratio = (background_intensity - 0.0f) / (3.0f - 0.0f);
      int fill_w = static_cast<int>(slider_w * bg_ratio);
      SDL_Rect slider_fill = {slider_x, y + 8, fill_w, 8};
      SDL_SetRenderDrawColor(renderer, 150, 100, 255, 255);
      SDL_RenderFillRect(renderer, &slider_fill);

      int handle_x = slider_x + fill_w - 3;
      SDL_Rect handle = {handle_x, y + 4, 6, 16};
      SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
      SDL_RenderFillRect(renderer, &handle);
   }

   void drawFuzzinessSlider(TTF_Font *ttf_font, int padding, int y, int control_width, float metal_fuzziness,
                            SDL_Color white, SliderBounds &fuzziness_slider, int label_width)
   {
      char label[32];
      snprintf(label, sizeof(label), "Metal fuzz: %.2f", metal_fuzziness);
      SDL_Surface *text_surface = TTF_RenderText_Blended(ttf_font, label, white);
      if (text_surface)
      {
         SDL_Texture *text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
         if (text_texture)
         {
            SDL_Rect text_rect = {padding, y + 5, text_surface->w, text_surface->h};
            SDL_RenderCopy(renderer, text_texture, nullptr, &text_rect);
            SDL_DestroyTexture(text_texture);
         }
         SDL_FreeSurface(text_surface);
      }

      int slider_x = padding + label_width;
      int slider_w = control_width - label_width;
      SDL_Rect slider_bg = {slider_x, y + 8, slider_w, 8};
      SDL_SetRenderDrawColor(renderer, 60, 60, 60, 255);
      SDL_RenderFillRect(renderer, &slider_bg);

      fuzziness_slider.x = slider_x;
      fuzziness_slider.y = y + 8;
      fuzziness_slider.width = slider_w;
      fuzziness_slider.height = 8;
      fuzziness_slider.min_val = 0.0f;
      fuzziness_slider.max_val = 5.0f;

      float fuzz_ratio = (metal_fuzziness - 0.0f) / (5.0f - 0.0f);
      int fill_w = static_cast<int>(slider_w * fuzz_ratio);
      SDL_Rect slider_fill = {slider_x, y + 8, fill_w, 8};
      SDL_SetRenderDrawColor(renderer, 200, 200, 100, 255);
      SDL_RenderFillRect(renderer, &slider_fill);

      int handle_x = slider_x + fill_w - 3;
      SDL_Rect handle = {handle_x, y + 4, 6, 16};
      SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
      SDL_RenderFillRect(renderer, &handle);
   }

   void drawGlassIORSlider(TTF_Font *ttf_font, int padding, int y, int control_width, float glass_ior,
                           SDL_Color white, SliderBounds &glass_ior_slider, int label_width)
   {
      char label[32];
      snprintf(label, sizeof(label), "Glass IOR: %.2f", glass_ior);
      SDL_Surface *text_surface = TTF_RenderText_Blended(ttf_font, label, white);
      if (text_surface)
      {
         SDL_Texture *text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
         if (text_texture)
         {
            SDL_Rect text_rect = {padding, y + 5, text_surface->w, text_surface->h};
            SDL_RenderCopy(renderer, text_texture, nullptr, &text_rect);
            SDL_DestroyTexture(text_texture);
         }
         SDL_FreeSurface(text_surface);
      }

      int slider_x = padding + label_width;
      int slider_w = control_width - label_width;
      SDL_Rect slider_bg = {slider_x, y + 8, slider_w, 8};
      SDL_SetRenderDrawColor(renderer, 60, 60, 60, 255);
      SDL_RenderFillRect(renderer, &slider_bg);

      glass_ior_slider.x = slider_x;
      glass_ior_slider.y = y + 8;
      glass_ior_slider.width = slider_w;
      glass_ior_slider.height = 8;
      glass_ior_slider.min_val = 1.0f;
      glass_ior_slider.max_val = 2.5f;

      float ior_ratio = (glass_ior - 1.0f) / (2.5f - 1.0f);
      int fill_w = static_cast<int>(slider_w * ior_ratio);
      SDL_Rect slider_fill = {slider_x, y + 8, fill_w, 8};
      SDL_SetRenderDrawColor(renderer, 100, 200, 255, 255);
      SDL_RenderFillRect(renderer, &slider_fill);

      int handle_x = slider_x + fill_w - 3;
      SDL_Rect handle = {handle_x, y + 4, 6, 16};
      SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
      SDL_RenderFillRect(renderer, &handle);
   }

   void drawDOFApertureSlider(TTF_Font *ttf_font, int padding, int y, int control_width, float dof_aperture,
                              SDL_Color white, SliderBounds &aperture_slider, int label_width)
   {
      char label[32];
      snprintf(label, sizeof(label), "DOF Aperture: %.2f", dof_aperture);
      SDL_Surface *text_surface = TTF_RenderText_Blended(ttf_font, label, white);
      if (text_surface)
      {
         SDL_Texture *text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
         if (text_texture)
         {
            SDL_Rect text_rect = {padding, y + 5, text_surface->w, text_surface->h};
            SDL_RenderCopy(renderer, text_texture, nullptr, &text_rect);
            SDL_DestroyTexture(text_texture);
         }
         SDL_FreeSurface(text_surface);
      }

      int slider_x = padding + label_width;
      int slider_w = control_width - label_width;
      SDL_Rect slider_bg = {slider_x, y + 8, slider_w, 8};
      SDL_SetRenderDrawColor(renderer, 60, 60, 60, 255);
      SDL_RenderFillRect(renderer, &slider_bg);

      aperture_slider.x = slider_x;
      aperture_slider.y = y + 8;
      aperture_slider.width = slider_w;
      aperture_slider.height = 8;
      aperture_slider.min_val = 0.0f;
      aperture_slider.max_val = 1.0f;

      float ratio = (dof_aperture - 0.0f) / (1.0f - 0.0f);
      int fill_w = static_cast<int>(slider_w * ratio);
      SDL_Rect slider_fill = {slider_x, y + 8, fill_w, 8};
      SDL_SetRenderDrawColor(renderer, 100, 200, 255, 255);
      SDL_RenderFillRect(renderer, &slider_fill);

      int handle_x = slider_x + fill_w - 3;
      SDL_Rect handle = {handle_x, y + 4, 6, 16};
      SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
      SDL_RenderFillRect(renderer, &handle);
   }

   void drawDOFFocusSlider(TTF_Font *ttf_font, int padding, int y, int control_width, float dof_focus_distance,
                           SDL_Color white, SliderBounds &focus_slider, int label_width)
   {
      char label[32];
      snprintf(label, sizeof(label), "DOF Focus: %.1f", dof_focus_distance);
      SDL_Surface *text_surface = TTF_RenderText_Blended(ttf_font, label, white);
      if (text_surface)
      {
         SDL_Texture *text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
         if (text_texture)
         {
            SDL_Rect text_rect = {padding, y + 5, text_surface->w, text_surface->h};
            SDL_RenderCopy(renderer, text_texture, nullptr, &text_rect);
            SDL_DestroyTexture(text_texture);
         }
         SDL_FreeSurface(text_surface);
      }

      int slider_x = padding + label_width;
      int slider_w = control_width - label_width;
      SDL_Rect slider_bg = {slider_x, y + 8, slider_w, 8};
      SDL_SetRenderDrawColor(renderer, 60, 60, 60, 255);
      SDL_RenderFillRect(renderer, &slider_bg);

      focus_slider.x = slider_x;
      focus_slider.y = y + 8;
      focus_slider.width = slider_w;
      focus_slider.height = 8;
      focus_slider.min_val = 1.0f;
      focus_slider.max_val = 50.0f;

      float ratio = (dof_focus_distance - 1.0f) / (50.0f - 1.0f);
      int fill_w = static_cast<int>(slider_w * ratio);
      SDL_Rect slider_fill = {slider_x, y + 8, fill_w, 8};
      SDL_SetRenderDrawColor(renderer, 255, 150, 100, 255);
      SDL_RenderFillRect(renderer, &slider_fill);

      int handle_x = slider_x + fill_w - 3;
      SDL_Rect handle = {handle_x, y + 4, 6, 16};
      SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
      SDL_RenderFillRect(renderer, &handle);
   }
#endif
};

#endif // SDL2_FOUND
