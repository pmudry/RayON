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

#include "constants.hpp"
#ifdef SDL2_FOUND

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
         cerr << "SDL initialization failed: " << SDL_GetError() << "\n";
         return false;
      }

#ifdef SDL2_TTF_FOUND
      if (TTF_Init() < 0)
      {
         cerr << "SDL_ttf initialization failed: " << TTF_GetError() << "\n";
         SDL_Quit();
         return false;
      }
#endif

      std::string window_title =
          "RayON (mui) v" + std::string(constants::version) + " - Interactive mode (LMB:Rotate RMB:Pan Wheel:Zoom)";
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
                       float glass_refraction_index, bool accumulation_enabled, bool auto_orbit_enabled,
                       SliderBounds &samples_slider_bounds, SliderBounds &intensity_slider_bounds,
                       SliderBounds &background_slider_bounds, SliderBounds &fuzziness_slider_bounds,
                       SliderBounds &glass_ior_slider_bounds, SDL_Rect &toggle_button_rect, SDL_Rect &orbit_button_rect)
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

      drawPanelBackground(padding, start_y, control_width, 5 * slider_height + button_row_height + 6 * spacing);

      // Draw two toggle buttons side by side
      int button_width = (control_width - spacing) / 2;
      drawToggleButton(small_font, padding, start_y, accumulation_enabled, white, toggle_button_rect, "Auto-Accum",
                       button_width);
      drawToggleButton(small_font, padding + button_width + spacing, start_y, auto_orbit_enabled, white,
                      orbit_button_rect, "Auto-Orbit", button_width);

      drawSamplesSlider(small_font, padding, start_y + button_row_height + spacing, control_width, samples_per_batch,
                        samples_slider_bounds, label_width);
      drawLightSlider(small_font, padding, start_y + button_row_height + slider_height + 2 * spacing, control_width,
                      light_intensity, intensity_slider_bounds, label_width);
      drawBackgroundSlider(small_font, padding, start_y + button_row_height + 2 * slider_height + 3 * spacing,
                           control_width, background_intensity, background_slider_bounds, label_width);
      drawFuzzinessSlider(small_font, padding, start_y + button_row_height + 3 * slider_height + 4 * spacing,
                          control_width, metal_fuzziness, fuzziness_slider_bounds, label_width);
      drawGlassIORSlider(small_font, padding, start_y + button_row_height + 4 * slider_height + 5 * spacing,
                         control_width, glass_refraction_index, glass_ior_slider_bounds, label_width);

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
      drawPanelBackground(start_x, start_y, control_width, button_row_height + 2 * slider_height + 3 * spacing);

      // DOF toggle button (full width)
      drawToggleButton(small_font, start_x, start_y, dof_enabled, white, dof_button_rect, "Depth of Field",
                       control_width);

      // DOF sliders
      drawDOFApertureSlider(small_font, start_x, start_y + button_row_height + spacing, control_width, dof_aperture,
                            dof_aperture_slider_bounds, label_width);
      drawDOFFocusSlider(small_font, start_x, start_y + button_row_height + slider_height + 2 * spacing, control_width,
                         dof_focus_distance, dof_focus_slider_bounds, label_width);

      if (small_font != ttf_font)
      {
         TTF_CloseFont(small_font);
      }
#endif
   }

   static void printControls(int samples_per_batch, int max_samples, bool auto_accumulate)
   {
      cout << "\n=== Interactive Ray Tracing with Real-time Display ===" << "\n";
      cout << "Controls:" "\n";
      cout << "  Left Mouse Button:   Rotate camera (orbit)" "\n";
      cout << "  Right Mouse Button:  Pan camera" "\n";
      cout << "  Mouse Wheel:         Zoom in/out" "\n";
      cout << "  SPACEBAR:            Toggle automatic accumulation" "\n";
      cout << "  O:                   Toggle auto-orbit camera" "\n";
      cout << "  H:                   Hide/show GUI controls" "\n";
      cout << "  Up/Down Arrows:      Adjust samples per batch (1-256)" "\n";
      cout << "  Left/Right Arrows:   Adjust light intensity (0.1-3.0)" "\n";
      cout << "  ESC:                 Exit" "\n" "\n";
      cout << "Sample accumulation: " << samples_per_batch << " samples per batch, up to " << max_samples
           << " total samples" "\n";
      cout << "Auto-accumulation: " << (auto_accumulate ? "ON" : "OFF")<<  "\n";
   }

   // Event handling
   static bool pollEvent(SDL_Event &event) { return SDL_PollEvent(&event); }

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

   static void cleanupSDL()
   {
#ifdef SDL2_TTF_FOUND
      TTF_Quit();
#endif
      SDL_Quit();
   }

   void loadLogo()
   {
      const auto relative_width = 0.3F; // Logo width relative to
      //  image width
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

   void loadFont()
   {
#ifdef SDL2_TTF_FOUND
      font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 16);
      if (font == nullptr)
      {
         font = TTF_OpenFont("/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf", 16);
      }
      if (font == nullptr)
      {
         cerr << "Warning: Could not load font: " << TTF_GetError()<<  "\n";
      }
#endif
   }

#ifdef SDL2_TTF_FOUND
   // Helper function to render text to screen
   void renderTextToScreen(TTF_Font *ttf_font, const char *text, SDL_Color color, int x, int y)
   {
      SDL_Surface *text_surface = TTF_RenderText_Blended(ttf_font, text, color);
      if (text_surface != nullptr)
      {
         SDL_Texture *text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
         if (text_texture != nullptr)
         {
            SDL_Rect text_rect = {x, y, text_surface->w, text_surface->h};
            SDL_RenderCopy(renderer, text_texture, nullptr, &text_rect);
            SDL_DestroyTexture(text_texture);
         }
         SDL_FreeSurface(text_surface);
      }
   }

   // Helper function to draw semi-transparent panel background
   void drawPanelBackground(int x, int y, int w, int h)
   {
      SDL_Rect bg_rect = {x - 5, y - 5, w + 10, h + 10};
      SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
      SDL_SetRenderDrawColor(renderer, 0, 0, 0, 200);
      SDL_RenderFillRect(renderer, &bg_rect);
   }

   // Helper function to draw a generic slider with customizable appearance
   void drawGenericSlider(TTF_Font *ttf_font, int padding, int y, int control_width, const char *label_format,
                          float value, float min_val, float max_val, SDL_Color fill_color,
                          SliderBounds &slider_bounds, int label_width)
   {
      char label[64];
      snprintf(label, sizeof(label), label_format, value);

      SDL_Color white = {255, 255, 255, 255};
      renderTextToScreen(ttf_font, label, white, padding, y);

      int slider_x = padding + label_width;
      int slider_w = control_width - label_width;
      SDL_Rect slider_bg = {slider_x, y + 8, slider_w, 8};
      SDL_SetRenderDrawColor(renderer, 60, 60, 60, 255);
      SDL_RenderFillRect(renderer, &slider_bg);

      slider_bounds.x = slider_x;
      slider_bounds.y = y + 8;
      slider_bounds.width = slider_w;
      slider_bounds.height = 8;
      slider_bounds.min_val = min_val;
      slider_bounds.max_val = max_val;

      float ratio = (value - min_val) / (max_val - min_val);
      int fill_w = static_cast<int>(slider_w * ratio);
      SDL_Rect slider_fill = {slider_x, y + 8, fill_w, 8};
      SDL_SetRenderDrawColor(renderer, fill_color.r, fill_color.g, fill_color.b, fill_color.a);
      SDL_RenderFillRect(renderer, &slider_fill);

      int handle_x = slider_x + fill_w - 3;
      SDL_Rect handle = {handle_x, y + 4, 6, 16};
      SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
      SDL_RenderFillRect(renderer, &handle);
   }

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

      renderTextToScreen(ttf_font, label, white, x + box_size + 8, y + 5);

      button_rect.x = x;
      button_rect.y = box_y;
      button_rect.w = std::min(max_width, box_size + 8 + 80);
      button_rect.h = box_size;
   }



   void drawSamplesSlider(TTF_Font *ttf_font, int padding, int y, int control_width, int samples_per_batch,
                          SliderBounds &samples_slider, int label_width)
   {
      SDL_Color color = {100, 150, 255, 255};
      drawGenericSlider(ttf_font, padding, y, control_width, "Samples: %d", static_cast<float>(samples_per_batch), 1.0f, 256.0f, color, samples_slider, label_width);
   }

   void drawLightSlider(TTF_Font *ttf_font, int padding, int y, int control_width, float light_intensity,
                        SliderBounds &intensity_slider, int label_width)
   {
      SDL_Color color = {255, 200, 100, 255};
      drawGenericSlider(ttf_font, padding, y, control_width, "Light: %.1f", light_intensity, 0.1f, 3.0f, color, intensity_slider, label_width);
   }

   void drawBackgroundSlider(TTF_Font *ttf_font, int padding, int y, int control_width, float background_intensity,
                             SliderBounds &background_slider, int label_width)
   {
      SDL_Color color = {150, 100, 255, 255};
      drawGenericSlider(ttf_font, padding, y, control_width, "Background: %.2f", background_intensity, 0.0f, 3.0f, color, background_slider, label_width);
   }

   void drawFuzzinessSlider(TTF_Font *ttf_font, int padding, int y, int control_width, float metal_fuzziness,
                            SliderBounds &fuzziness_slider, int label_width)
   {
      SDL_Color color = {200, 200, 100, 255};
      drawGenericSlider(ttf_font, padding, y, control_width, "Metal fuzz: %.2f", metal_fuzziness, 0.0f, 5.0f, color, fuzziness_slider, label_width);
   }

   void drawGlassIORSlider(TTF_Font *ttf_font, int padding, int y, int control_width, float glass_ior,
                           SliderBounds &glass_ior_slider, int label_width)
   {
      SDL_Color color = {100, 200, 255, 255};
      drawGenericSlider(ttf_font, padding, y, control_width, "Glass IOR: %.2f", glass_ior, 1.0f, 2.5f, color, glass_ior_slider, label_width);
   }

   void drawDOFApertureSlider(TTF_Font *ttf_font, int padding, int y, int control_width, float dof_aperture,
                              SliderBounds &aperture_slider, int label_width)
   {
      SDL_Color color = {100, 200, 255, 255};
      drawGenericSlider(ttf_font, padding, y, control_width, "DOF Aperture: %.2f", dof_aperture, 0.0f, 1.0f, color, aperture_slider, label_width);
   }

   void drawDOFFocusSlider(TTF_Font *ttf_font, int padding, int y, int control_width, float dof_focus_distance,
                           SliderBounds &focus_slider, int label_width)
   {
      SDL_Color color = {255, 150, 100, 255};
      drawGenericSlider(ttf_font, padding, y, control_width, "DOF Focus: %.1f", dof_focus_distance, 1.0f, 50.0f, color, focus_slider, label_width);
   }
#endif
};

#endif // SDL2_FOUND
