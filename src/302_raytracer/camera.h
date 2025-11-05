/**
 * @class Camera
 * @brief Represents a camera in a ray tracing system, capable of rendering scenes using various methods.
 *
 * The Camera class provides functionality to render scenes using ray tracing. It supports multiple rendering
 * methods, including CPU-based, CUDA-based, and real-time rendering with SDL2. The camera can be configured
 * with parameters such as field of view, resolution, and sampling for anti-aliasing.
 */

#include "camera_cuda.h"
#include "constants.h"
#include "utils.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#ifdef SDL2_FOUND
#include <SDL.h>
#ifdef SDL2_TTF_FOUND
#include <SDL_ttf.h>
#endif
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "../external/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../external/stb_image_resize2.h"
#endif

#pragma once

// Structure to hold slider interaction bounds
struct SliderBounds {
   int x, y, width, height;
   float min_val, max_val;
   float* value_ptr;
};

class Camera
{

 public:
   // Image parameters
   int image_width;
   int image_height;
   int image_channels; // Number of color channels per pixel (e.g., 3 for RGB)

   // Camera
   double vfov = 35.0;                   // Vertical field of view in degrees
   Point3 lookfrom = Point3(-2, 2, 5);   // Point camera is looking from
   Point3 lookat = Point3(-2, -0.5, -1); // Point camera is looking at
   Vec3 vup = Vec3(0, 1, 0);             // Camera-relative "up" direction

   // Ray tracing
   std::atomic<long long> n_rays{0};           // Number of rays traced so far with this cam (thread-safe)
   int samples_per_pixel;                      // Number of samples per pixel for anti-aliasing
   const int max_depth = constants::MAX_DEPTH; // Maximum ray bounce depth

   Camera(const Point3 &center, const int image_width, const int image_height, const int image_channels,
          int samples_per_pixel = 1)
       : image_width(image_width), image_height(image_height), image_channels(image_channels),
         samples_per_pixel(samples_per_pixel), camera_center(center)
   {
      initialize();
   }

   Camera() : Camera(Vec3(0, 0, 0), 720, 3, 1) {}

   /**
    * @brief Renders the entire image sequentially pixel by pixel using ray tracing
    *
    * This method performs sequential rendering by iterating through each pixel in the image
    * from top-left to bottom-right.
    *
    * @param scene The hittable scene object containing all geometry to render
    * @param image Vector buffer to store the rendered RGB pixel data (modified in-place)
    */
   void renderPixels(const Hittable &scene, vector<unsigned char> &image)
   {
      auto start_time = std::chrono::high_resolution_clock::now();

      // Render each pixel in the image sequentially
      for (int y = 0; y < image_height; ++y)
      {
         for (int x = 0; x < image_width; ++x)
         {
            // Compute the color for this pixel using ray tracing with anti-aliasing
            Color pixel_color = computePixelColor(scene, x, y);

            // Store the computed color in the image buffer
            setPixel(image, x, y, pixel_color);
         }

         // Show progress after completing each row
         showProgress(y, image_height);
      }

      showProgress(image_height - 1, image_height);

      auto end_time = std::chrono::high_resolution_clock::now();

      cout << endl;
      cout << "CPU single thread rendering completed in " << timeStr(end_time - start_time) << endl;
   }

   /**
    * @brief Renders the entire image using parallel processing for improved performance
    *
    * This method divides the image into horizontal chunks and processes them concurrently
    * using multiple threads. Each thread renders a subset of rows independently, with
    * thread-safe progress tracking using a mutex. The workload is distributed across
    * a fixed number of threads with the last thread handling any remaining rows
    * if the image height doesn't divide evenly.
    *
    * @param scene The hittable scene object containing all geometry to render
    * @param image Vector buffer to store the rendered RGB pixel data (modified in-place)
    *
    */
   void renderPixelsParallel(const Hittable &scene, vector<unsigned char> &image)
   {
      const int num_threads = std::thread::hardware_concurrency(); // Get the number of available hardware threads
      std::vector<std::thread> threads(num_threads);
      std::mutex progress_mutex;
      int completed_rows = 0; // Track globally completed rows

      auto start_time = std::chrono::high_resolution_clock::now();

      auto render_chunk = [&](int start_y, int end_y)
      {
         for (int y = start_y; y < end_y; ++y)
         {
            for (int x = 0; x < image_width; ++x)
            {
               // Compute the color for this pixel using the shared helper method
               Color pixel_color = computePixelColor(scene, x, y);

               // Store the computed color in the image buffer
               setPixel(image, x, y, pixel_color);
            }

            // Update progress - increment global counter and show progress
            {
               std::lock_guard<std::mutex> lock(progress_mutex);
               completed_rows++;
               showProgress(completed_rows - 1, image_height);
            }
         }
      };

      int chunk_size = image_height / num_threads;
      for (int t = 0; t < num_threads; ++t)
      {
         int start_y = t * chunk_size;
         int end_y = (t == num_threads - 1) ? image_height : start_y + chunk_size;
         threads[t] = std::thread(render_chunk, start_y, end_y);
      }

      for (auto &thread : threads)
      {
         thread.join();
      }

      showProgress(image_height - 1, image_height);

      auto end_time = std::chrono::high_resolution_clock::now();

      cout << endl;
      cout << "Parallel rendering (using " << num_threads << " threads) completed in " << timeStr(end_time - start_time)
           << endl;
   }

   /**
    * @brief Renders the image using CUDA for parallel processing.
    *
    * This method leverages CUDA to perform ray tracing computations on the GPU,
    * significantly accelerating the rendering process. It calculates the color
    * of each pixel in the image buffer and updates the ray count. The method
    * also measures and displays the time taken for the rendering process.
    *
    * @param image A vector of unsigned char representing the image buffer where
    *              the rendered pixel data will be stored. The buffer must be
    *              pre-allocated with a size of (image_width * image_height * image_channels).
    */
   void renderPixelsCUDA(vector<unsigned char> &image)
   {
      auto start_time = std::chrono::high_resolution_clock::now();
      printf("CUDA renderer starting: %dx%d, %d samples, max_depth=%d\n", image_width, image_height, samples_per_pixel,
             max_depth);

      // Call CUDA rendering function with expanded parameters and get ray count back
      unsigned long long cuda_ray_count =
          ::renderPixelsCUDA(image.data(), image_width, image_height, camera_center.x(), camera_center.y(),
                             camera_center.z(), pixel00_loc.x(), pixel00_loc.y(), pixel00_loc.z(), pixel_delta_u.x(),
                             pixel_delta_u.y(), pixel_delta_u.z(), pixel_delta_v.x(), pixel_delta_v.y(),
                             pixel_delta_v.z(), samples_per_pixel, max_depth);

      // Add the CUDA ray count to our atomic counter
      n_rays.fetch_add(cuda_ray_count, std::memory_order_relaxed);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = end_time - start_time;
      cout << "CUDA rendering completed in " << timeStr(duration) << endl;
   }

#ifdef SDL2_FOUND
   /**
    * @brief Render image in tiles with event checking between tiles for responsiveness
    * 
    * Breaks the image into horizontal strips (tiles) and renders each tile separately.
    * Checks for SDL events between tiles, allowing camera movement to interrupt rendering.
    * 
    * @param image Output buffer for the rendered image
    * @param tile_height Height of each tile in pixels (smaller = more responsive, but more overhead)
    * @param renderer SDL renderer for displaying progress
    * @param texture SDL texture to update
    * @param running Reference to running flag (set to false on quit events)
    * @param camera_changed Reference to camera_changed flag (set to true on camera movement)
    * @return true if rendering completed, false if interrupted by camera movement
    */
   /**
    * @brief Render image in rectangular block tiles with event checking between tiles for responsiveness
    * 
    * Breaks the image into a grid of rectangular tiles and renders each block separately.
    * Checks for SDL events between tiles, allowing camera movement to interrupt rendering.
    * 
    * @param image Output buffer for the rendered image
    * @param tiles_x Number of tiles horizontally
    * @param tiles_y Number of tiles vertically
    * @param renderer SDL renderer for displaying progress
    * @param texture SDL texture to update
    * @param logo_texture Logo texture to overlay (can be nullptr)
    * @param logo_rect Rectangle for logo positioning
    * @param running Reference to running flag (set to false on quit events)
    * @param camera_changed Reference to camera_changed flag (set to true on camera movement)
    * @return true if rendering completed, false if interrupted by camera movement
    */
   bool renderTiled(vector<unsigned char> &image, int tiles_x, int tiles_y, SDL_Renderer *renderer, 
                    SDL_Texture *texture, SDL_Texture *logo_texture, const SDL_Rect &logo_rect,
                    void *font, bool &running, bool &camera_changed)
   {
      int tile_width = (image_width + tiles_x - 1) / tiles_x;
      int tile_height = (image_height + tiles_y - 1) / tiles_y;
      
      // Render tiles in a grid pattern
      for (int tile_y = 0; tile_y < tiles_y; tile_y++)
      {
         for (int tile_x = 0; tile_x < tiles_x; tile_x++)
         {
            // Check for events before rendering each tile
            SDL_Event event;
            while (SDL_PollEvent(&event))
            {
               if (event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE))
               {
                  running = false;
                  return false;
               }
               
               // Handle camera movement events
               if (event.type == SDL_MOUSEBUTTONDOWN)
               {
                  if (event.button.button == SDL_BUTTON_LEFT)
                  {
                     left_button_down = true;
                     last_mouse_x = event.button.x;
                     last_mouse_y = event.button.y;
                  }
                  else if (event.button.button == SDL_BUTTON_RIGHT)
                  {
                     right_button_down = true;
                     last_mouse_x = event.button.x;
                     last_mouse_y = event.button.y;
                  }
               }
               else if (event.type == SDL_MOUSEBUTTONUP)
               {
                  if (event.button.button == SDL_BUTTON_LEFT) left_button_down = false;
                  if (event.button.button == SDL_BUTTON_RIGHT) right_button_down = false;
               }
               else if (event.type == SDL_MOUSEMOTION)
               {
                  int mouse_x = event.motion.x;
                  int mouse_y = event.motion.y;
                  int dx = mouse_x - last_mouse_x;
                  int dy = mouse_y - last_mouse_y;

                  if (left_button_down)
                  {
                     camera_azimuth += dx * 0.01;
                     camera_elevation -= dy * 0.01;
                     
                     const double max_elevation = 1.5;
                     if (camera_elevation > max_elevation) camera_elevation = max_elevation;
                     if (camera_elevation < -max_elevation) camera_elevation = -max_elevation;
                     
                     double cos_elev = cos(camera_elevation);
                     lookfrom.e[0] = lookat.x() + camera_distance * cos_elev * sin(camera_azimuth);
                     lookfrom.e[1] = lookat.y() + camera_distance * sin(camera_elevation);
                     lookfrom.e[2] = lookat.z() + camera_distance * cos_elev * cos(camera_azimuth);
                     
                     camera_changed = true;
                     return false; // Interrupt rendering
                  }
                  else if (right_button_down)
                  {
                     Vec3 right = unit_vector(cross(lookfrom - lookat, vup));
                     Vec3 up = unit_vector(cross(right, lookfrom - lookat));
                     
                     double pan_speed = camera_distance * 0.001;
                     Vec3 pan_offset = right * (-dx * pan_speed) + up * (dy * pan_speed);
                     
                     lookfrom = lookfrom + pan_offset;
                     lookat = lookat + pan_offset;
                     
                     camera_changed = true;
                     return false; // Interrupt rendering
                  }

                  last_mouse_x = mouse_x;
                  last_mouse_y = mouse_y;
               }
               else if (event.type == SDL_MOUSEWHEEL)
               {
                  double zoom_speed = camera_distance * 0.1;
                  camera_distance -= event.wheel.y * zoom_speed;
                  if (camera_distance < 0.5) camera_distance = 0.5;
                  if (camera_distance > 50.0) camera_distance = 50.0;
                  
                  double cos_elev = cos(camera_elevation);
                  lookfrom.e[0] = lookat.x() + camera_distance * cos_elev * sin(camera_azimuth);
                  lookfrom.e[1] = lookat.y() + camera_distance * sin(camera_elevation);
                  lookfrom.e[2] = lookat.z() + camera_distance * cos_elev * cos(camera_azimuth);
                  
                  camera_changed = true;
                  return false; // Interrupt rendering
               }
            }
            
            // Calculate tile boundaries
            int x_start = tile_x * tile_width;
            int y_start = tile_y * tile_height;
            int x_end = std::min(x_start + tile_width, image_width);
            int y_end = std::min(y_start + tile_height, image_height);
            int actual_tile_width = x_end - x_start;
            int actual_tile_height = y_end - y_start;
            
            // Create a temporary buffer for this tile
            vector<unsigned char> tile_buffer(actual_tile_width * actual_tile_height * image_channels);
            
            // Adjust pixel00_loc for this tile (move right by x_start, down by y_start)
            Point3 tile_pixel00 = pixel00_loc + pixel_delta_u * x_start + pixel_delta_v * y_start;
            
            // Render this tile
            unsigned long long tile_ray_count = ::renderPixelsCUDA(
               tile_buffer.data(), actual_tile_width, actual_tile_height,
               camera_center.x(), camera_center.y(), camera_center.z(),
               tile_pixel00.x(), tile_pixel00.y(), tile_pixel00.z(),
               pixel_delta_u.x(), pixel_delta_u.y(), pixel_delta_u.z(),
               pixel_delta_v.x(), pixel_delta_v.y(), pixel_delta_v.z(),
               samples_per_pixel, max_depth
            );
            
            n_rays.fetch_add(tile_ray_count, std::memory_order_relaxed);
            
            // Copy tile into main image buffer
            for (int y = 0; y < actual_tile_height; y++)
            {
               int src_offset = y * actual_tile_width * image_channels;
               int dst_offset = (y_start + y) * image_width * image_channels + x_start * image_channels;
               memcpy(&image[dst_offset], &tile_buffer[src_offset], actual_tile_width * image_channels);
            }
            
            // Update SDL display immediately after each tile
            SDL_UpdateTexture(texture, nullptr, image.data(), image_width * image_channels);
            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, texture, nullptr, nullptr);
            
            // Overlay logo if available
            if (logo_texture)
            {
               SDL_RenderCopy(renderer, logo_texture, nullptr, &logo_rect);
            }
            
            // Draw sample count text in upper-right corner
            drawSampleCountText(renderer, font, samples_per_pixel);
            
            SDL_RenderPresent(renderer);
         }
      }
      
      return true; // Rendering completed successfully
   }

   /**
    * @brief Draws the current sample count text in the upper-right corner using TTF
    */
   void drawSampleCountText(SDL_Renderer* renderer, void* font_ptr, int sample_count)
   {
#ifdef SDL2_TTF_FOUND
      TTF_Font* font = static_cast<TTF_Font*>(font_ptr);
      if (!font) return; // Skip if font not loaded
      
      // Create text string
      std::string text = std::to_string(sample_count) + " SPP";
      
      // Render text to surface with white color
      SDL_Color white = {255, 255, 255, 255};
      SDL_Surface* text_surface = TTF_RenderText_Blended(font, text.c_str(), white);
      if (!text_surface) return;
      
      // Create texture from surface
      SDL_Texture* text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
      
      if (text_texture)
      {
         // Calculate position (upper-right corner with padding)
         int text_width = text_surface->w;
         int text_height = text_surface->h;
         int padding = 15;
         
         // Fixed box width to accommodate "4096 SPP" (approximately 90 pixels at size 16)
         int box_width = 90;
         
         // Draw semi-transparent background box with fixed width
         SDL_Rect bg_rect = {
            image_width - box_width - padding,
            padding - 5,
            box_width,
            text_height + 10
         };
         SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
         SDL_SetRenderDrawColor(renderer, 0, 0, 0, 200); // Semi-transparent black
         SDL_RenderFillRect(renderer, &bg_rect);
         
         // Center text within the fixed-width box
         SDL_Rect text_rect = {
            image_width - padding - box_width + (box_width - text_width) / 2,
            padding,
            text_width,
            text_height
         };
         SDL_RenderCopy(renderer, text_texture, nullptr, &text_rect);
         
         SDL_DestroyTexture(text_texture);
      }
      
      SDL_FreeSurface(text_surface);
#else
      // Fallback: do nothing if SDL_ttf is not available
      (void)renderer;
      (void)font_ptr;
      (void)sample_count;
#endif
   }

   void drawUIControls(SDL_Renderer* renderer, void* font_ptr, float gamma, float light_intensity, float background_intensity, bool accumulation_enabled, 
                       SliderBounds* gamma_slider, SliderBounds* intensity_slider, SliderBounds* background_slider, SDL_Rect* toggle_button_rect)
   {
#ifdef SDL2_TTF_FOUND
      TTF_Font* font = static_cast<TTF_Font*>(font_ptr);
      if (!font) return; // Skip if font not loaded
      
      // Use smaller font size for UI controls
      TTF_Font* small_font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12);
      if (!small_font) small_font = font; // Fallback to regular font
      
      int padding = 15;
      int control_width = 220;
      int slider_height = 25;
      int spacing = 8;
      int start_y = image_height - (4 * slider_height + 3 * spacing + padding);
      
      SDL_Color white = {255, 255, 255, 255};
      SDL_Color green = {0, 255, 0, 255};
      SDL_Color red = {255, 0, 0, 255};
      
      // Draw semi-transparent background
      SDL_Rect bg_rect = {
         padding - 5,
         start_y - 5,
         control_width + 10,
         4 * slider_height + 3 * spacing + 10
      };
      SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
      SDL_SetRenderDrawColor(renderer, 0, 0, 0, 200);
      SDL_RenderFillRect(renderer, &bg_rect);
      
      // Toggle button with checkbox
      {
         // Draw checkbox box
         int box_size = 14;
         int box_y = start_y + 5;
         SDL_Rect checkbox_bg = {padding, box_y, box_size, box_size};
         
         // Draw box border
         SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
         SDL_RenderDrawRect(renderer, &checkbox_bg);
         
         // Fill box based on state
         SDL_Rect checkbox_fill = {padding + 2, box_y + 2, box_size - 4, box_size - 4};
         if (accumulation_enabled) {
            SDL_SetRenderDrawColor(renderer, 0, 200, 0, 255); // Green when ON
         } else {
            SDL_SetRenderDrawColor(renderer, 200, 0, 0, 255); // Red when OFF
         }
         SDL_RenderFillRect(renderer, &checkbox_fill);
         
         // Draw label text (always white)
         std::string text = "Auto-Accum";
         SDL_Surface* text_surface = TTF_RenderText_Blended(small_font, text.c_str(), white);
         if (text_surface)
         {
            SDL_Texture* text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
            if (text_texture)
            {
               SDL_Rect text_rect = {padding + box_size + 8, start_y + 5, text_surface->w, text_surface->h};
               SDL_RenderCopy(renderer, text_texture, nullptr, &text_rect);
               SDL_DestroyTexture(text_texture);
            }
            SDL_FreeSurface(text_surface);
         }
         
         // Store toggle button bounds for click detection (include both box and text)
         if (toggle_button_rect) {
            toggle_button_rect->x = padding;
            toggle_button_rect->y = box_y;
            toggle_button_rect->w = box_size + 8 + 80; // box + spacing + text width estimate
            toggle_button_rect->h = box_size;
         }
      }
      
      // Gamma slider
      {
         int y = start_y + slider_height + spacing;
         
         // Label with value
         char label[32];
         snprintf(label, sizeof(label), "Gamma: %.1f", gamma);
         SDL_Surface* text_surface = TTF_RenderText_Blended(small_font, label, white);
         if (text_surface)
         {
            SDL_Texture* text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
            if (text_texture)
            {
               SDL_Rect text_rect = {padding, y + 5, text_surface->w, text_surface->h};
               SDL_RenderCopy(renderer, text_texture, nullptr, &text_rect);
               SDL_DestroyTexture(text_texture);
            }
            SDL_FreeSurface(text_surface);
         }
         
         // Slider bar background
         int slider_x = padding + 80;
         int slider_w = control_width - 80;
         SDL_Rect slider_bg = {slider_x, y + 8, slider_w, 8};
         SDL_SetRenderDrawColor(renderer, 60, 60, 60, 255);
         SDL_RenderFillRect(renderer, &slider_bg);
         
         // Store slider bounds for mouse interaction
         if (gamma_slider) {
            gamma_slider->x = slider_x;
            gamma_slider->y = y + 8;
            gamma_slider->width = slider_w;
            gamma_slider->height = 8;
            gamma_slider->min_val = 0.5f;
            gamma_slider->max_val = 3.0f;
         }
         
         // Slider fill (gamma range: 0.5 - 3.0)
         float gamma_ratio = (gamma - 0.5f) / (3.0f - 0.5f);
         int fill_w = static_cast<int>(slider_w * gamma_ratio);
         SDL_Rect slider_fill = {slider_x, y + 8, fill_w, 8};
         SDL_SetRenderDrawColor(renderer, 100, 150, 255, 255);
         SDL_RenderFillRect(renderer, &slider_fill);
         
         // Slider handle
         int handle_x = slider_x + fill_w - 3;
         SDL_Rect handle = {handle_x, y + 4, 6, 16};
         SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
         SDL_RenderFillRect(renderer, &handle);
      }
      
      // Light intensity slider
      {
         int y = start_y + 2 * slider_height + 2 * spacing;
         
         // Label with value
         char label[32];
         snprintf(label, sizeof(label), "Light: %.1f", light_intensity);
         SDL_Surface* text_surface = TTF_RenderText_Blended(small_font, label, white);
         if (text_surface)
         {
            SDL_Texture* text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
            if (text_texture)
            {
               SDL_Rect text_rect = {padding, y + 5, text_surface->w, text_surface->h};
               SDL_RenderCopy(renderer, text_texture, nullptr, &text_rect);
               SDL_DestroyTexture(text_texture);
            }
            SDL_FreeSurface(text_surface);
         }
         
         // Slider bar background
         int slider_x = padding + 80;
         int slider_w = control_width - 80;
         SDL_Rect slider_bg = {slider_x, y + 8, slider_w, 8};
         SDL_SetRenderDrawColor(renderer, 60, 60, 60, 255);
         SDL_RenderFillRect(renderer, &slider_bg);
         
         // Store slider bounds for mouse interaction
         if (intensity_slider) {
            intensity_slider->x = slider_x;
            intensity_slider->y = y + 8;
            intensity_slider->width = slider_w;
            intensity_slider->height = 8;
            intensity_slider->min_val = 0.1f;
            intensity_slider->max_val = 3.0f;
         }
         
         // Slider fill (intensity range: 0.1 - 3.0)
         float intensity_ratio = (light_intensity - 0.1f) / (3.0f - 0.1f);
         int fill_w = static_cast<int>(slider_w * intensity_ratio);
         SDL_Rect slider_fill = {slider_x, y + 8, fill_w, 8};
         SDL_SetRenderDrawColor(renderer, 255, 200, 100, 255);
         SDL_RenderFillRect(renderer, &slider_fill);
         
         // Slider handle
         int handle_x = slider_x + fill_w - 3;
         SDL_Rect handle = {handle_x, y + 4, 6, 16};
         SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
         SDL_RenderFillRect(renderer, &handle);
      }
      
      // Background intensity slider
      {
         int y = start_y + 3 * slider_height + 3 * spacing;
         
         // Label with value
         char label[32];
         snprintf(label, sizeof(label), "Background: %.2f", background_intensity);
         SDL_Surface* text_surface = TTF_RenderText_Blended(small_font, label, white);
         if (text_surface)
         {
            SDL_Texture* text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
            if (text_texture)
            {
               SDL_Rect text_rect = {padding, y + 5, text_surface->w, text_surface->h};
               SDL_RenderCopy(renderer, text_texture, nullptr, &text_rect);
               SDL_DestroyTexture(text_texture);
            }
            SDL_FreeSurface(text_surface);
         }
         
         // Slider bar background
         int slider_x = padding + 100;
         int slider_w = control_width - 100;
         SDL_Rect slider_bg = {slider_x, y + 8, slider_w, 8};
         SDL_SetRenderDrawColor(renderer, 60, 60, 60, 255);
         SDL_RenderFillRect(renderer, &slider_bg);
         
         // Store slider bounds for mouse interaction
         if (background_slider) {
            background_slider->x = slider_x;
            background_slider->y = y + 8;
            background_slider->width = slider_w;
            background_slider->height = 8;
            background_slider->min_val = 0.0f;
            background_slider->max_val = 3.0f;
         }
         
         // Slider fill (background range: 0.0 - 3.0)
         float bg_ratio = (background_intensity - 0.0f) / (3.0f - 0.0f);
         int fill_w = static_cast<int>(slider_w * bg_ratio);
         SDL_Rect slider_fill = {slider_x, y + 8, fill_w, 8};
         SDL_SetRenderDrawColor(renderer, 150, 100, 255, 255);
         SDL_RenderFillRect(renderer, &slider_fill);
         
         // Slider handle
         int handle_x = slider_x + fill_w - 3;
         SDL_Rect handle = {handle_x, y + 4, 6, 16};
         SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
         SDL_RenderFillRect(renderer, &handle);
      }
      
      // Close small font if we opened it
      if (small_font != font) {
         TTF_CloseFont(small_font);
      }
#else
      // Fallback: do nothing if SDL_ttf is not available
      (void)renderer;
      (void)font_ptr;
      (void)gamma;
      (void)light_intensity;
      (void)background_intensity;
      (void)accumulation_enabled;
      (void)gamma_slider;
      (void)intensity_slider;
      (void)background_slider;
      (void)toggle_button_rect;
#endif
   }

   /**
    * @brief Interactive SDL rendering with continuous sample accumulation
    * 
    * This method displays the ray traced image in an SDL window while continuously
    * accumulating samples for progressive quality improvement. Uses true CUDA-based
    * sample accumulation for optimal performance and smooth visual progression.
    * 
    * Interactive camera controls:
    * - Left mouse button: Rotate camera (orbit around lookat point)
    * - Right mouse button: Pan camera (move lookat point)
    * - Mouse wheel: Zoom in/out (change camera distance)
    * - Spacebar: Toggle automatic sample accumulation
    * - Up/Down arrows: Adjust gamma correction (0.5 - 3.0)
    * - Left/Right arrows: Adjust light intensity (0.1 - 3.0)
    * - Any camera movement automatically restarts rendering from scratch
    * - ESC/Close window: Exit
    *
    * The renderer starts with samples_per_batch samples for fast preview,
    * then continuously adds samples_per_batch samples until reaching max_samples.
    * Each batch is rendered on the GPU and accumulated into a floating-point buffer,
    * providing smooth quality improvement without discrete quality jumps.
    * 
    * @param image The final image buffer to store the render
    * @param max_samples Maximum total samples to accumulate (default: 1024)
    * @param samples_per_batch Number of samples to add per batch (default: 8)
    * @param auto_accumulate Enable automatic sample accumulation (default: true)
    */
   void renderPixelsSDLContinuous(vector<unsigned char> &image, int max_samples = 2048, int samples_per_batch = 8, bool auto_accumulate = true)
   {
      // Initialize SDL
      if (SDL_Init(SDL_INIT_VIDEO) < 0)
      {
         cerr << "SDL initialization failed: " << SDL_GetError() << endl;
         return;
      }
      
#ifdef SDL2_TTF_FOUND
      // Initialize SDL_ttf for text rendering
      if (TTF_Init() < 0)
      {
         cerr << "SDL_ttf initialization failed: " << TTF_GetError() << endl;
         SDL_Quit();
         return;
      }
#endif

      // Create window
      SDL_Window *window = SDL_CreateWindow("ISC - 302 ray tracer (mui) / Continuous mode (LMB:Rotate RMB:Pan Wheel:Zoom)", 
                                             SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
                                             image_width, image_height, SDL_WINDOW_SHOWN);
      if (!window)
      {
         cerr << "Window creation failed: " << SDL_GetError() << endl;
#ifdef SDL2_TTF_FOUND
         TTF_Quit();
#endif
         SDL_Quit();
         return;
      }

      // Create renderer
      SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
      if (!renderer)
      {
         cerr << "Renderer creation failed: " << SDL_GetError() << endl;
         SDL_DestroyWindow(window);
         SDL_Quit();
         return;
      }

      // Create texture for displaying the image
      SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING,
                                                image_width, image_height);
      if (!texture)
      {
         cerr << "Texture creation failed: " << SDL_GetError() << endl;
         SDL_DestroyRenderer(renderer);
         SDL_DestroyWindow(window);
         SDL_Quit();
         return;
      }

      // Load ISC logo using stb_image
      int logo_img_width, logo_img_height, logo_img_channels;
      unsigned char* logo_data = stbi_load("../res/ISC Logo inline white v3 - 1500px.png", 
                                            &logo_img_width, &logo_img_height, &logo_img_channels, 4); // Force RGBA
      
      SDL_Texture* logo_texture = nullptr;
      SDL_Rect logo_rect;
      
      if (logo_data)
      {
         // Calculate logo dimensions (1/5 of window width, maintain aspect ratio)
         int target_logo_width = image_width / 5;
         int target_logo_height = (logo_img_height * target_logo_width) / logo_img_width;
         
         // Resize logo using high-quality Mitchell filter
         unsigned char* resized_logo = new unsigned char[target_logo_width * target_logo_height * 4];
         stbir_resize_uint8_srgb(logo_data, logo_img_width, logo_img_height, 0,
                                  resized_logo, target_logo_width, target_logo_height, 0,
                                  STBIR_RGBA);
         
         // Create SDL surface from resized logo data
         SDL_Surface* logo_surface = SDL_CreateRGBSurfaceFrom(
            resized_logo, target_logo_width, target_logo_height, 32, target_logo_width * 4,
            0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);
         
         if (logo_surface)
         {
            logo_texture = SDL_CreateTextureFromSurface(renderer, logo_surface);
            SDL_SetTextureBlendMode(logo_texture, SDL_BLENDMODE_BLEND);
            
            // Position logo in bottom-right corner with 10px padding
            logo_rect.w = target_logo_width;
            logo_rect.h = target_logo_height;
            logo_rect.x = image_width - target_logo_width - 10;
            logo_rect.y = image_height - target_logo_height - 10;
            
            SDL_FreeSurface(logo_surface);
         }
         
         delete[] resized_logo;
         stbi_image_free(logo_data);
      }
      
      // Load font for text rendering (using system mono font or DejaVu Sans Mono)
      void* font = nullptr;
#ifdef SDL2_TTF_FOUND
      font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 16);
      if (!font) {
         // Try alternative font path
         font = TTF_OpenFont("/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf", 16);
      }
      if (!font) {
         cerr << "Warning: Could not load font: " << TTF_GetError() << endl;
      }
#endif

      cout << "\n=== Interactive Ray Tracing with Real-time Display ===" << endl;
      cout << "Controls:" << endl;
      cout << "  Left Mouse Button:   Rotate camera (orbit)" << endl;
      cout << "  Right Mouse Button:  Pan camera" << endl;
      cout << "  Mouse Wheel:         Zoom in/out" << endl;
      cout << "  SPACEBAR:            Toggle automatic accumulation" << endl;
      cout << "  Up/Down Arrows:      Adjust gamma (0.5-3.0)" << endl;
      cout << "  Left/Right Arrows:   Adjust light intensity (0.1-3.0)" << endl;
      cout << "  ESC:                 Exit" << endl;
      cout << "Sample accumulation: " << samples_per_batch << " samples per batch, up to " << max_samples << " total samples" << endl;
      cout << "Auto-accumulation: " << (auto_accumulate ? "ON" : "OFF") << endl;
      cout << "Using CUDA GPU acceleration with progressive refinement" << endl;
      
      bool running = true;
      bool camera_changed = true; // Trigger initial render
      bool accumulation_enabled = auto_accumulate; // Current state of auto-accumulation
      int current_samples = 0; // Current accumulated sample count
      float gamma = 2.0f; // Gamma correction value (default 2.0 for standard sRGB)
      float light_intensity = 1.0f; // Light intensity multiplier
      float background_intensity = 1.0f; // Background gradient brightness multiplier
      bool needs_rerender = false; // Flag to trigger re-render for gamma/intensity changes
      
      // Set initial light intensity and background in CUDA
      ::setLightIntensity(light_intensity);
      ::setBackgroundIntensity(background_intensity);
      
      // Slider bounds for mouse interaction
      SliderBounds gamma_slider_bounds = {0, 0, 0, 0, 0.5f, 3.0f, &gamma};
      SliderBounds intensity_slider_bounds = {0, 0, 0, 0, 0.1f, 3.0f, &light_intensity};
      SliderBounds background_slider_bounds = {0, 0, 0, 0, 0.0f, 3.0f, &background_intensity};
      SDL_Rect toggle_button_rect = {0, 0, 0, 0};
      bool dragging_slider = false;
      SliderBounds* active_slider = nullptr;
      
      SDL_Event event;
      vector<unsigned char> display_image(image_width * image_height * image_channels);
      vector<float> accum_buffer(image_width * image_height * image_channels, 0.0f);  // Accumulation buffer for CUDA
      void *d_rand_states = nullptr;  // Persistent device random states

      // Initialize camera control state
      left_button_down = false;
      right_button_down = false;
      last_mouse_x = 0;
      last_mouse_y = 0;
      
      // Initialize camera orbit parameters
      camera_distance = (lookfrom - lookat).length();
      camera_azimuth = 0.0;
      camera_elevation = 0.0;
      
      // Calculate initial angles from current camera position
      Vec3 to_camera = lookfrom - lookat;
      camera_distance = to_camera.length();
      camera_azimuth = atan2(to_camera.x(), to_camera.z());
      camera_elevation = asin(to_camera.y() / camera_distance);

      auto total_start = std::chrono::high_resolution_clock::now();

      // Main interaction loop
      while (running)
      {
         // Handle SDL events
         while (SDL_PollEvent(&event))
         {
            if (event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE))
            {
               running = false;
            }
            else if (event.type == SDL_KEYDOWN)
            {
               if (event.key.keysym.sym == SDLK_SPACE)
               {
                  // Toggle automatic accumulation
                  accumulation_enabled = !accumulation_enabled;
                  cout << "\n[Auto-accumulation " << (accumulation_enabled ? "ENABLED" : "DISABLED") << "]" << endl;
               }
               else if (event.key.keysym.sym == SDLK_UP)
               {
                  // Increase gamma
                  gamma = std::min(3.0f, gamma + 0.1f);
                  needs_rerender = true;
               }
               else if (event.key.keysym.sym == SDLK_DOWN)
               {
                  // Decrease gamma
                  gamma = std::max(0.5f, gamma - 0.1f);
                  needs_rerender = true;
               }
               else if (event.key.keysym.sym == SDLK_RIGHT)
               {
                  // Increase light intensity - restart render with new light
                  light_intensity = std::min(3.0f, light_intensity + 0.1f);
                  ::setLightIntensity(light_intensity);
                  camera_changed = true; // Restart rendering
               }
               else if (event.key.keysym.sym == SDLK_LEFT)
               {
                  // Decrease light intensity - restart render with new light
                  light_intensity = std::max(0.1f, light_intensity - 0.1f);
                  ::setLightIntensity(light_intensity);
                  camera_changed = true; // Restart rendering
               }
            }
            else if (event.type == SDL_MOUSEBUTTONDOWN)
            {
               if (event.button.button == SDL_BUTTON_LEFT)
               {
                  int mx = event.button.x;
                  int my = event.button.y;
                  
                  // Check if clicking on toggle button
                  if (mx >= toggle_button_rect.x && mx <= toggle_button_rect.x + toggle_button_rect.w &&
                      my >= toggle_button_rect.y && my <= toggle_button_rect.y + toggle_button_rect.h)
                  {
                     // Toggle accumulation
                     accumulation_enabled = !accumulation_enabled;
                     cout << "\n[Auto-accumulation " << (accumulation_enabled ? "ENABLED" : "DISABLED") << "]" << endl;
                  }
                  // Check if clicking on gamma slider
                  else if (mx >= gamma_slider_bounds.x && mx <= gamma_slider_bounds.x + gamma_slider_bounds.width &&
                      my >= gamma_slider_bounds.y - 5 && my <= gamma_slider_bounds.y + gamma_slider_bounds.height + 5)
                  {
                     dragging_slider = true;
                     active_slider = &gamma_slider_bounds;
                     
                     // Update value immediately
                     float ratio = (float)(mx - gamma_slider_bounds.x) / gamma_slider_bounds.width;
                     gamma = gamma_slider_bounds.min_val + ratio * (gamma_slider_bounds.max_val - gamma_slider_bounds.min_val);
                     gamma = std::max(gamma_slider_bounds.min_val, std::min(gamma_slider_bounds.max_val, gamma));
                     needs_rerender = true;
                  }
                  // Check if clicking on intensity slider
                  else if (mx >= intensity_slider_bounds.x && mx <= intensity_slider_bounds.x + intensity_slider_bounds.width &&
                           my >= intensity_slider_bounds.y - 5 && my <= intensity_slider_bounds.y + intensity_slider_bounds.height + 5)
                  {
                     dragging_slider = true;
                     active_slider = &intensity_slider_bounds;
                     
                     // Update value immediately and restart render
                     float ratio = (float)(mx - intensity_slider_bounds.x) / intensity_slider_bounds.width;
                     light_intensity = intensity_slider_bounds.min_val + ratio * (intensity_slider_bounds.max_val - intensity_slider_bounds.min_val);
                     light_intensity = std::max(intensity_slider_bounds.min_val, std::min(intensity_slider_bounds.max_val, light_intensity));
                     ::setLightIntensity(light_intensity);
                     camera_changed = true; // Restart rendering
                  }
                  // Check if clicking on background slider
                  else if (mx >= background_slider_bounds.x && mx <= background_slider_bounds.x + background_slider_bounds.width &&
                           my >= background_slider_bounds.y - 5 && my <= background_slider_bounds.y + background_slider_bounds.height + 5)
                  {
                     dragging_slider = true;
                     active_slider = &background_slider_bounds;
                     
                     // Update value immediately and restart render
                     float ratio = (float)(mx - background_slider_bounds.x) / background_slider_bounds.width;
                     background_intensity = background_slider_bounds.min_val + ratio * (background_slider_bounds.max_val - background_slider_bounds.min_val);
                     background_intensity = std::max(background_slider_bounds.min_val, std::min(background_slider_bounds.max_val, background_intensity));
                     ::setBackgroundIntensity(background_intensity);
                     camera_changed = true; // Restart rendering
                  }
                  else
                  {
                     // Normal camera rotation
                     left_button_down = true;
                     last_mouse_x = event.button.x;
                     last_mouse_y = event.button.y;
                  }
               }
               else if (event.button.button == SDL_BUTTON_RIGHT)
               {
                  right_button_down = true;
                  last_mouse_x = event.button.x;
                  last_mouse_y = event.button.y;
               }
            }
            else if (event.type == SDL_MOUSEBUTTONUP)
            {
               if (event.button.button == SDL_BUTTON_LEFT)
               {
                  left_button_down = false;
                  dragging_slider = false;
                  active_slider = nullptr;
               }
               else if (event.button.button == SDL_BUTTON_RIGHT)
               {
                  right_button_down = false;
               }
            }
            else if (event.type == SDL_MOUSEMOTION)
            {
               int mouse_x = event.motion.x;
               int mouse_y = event.motion.y;
               int dx = mouse_x - last_mouse_x;
               int dy = mouse_y - last_mouse_y;

               if (dragging_slider && active_slider)
               {
                  // Update slider value based on mouse position
                  float ratio = (float)(mouse_x - active_slider->x) / active_slider->width;
                  ratio = std::max(0.0f, std::min(1.0f, ratio));
                  float new_value = active_slider->min_val + ratio * (active_slider->max_val - active_slider->min_val);
                  
                  if (active_slider == &gamma_slider_bounds)
                  {
                     gamma = new_value;
                     needs_rerender = true;
                  }
                  else if (active_slider == &intensity_slider_bounds)
                  {
                     light_intensity = new_value;
                     ::setLightIntensity(light_intensity);
                     camera_changed = true; // Restart rendering with new light
                  }
                  else if (active_slider == &background_slider_bounds)
                  {
                     background_intensity = new_value;
                     ::setBackgroundIntensity(background_intensity);
                     camera_changed = true; // Restart rendering with new background
                  }
               }
               else if (left_button_down)
               {
                  // Rotate camera (orbit around lookat point)
                  camera_azimuth += dx * 0.01;
                  camera_elevation -= dy * 0.01;
                  
                  // Clamp elevation to avoid gimbal lock
                  const double max_elevation = 1.5;
                  if (camera_elevation > max_elevation) camera_elevation = max_elevation;
                  if (camera_elevation < -max_elevation) camera_elevation = -max_elevation;
                  
                  // Update camera position
                  lookfrom.e[0] = lookat.x() + camera_distance * cos(camera_elevation) * sin(camera_azimuth);
                  lookfrom.e[1] = lookat.y() + camera_distance * sin(camera_elevation);
                  lookfrom.e[2] = lookat.z() + camera_distance * cos(camera_elevation) * cos(camera_azimuth);
                  
                  camera_changed = true;
               }
               else if (right_button_down)
               {
                  // Pan camera (move lookat point)
                  double pan_speed = camera_distance * 0.001;
                  Vec3 right = unit_vector(cross(vup, w)) * (-dx * pan_speed);
                  Vec3 up = unit_vector(vup) * (dy * pan_speed);
                  
                  lookat += right + up;
                  lookfrom += right + up;
                  
                  camera_changed = true;
               }

               last_mouse_x = mouse_x;
               last_mouse_y = mouse_y;
            }
            else if (event.type == SDL_MOUSEWHEEL)
            {
               // Zoom in/out
               double zoom_factor = (event.wheel.y > 0) ? 0.9 : 1.1;
               camera_distance *= zoom_factor;
               
               // Update camera position
               lookfrom.e[0] = lookat.x() + camera_distance * cos(camera_elevation) * sin(camera_azimuth);
               lookfrom.e[1] = lookat.y() + camera_distance * sin(camera_elevation);
               lookfrom.e[2] = lookat.z() + camera_distance * cos(camera_elevation) * cos(camera_azimuth);
               
               camera_changed = true;
            }
         }
         
         // If camera changed, restart rendering from scratch
         if (camera_changed)
         {
            camera_changed = false;
            current_samples = 0;
            std::fill(accum_buffer.begin(), accum_buffer.end(), 0.0f);  // Clear accumulation buffer
            
            // Free and reset device random states to force reinitialization
            if (d_rand_states != nullptr) {
               freeDeviceRandomStates(d_rand_states);
               d_rand_states = nullptr;
            }
            
            initialize(); // Recalculate camera parameters
         }
         
         // If gamma or light intensity changed, reprocess display from accumulation buffer
         if (needs_rerender && current_samples > 0)
         {
            needs_rerender = false;
            
            // Reprocess accumulated buffer with new gamma (light intensity already in render)
            for (int j = 0; j < image_height; j++)
            {
               for (int i = 0; i < image_width; i++)
               {
                  int pixel_idx = j * image_width + i;
                  int display_idx = pixel_idx * image_channels;
                  int accum_idx = pixel_idx * 3;
                  
                  // Get accumulated color
                  float r = accum_buffer[accum_idx + 0] / current_samples;
                  float g = accum_buffer[accum_idx + 1] / current_samples;
                  float b = accum_buffer[accum_idx + 2] / current_samples;
                  
                  // Apply gamma correction (light intensity already in render)
                  r = pow(r, 1.0f / gamma);
                  g = pow(g, 1.0f / gamma);
                  b = pow(b, 1.0f / gamma);
                  
                  // Clamp and convert to bytes
                  static const Interval intensity_range(0.0, 0.999);
                  display_image[display_idx + 0] = static_cast<unsigned char>(256 * intensity_range.clamp(r));
                  display_image[display_idx + 1] = static_cast<unsigned char>(256 * intensity_range.clamp(g));
                  display_image[display_idx + 2] = static_cast<unsigned char>(256 * intensity_range.clamp(b));
                  if (image_channels == 4)
                     display_image[display_idx + 3] = 255;
               }
            }
            
            // Update display
            SDL_UpdateTexture(texture, nullptr, display_image.data(), image_width * image_channels);
            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, texture, nullptr, nullptr);
            
            // Draw logo if available
            if (logo_texture)
            {
               SDL_RenderCopy(renderer, logo_texture, nullptr, &logo_rect);
            }
            
            // Draw sample count text
            drawSampleCountText(renderer, font, current_samples);
            
            // Draw UI controls (sliders and toggle button)
            drawUIControls(renderer, font, gamma, light_intensity, background_intensity, accumulation_enabled, &gamma_slider_bounds, &intensity_slider_bounds, &background_slider_bounds, &toggle_button_rect);
            
            SDL_RenderPresent(renderer);
            
            // Copy to final image buffer
            image = display_image;
         }
         
         // Continue adding samples if we haven't reached max and accumulation is enabled
         if (current_samples < max_samples && !camera_changed && running && accumulation_enabled)
         {
            // Add samples_per_batch new samples to the accumulation
            current_samples += samples_per_batch;
            if (current_samples > max_samples)
               current_samples = max_samples;
            
            int actual_samples_to_add = samples_per_batch;
            if (current_samples > max_samples)
               actual_samples_to_add = max_samples - (current_samples - samples_per_batch);
            
            // Call CUDA accumulative rendering with persistent random states
            unsigned long long cuda_ray_count = ::renderPixelsCUDAAccumulative(
                display_image.data(), accum_buffer.data(), 
                image_width, image_height,
                camera_center.x(), camera_center.y(), camera_center.z(),
                pixel00_loc.x(), pixel00_loc.y(), pixel00_loc.z(),
                pixel_delta_u.x(), pixel_delta_u.y(), pixel_delta_u.z(),
                pixel_delta_v.x(), pixel_delta_v.y(), pixel_delta_v.z(),
                actual_samples_to_add, current_samples, max_depth,
                &d_rand_states);  // Pass persistent random states
            
            n_rays.fetch_add(cuda_ray_count, std::memory_order_relaxed);

            // Convert accumulated buffer to display image with current gamma (light intensity already in render)
            for (int j = 0; j < image_height; j++)
            {
               for (int i = 0; i < image_width; i++)
               {
                  int pixel_idx = j * image_width + i;
                  int display_idx = pixel_idx * image_channels;
                  int accum_idx = pixel_idx * 3;
                  
                  // Get accumulated color
                  float r = accum_buffer[accum_idx + 0] / current_samples;
                  float g = accum_buffer[accum_idx + 1] / current_samples;
                  float b = accum_buffer[accum_idx + 2] / current_samples;
                  
                  // Apply gamma correction (light intensity already in render)
                  r = pow(r, 1.0f / gamma);
                  g = pow(g, 1.0f / gamma);
                  b = pow(b, 1.0f / gamma);
                  
                  // Clamp and convert to bytes
                  static const Interval intensity_range(0.0, 0.999);
                  display_image[display_idx + 0] = static_cast<unsigned char>(256 * intensity_range.clamp(r));
                  display_image[display_idx + 1] = static_cast<unsigned char>(256 * intensity_range.clamp(g));
                  display_image[display_idx + 2] = static_cast<unsigned char>(256 * intensity_range.clamp(b));
                  if (image_channels == 4)
                     display_image[display_idx + 3] = 255;
               }
            }

            // Update display
            SDL_UpdateTexture(texture, nullptr, display_image.data(), image_width * image_channels);
            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, texture, nullptr, nullptr);
            
            // Draw logo if available
            if (logo_texture)
            {
               SDL_RenderCopy(renderer, logo_texture, nullptr, &logo_rect);
            }
            
            // Draw sample count text
            drawSampleCountText(renderer, font, current_samples);
            
            // Draw UI controls (sliders and toggle button)
            drawUIControls(renderer, font, gamma, light_intensity, background_intensity, accumulation_enabled, &gamma_slider_bounds, &intensity_slider_bounds, &background_slider_bounds, &toggle_button_rect);
            
            SDL_RenderPresent(renderer);

            // Copy to final image buffer
            image = display_image;

            cout << " done" << endl;
         }
         else if (current_samples >= max_samples && !camera_changed)
         {
            // Rendering complete, just wait for events
            SDL_Delay(16); // ~60 FPS event polling
         }
      }

      auto total_end = std::chrono::high_resolution_clock::now();
      cout << "\nTotal session time: " << timeStr(total_end - total_start) << endl;

      // Cleanup device random states
      if (d_rand_states != nullptr) {
         freeDeviceRandomStates(d_rand_states);
         d_rand_states = nullptr;
      }

      // Cleanup
#ifdef SDL2_TTF_FOUND
      if (font)
      {
         TTF_CloseFont(static_cast<TTF_Font*>(font));
      }
      TTF_Quit();
#endif
      
      if (logo_texture)
      {
         SDL_DestroyTexture(logo_texture);
      }
      SDL_DestroyTexture(texture);
      SDL_DestroyRenderer(renderer);
      SDL_DestroyWindow(window);
      SDL_Quit();
   }

#endif // SDL2_FOUND

 private:
   Point3 camera_center; // Camera center
   Point3 pixel00_loc;   // Location of pixel 0, 0
   Vec3 pixel_delta_u;   // Offset to pixel to the right
   Vec3 pixel_delta_v;   // Offset to pixel below
   Vec3 u, v, w;         // Camera frame basis vectors

#ifdef SDL2_FOUND
   // Camera control state for interactive mode
   bool left_button_down = false;
   bool right_button_down = false;
   int last_mouse_x = 0;
   int last_mouse_y = 0;
   double camera_azimuth = 0.0;
   double camera_elevation = 0.0;
   double camera_distance = 0.0;
#endif

   void initialize()
   {
      camera_center = lookfrom;

      // Determine viewport dimensions
      auto focal_length = (lookfrom - lookat).length();
      auto theta = utils::degrees_to_radians(vfov);
      auto h = tan(theta / 2);

      auto viewport_height = 2 * h * focal_length;
      auto viewport_width = viewport_height * (double(image_width) / image_height);

      // Compute camera basis vectors
      w = unit_vector(lookfrom - lookat);
      u = unit_vector(cross(vup, w));
      v = cross(w, u);

      // Calculate the vectors across the horizontal and down the vertical viewport edges
      auto viewport_u = viewport_width * u;
      auto viewport_v = viewport_height * -v;

      // Calculate the horizontal and vertical delta vectors from pixel to pixel
      pixel_delta_u = viewport_u / image_width;
      pixel_delta_v = viewport_v / image_height;

      // Calculate the location of the upper left pixel
      auto viewport_upper_left = camera_center - (focal_length * w) - viewport_u / 2 - viewport_v / 2;
      pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
   }

   /**
    * @brief Computes the color for a single pixel using ray tracing with anti-aliasing
    * This helper method contains the core pixel rendering logic
    *
    * @param scene The scene to render
    * @param x Pixel x coordinate
    * @param y Pixel y coordinate
    * @return Color The computed pixel color after anti-aliasing
    */
   Color computePixelColor(const Hittable &scene, int x, int y)
   {
      Color pixel_color(0, 0, 0); // The pixel color starts as black

      // Supersampling anti-aliasing by averaging multiple samples per pixel
      for (int s = 0; s < samples_per_pixel; ++s)
      {
         // Random offsets in the range [-0.5, 0.5) for jittering around the pixel center
         // but remaining within the pixel area
         double offset_x = RndGen::random_double() - 0.5;
         double offset_y = RndGen::random_double() - 0.5;

         // Calculate the direction of the ray for the current pixel
         Vec3 pixel_center = pixel00_loc + (x + offset_x) * pixel_delta_u + (y + offset_y) * pixel_delta_v;
         Vec3 ray_direction = pixel_center - camera_center;

         // Create a ray from the camera center through the pixel
         Ray ray(camera_center, unit_vector(ray_direction));

         // And launch baby, launch the ray to get the color
         Color sample(ray_color(ray, scene, max_depth));
         pixel_color += sample;
      }

      pixel_color /= samples_per_pixel; // Average the samples
      return pixel_color;
   }

   /**
    * Computes the color seen along a ray by tracing it through the scene
    */
   inline Color ray_color(const Ray &r, const Hittable &world, int depth)
   {
      if (depth <= 0)
         return Color(0, 0, 0); // No more light is gathered

      n_rays.fetch_add(1, std::memory_order_relaxed);

      Hit_record rec;

      if (world.hit(r, Interval(0.0001, inf), rec))
      {
         // Display only normal
         // return 0.5 * (rec.normal + Color(1, 1, 1));

         // if (rec.isMirror)
         // {
         //    Vec3 reflected = r.direction() - 2 * dot(r.direction(), rec.normal) * rec.normal;
         //    return Color(1, 0.85, .47) * 0.2 + 0.8 * ray_color(Ray(rec.p, unit_vector(reflected)), world, depth - 1);
         // }

         Ray scattered;
         Color attenuation;

         if (Constant *c = dynamic_cast<Constant *>(rec.mat_ptr.get()))
         {
            return c->color;
         }

         if (ShowNormals *c = dynamic_cast<ShowNormals *>(rec.mat_ptr.get()))
         {
            rec.mat_ptr->scatter(r, rec, attenuation, scattered);
            return attenuation;
         }

         if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
         {
            // For constant materials, the scattered ray direction is zero, so we just return the attenuation
            if (scattered.direction().length() == 0.0)
               return attenuation;
            else
               return attenuation * ray_color(scattered, world, depth - 1);
         }
      }

      // A blue to white gradient universe, where unit_direction varies between -1 and +1 in x and y
      Vec3 unit_direction = unit_vector(r.direction());
      float t = 0.5f * (unit_direction.y() + 1.0f);
      return (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
   }

   /**
    * Sets the pixel color in the image buffer
    */
   inline void setPixel(vector<unsigned char> &image, int x, int y, Color &c)
   {
      int index = (y * image_width + x) * image_channels;

      static const Interval intensity(0.0, 0.999);
      
      // Apply gamma correction (linear to sRGB) for proper display
      auto linear_to_gamma = [](double linear) {
         return sqrt(linear); // Simple gamma 2.0 approximation
      };

      image[index + 0] = static_cast<int>(intensity.clamp(linear_to_gamma(c.x())) * 256);
      image[index + 1] = static_cast<int>(intensity.clamp(linear_to_gamma(c.y())) * 256);
      image[index + 2] = static_cast<int>(intensity.clamp(linear_to_gamma(c.z())) * 256);
   }

   /***
    * Utility functions
    */
   void showProgress(int current, int total)
   {
      const int barWidth = 70;
      static int frame = 0;
      const char *spinner = "|/-\\";
      float progress = (float)(current + 1) / total;
      int pos = barWidth * progress;

      cout << "Rendering: " << spinner[frame++ % 4] << " [";
      for (int i = 0; i < barWidth; ++i)
      {
         if (i < pos)
         {
            cout << "█";
         }
         else
         {
            cout << "░";
         }
      }
      cout << "] " << int(progress * 100.0) << " %\r";
      cout.flush();
   }

   string timeStr(std::chrono::nanoseconds duration)
   {
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
      auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
      auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);

      std::ostringstream s;

      if (minutes.count() > 0)
      {
         s << minutes.count() << " minutes and " << (seconds.count() % 60) << " seconds";
      }
      else if (seconds.count() >= 10)
      {
         s << seconds.count() << " seconds";
      }
      else if (seconds.count() >= 1)
      {
         double sec_with_decimal = ms.count() / 1000.0;
         s << std::fixed << std::setprecision(2) << sec_with_decimal << " seconds";
      }
      else
      {
         s << ms.count() << " milliseconds";
      }

      return s.str();
   }
};