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

   /**
    * @brief Renders progressively with SDL2 window display showing incremental quality improvements
    * 
    * This method displays the image in an SDL window while rendering with increasing sample rates.
    * It starts with low sample counts for quick preview and progressively increases quality.
    * Interactive camera controls with automatic re-rendering:
    * - Left mouse button: Rotate camera (orbit around lookat point)
    * - Right mouse button: Pan camera (move lookat point)
    * - Mouse wheel: Zoom in/out (change camera distance)
    * - Any camera movement automatically triggers re-rendering from 8 samples
    * - ESC/Close window: Exit
    *
    * The interactive mode starts rendering from 8 samples for fast response.
    * Any camera movement interrupts current rendering and immediately starts a new render.
    * Maximum sample count capped at 256 for interactive responsiveness.
    * For highest quality renders, use the regular CUDA rendering mode.
    * 
    * Tiled rendering: Frame is divided into fixed 8x8 block grid (64 tiles) for consistent
    * visual appearance and responsive interaction at all quality levels.
    * 500ms delay between quality stages to allow viewing each stage clearly.
    * 
    * @param image The final image buffer to store the highest quality render
    * @param sample_stages Vector of sample counts for progressive rendering (default: {8, 16, 32, 64, 128, 256})
    */
   void renderPixelsSDLProgressive(vector<unsigned char> &image, const vector<int> &sample_stages = {8, 16, 32, 64, 128, 256})
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
      SDL_Window *window = SDL_CreateWindow("ISC - 302 ray tracer (mui) / Interactive mode (LMB:Rotate RMB:Pan Wheel:Zoom)", 
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
      
      SDL_Texture *logo_texture = nullptr;
      SDL_Rect logo_rect = {0, 0, 0, 0};
      
      if (logo_data)
      {
         cout << "Logo loaded: " << logo_img_width << "x" << logo_img_height << " channels: " << logo_img_channels << endl;
         
         // Calculate target logo size (1/10th of window width, maintain aspect ratio)
         int logo_width = image_width / 5;
         int logo_height = (logo_width * logo_img_height) / logo_img_width;
         
         // Pre-scale logo using high-quality Mitchell filter for best quality
         unsigned char* scaled_logo = new unsigned char[logo_width * logo_height * 4];
         stbir_resize_uint8_srgb(logo_data, logo_img_width, logo_img_height, 0,
                                  scaled_logo, logo_width, logo_height, 0,
                                  STBIR_RGBA);
         
         // Create surface from pre-scaled logo data
         SDL_Surface *logo_surface = SDL_CreateRGBSurfaceFrom(
            scaled_logo, logo_width, logo_height, 32, logo_width * 4,
            0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);
         
         if (logo_surface)
         {
            logo_texture = SDL_CreateTextureFromSurface(renderer, logo_surface);
            SDL_FreeSurface(logo_surface);
            
            if (logo_texture)
            {
               // Enable alpha blending for logo
               SDL_SetTextureBlendMode(logo_texture, SDL_BLENDMODE_BLEND);
               
               // Position at bottom-right corner
               logo_rect.x = image_width - logo_width - 10; // 10px padding from edge
               logo_rect.y = image_height - logo_height - 10;
               logo_rect.w = logo_width;
               logo_rect.h = logo_height;
            }
         }
         
         // Clean up
         delete[] scaled_logo;
         stbi_image_free(logo_data);
      }
      else
      {
         cerr << "Warning: Could not load logo: " << stbi_failure_reason() << endl;
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

      cout << "\n=== Interactive Progressive SDL Rendering ===" << endl;
      cout << "Controls:" << endl;
      cout << "  Left Mouse Button:  Rotate camera (orbit)" << endl;
      cout << "  Right Mouse Button: Pan camera" << endl;
      cout << "  Mouse Wheel:        Zoom in/out" << endl;
      cout << "  ESC:                Exit" << endl;
      cout << "Progressive quality stages: 8 → 16 → 32 → 64 → 128 → 256 samples" << endl;
      cout << "Full-frame rendering for 8 & 16 samples, 8x8 tiles for higher quality" << endl;
      
      bool running = true;
      bool camera_changed = true; // Trigger initial render
      bool rendering = false; // Track if currently rendering
      size_t current_stage = 0; // Track current rendering stage
      SDL_Event event;
      vector<unsigned char> stage_image(image_width * image_height * image_channels);

      // Initialize camera control state (now member variables)
      left_button_down = false;
      right_button_down = false;
      last_mouse_x = 0;
      last_mouse_y = 0;
      
      // Find starting stage (first stage with >= 8 samples for quick preview)
      size_t quick_start_stage = 0;
      for (size_t i = 0; i < sample_stages.size(); i++)
      {
         if (sample_stages[i] >= 8)
         {
            quick_start_stage = i;
            break;
         }
      }
      
      cout << "Quick start stage index: " << quick_start_stage << " (samples: " << sample_stages[quick_start_stage] << ")" << endl;
      
      // Initialize camera orbit parameters
      camera_distance = (lookfrom - lookat).length();
      camera_azimuth = 0.0;   // Horizontal rotation angle
      camera_elevation = 0.0; // Vertical rotation angle (now member variable)
      
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
            else if (event.type == SDL_MOUSEBUTTONDOWN)
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
               if (event.button.button == SDL_BUTTON_LEFT)
               {
                  left_button_down = false;
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

               if (left_button_down)
               {
                  // Rotate camera (orbit around lookat point)
                  camera_azimuth += dx * 0.01;
                  camera_elevation -= dy * 0.01;
                  
                  // Clamp elevation to avoid gimbal lock
                  const double max_elevation = 1.5;
                  if (camera_elevation > max_elevation) camera_elevation = max_elevation;
                  if (camera_elevation < -max_elevation) camera_elevation = -max_elevation;
                  
                  // Calculate new camera position
                  double cos_elev = cos(camera_elevation);
                  lookfrom.e[0] = lookat.x() + camera_distance * cos_elev * sin(camera_azimuth);
                  lookfrom.e[1] = lookat.y() + camera_distance * sin(camera_elevation);
                  lookfrom.e[2] = lookat.z() + camera_distance * cos_elev * cos(camera_azimuth);
                  
                  // Trigger re-render immediately
                  camera_changed = true;
               }
               else if (right_button_down)
               {
                  // Pan camera (move both lookfrom and lookat)
                  Vec3 right = unit_vector(cross(lookfrom - lookat, vup));
                  Vec3 up = unit_vector(cross(right, lookfrom - lookat));
                  
                  double pan_speed = camera_distance * 0.001;
                  Vec3 pan_offset = right * (-dx * pan_speed) + up * (dy * pan_speed);
                  
                  lookfrom = lookfrom + pan_offset;
                  lookat = lookat + pan_offset;
                  
                  // Trigger re-render immediately
                  camera_changed = true;
               }

               last_mouse_x = mouse_x;
               last_mouse_y = mouse_y;
            }
            else if (event.type == SDL_MOUSEWHEEL)
            {
               // Zoom in/out (change camera distance)
               double zoom_factor = event.wheel.y > 0 ? 0.9 : 1.1;
               camera_distance *= zoom_factor;
               
               // Recalculate camera position maintaining angles
               double cos_elev = cos(camera_elevation);
               lookfrom.e[0] = lookat.x() + camera_distance * cos_elev * sin(camera_azimuth);
               lookfrom.e[1] = lookat.y() + camera_distance * sin(camera_elevation);
               lookfrom.e[2] = lookat.z() + camera_distance * cos_elev * cos(camera_azimuth);
               
               // Trigger re-render immediately
               camera_changed = true;
            }
         }

         // Render if camera has changed
         if (camera_changed)
         {
            camera_changed = false;
            rendering = true;
            initialize(); // Recalculate camera parameters
            
            // Start from quick preview stage when camera moves
            current_stage = quick_start_stage;
            cout << "\n[Camera changed - restarting from stage " << (current_stage + 1) 
                 << " (index " << current_stage << ", samples: " << sample_stages[current_stage] << ")]" << endl;
         }
         
         // Continue progressive rendering through stages
         if (rendering && current_stage < sample_stages.size() && !camera_changed && running)
         {
            int current_samples = sample_stages[current_stage];
            int original_samples = samples_per_pixel;
            samples_per_pixel = current_samples;

            // For 8 and 16 samples, use full-frame rendering (1x1 tile = whole image)
            // For higher samples, use 8x8 tiles for responsiveness
            int tiles_per_axis = (current_samples <= 16) ? 1 : 8;
            
            const char* render_mode = (tiles_per_axis == 1) ? "full frame" : "8x8 tiles";
            cout << "Rendering stage " << (current_stage - quick_start_stage + 1) 
                 << " with " << current_samples << " samples (" 
                 << render_mode << ")..." << flush;

            // Render with tiled CUDA rendering for responsiveness
            bool completed = renderTiled(stage_image, tiles_per_axis, tiles_per_axis, renderer, texture, 
                                        logo_texture, logo_rect, font, running, camera_changed);

            if (completed)
            {
               cout << " done" << endl;

               // Copy to final image buffer
               image = stage_image;

               samples_per_pixel = original_samples;
               
               // Move to next stage for next iteration
               current_stage++;
               
               // Add 500ms delay after displaying each stage so user can see the quality improvement
               if (current_stage < sample_stages.size())
               {
                  //cout << "delay !" << endl;
                  //SDL_Delay(1500);
               }
            }
            else
            {
               cout << " interrupted" << endl;
               samples_per_pixel = original_samples;
               // camera_changed or running will be handled in the next loop iteration
            }
            
            // If we've completed all stages, stop rendering
            if (current_stage >= sample_stages.size())
            {
               rendering = false;
            }
         }
         
         // Small delay when idle to avoid busy-waiting
         if (!rendering && !camera_changed)
         {
            SDL_Delay(16); // ~60 FPS event polling
         }
      }

      auto total_end = std::chrono::high_resolution_clock::now();
      cout << "\nTotal session time: " << timeStr(total_end - total_start) << endl;

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