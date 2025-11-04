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
    * @brief Renders progressively with SDL2 window display showing incremental quality improvements
    * 
    * This method displays the image in an SDL window while rendering with increasing sample rates.
    * It starts with low sample counts for quick preview and progressively increases quality.
    * Interactive camera controls:
    * - Left mouse button: Rotate camera (orbit around lookat point)
    * - Right mouse button: Pan camera (move lookat point)
    * - Mouse wheel: Zoom in/out (change camera distance)
    * - ESC/Close window: Stop rendering
    * - Space: Re-render with current camera position
    * 
    * @param image The final image buffer to store the highest quality render
    * @param sample_stages Vector of sample counts for progressive rendering (e.g., {1, 4, 16, 64, 256})
    */
   void renderPixelsSDLProgressive(vector<unsigned char> &image, const vector<int> &sample_stages = {1, 4, 16, 64, 256})
   {
      // Initialize SDL
      if (SDL_Init(SDL_INIT_VIDEO) < 0)
      {
         cerr << "SDL initialization failed: " << SDL_GetError() << endl;
         return;
      }

      // Create window
      SDL_Window *window = SDL_CreateWindow("Ray Tracer - Interactive Camera (LMB:Rotate RMB:Pan Wheel:Zoom Space:Render)", 
                                             SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
                                             image_width, image_height, SDL_WINDOW_SHOWN);
      if (!window)
      {
         cerr << "Window creation failed: " << SDL_GetError() << endl;
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

      cout << "\n=== Interactive Progressive SDL Rendering ===" << endl;
      cout << "Controls:" << endl;
      cout << "  Left Mouse Button:  Rotate camera (orbit)" << endl;
      cout << "  Right Mouse Button: Pan camera" << endl;
      cout << "  Mouse Wheel:        Zoom in/out" << endl;
      cout << "  Space:              Re-render with current camera" << endl;
      cout << "  ESC:                Exit\n" << endl;

      bool running = true;
      bool camera_changed = true; // Trigger initial render
      SDL_Event event;
      vector<unsigned char> stage_image(image_width * image_height * image_channels);

      // Camera control state
      bool left_button_down = false;
      bool right_button_down = false;
      int last_mouse_x = 0, last_mouse_y = 0;
      
      // Camera orbit parameters
      double camera_distance = (lookfrom - lookat).length();
      double camera_azimuth = 0.0;   // Horizontal rotation angle
      double camera_elevation = 0.0; // Vertical rotation angle
      
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
            else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_SPACE)
            {
               camera_changed = true;
               cout << "\nRe-rendering with new camera position..." << endl;
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
               
               camera_changed = true;
            }
         }

         // Render if camera has changed
         if (camera_changed)
         {
            camera_changed = false;
            initialize(); // Recalculate camera parameters
            
            // Progressive rendering through sample stages
            for (size_t stage = 0; stage < sample_stages.size() && !camera_changed && running; stage++)
            {
               int current_samples = sample_stages[stage];
               int original_samples = samples_per_pixel;
               samples_per_pixel = current_samples;

               cout << "Rendering stage " << (stage + 1) << "/" << sample_stages.size() << " with " << current_samples
                    << " samples per pixel..." << flush;

               auto stage_start = std::chrono::high_resolution_clock::now();

               // Render with CUDA for this stage
               unsigned long long cuda_ray_count =
                   ::renderPixelsCUDA(stage_image.data(), image_width, image_height, camera_center.x(), camera_center.y(),
                                      camera_center.z(), pixel00_loc.x(), pixel00_loc.y(), pixel00_loc.z(), pixel_delta_u.x(),
                                      pixel_delta_u.y(), pixel_delta_u.z(), pixel_delta_v.x(), pixel_delta_v.y(),
                                      pixel_delta_v.z(), current_samples, max_depth);

               n_rays.fetch_add(cuda_ray_count, std::memory_order_relaxed);

               auto stage_end = std::chrono::high_resolution_clock::now();
               cout << " completed in " << timeStr(stage_end - stage_start) << endl;

               // Update SDL texture and display
               SDL_UpdateTexture(texture, nullptr, stage_image.data(), image_width * image_channels);
               SDL_RenderClear(renderer);
               SDL_RenderCopy(renderer, texture, nullptr, nullptr);
               SDL_RenderPresent(renderer);

               // Copy to final image buffer
               image = stage_image;

               // Check for events during rendering (to allow interruption)
               while (SDL_PollEvent(&event))
               {
                  if (event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE))
                  {
                     running = false;
                     cout << "\nRendering stopped by user at stage " << (stage + 1) << "/" << sample_stages.size() << endl;
                     break;
                  }
                  else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_SPACE)
                  {
                     camera_changed = true;
                     cout << "\nRe-rendering interrupted, starting new render..." << endl;
                     break;
                  }
                  // Handle camera controls even during rendering
                  else if (event.type == SDL_MOUSEBUTTONDOWN || event.type == SDL_MOUSEMOTION || 
                           event.type == SDL_MOUSEWHEEL || event.type == SDL_MOUSEBUTTONUP)
                  {
                     // Re-queue the event for next loop iteration
                     SDL_PushEvent(&event);
                  }
               }

               samples_per_pixel = original_samples;
            }
         }
         else
         {
            // Just wait for events when not rendering
            SDL_Delay(16); // ~60 FPS event polling
         }
      }

      auto total_end = std::chrono::high_resolution_clock::now();
      cout << "\nTotal session time: " << timeStr(total_end - total_start) << endl;

      // Cleanup
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

      image[index + 0] = static_cast<int>(intensity.clamp(c.x()) * 256);
      image[index + 1] = static_cast<int>(intensity.clamp(c.y()) * 256);
      image[index + 2] = static_cast<int>(intensity.clamp(c.z()) * 256);
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