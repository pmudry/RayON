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
#include <mutex>
#include <thread>
#include <vector>

#ifdef SDL2_FOUND
#include <SDL2/SDL.h>
#endif

#pragma once

inline double degrees_to_radians(double degrees) { return degrees * M_PI / 180.0; }

class Camera
{

 public:
   int image_width;
   int image_height;
   double vfov = 35.0;                   // Vertical field of view in degrees
   Point3 lookfrom = Point3(-2, 2, 5);   // Point camera is looking from
   Point3 lookat = Point3(-2, -0.5, -1); // Point camera is looking at
   Vec3 vup = Vec3(0, 1, 0);             // Camera-relative "up" direction

   int samples_per_pixel;                      // Number of samples per pixel for anti-aliasing
   const int max_depth = constants::MAX_DEPTH; // Maximum ray bounce depth

   std::atomic<long long> n_rays{0}; // Number of rays traced so far with this cam (thread-safe)

   Camera(const Point3 &center, const int image_width, const int image_height, const int image_channels,
          int samples_per_pixel = 1)
       : image_width(image_width), image_height(image_height), samples_per_pixel(samples_per_pixel),
         image_channels(image_channels), camera_center(center)
   {
      initialize();
   }

   Camera() : Camera(Vec3(0, 0, 0), 720, 3, 1) {}

   void renderPixels(const Hittable &scene, vector<unsigned char> &image)
   {

      for (int y = 0; y < image_height; ++y)
      {
         for (int x = 0; x < image_width; ++x)
         {
            Color pixel_color(0, 0, 0); // The pixel color starts as black

            // Supersampling anti-aliasing by averaging multiple samples per pixel
            for (int s = 0; s < samples_per_pixel; ++s)
            {
               // Random offsets in the range [0, 1) for jittering within the pixel
               double offset_x = RndGen::random_double();
               double offset_y = RndGen::random_double();

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

            setPixel(image, x, y, pixel_color);
         }

         showProgress(y, image_height);
      }

      showProgress(image_height - 1, image_height);
      cout << endl;
   }

   void renderPixelsParallel(const Hittable &scene, vector<unsigned char> &image)
   {
      const int num_threads = 72;
      std::vector<std::thread> threads(num_threads);
      std::mutex progress_mutex;
      int completed_rows = 0; // Track globally completed rows

      auto render_chunk = [&](int start_y, int end_y)
      {
         for (int y = start_y; y < end_y; ++y)
         {
            for (int x = 0; x < image_width; ++x)
            {
               Color pixel_color(0, 0, 0); // The pixel color starts as black

               // Supersampling anti-aliasing by averaging multiple samples per pixel
               for (int s = 0; s < samples_per_pixel; ++s)
               {
                  // Random offsets in the range [0, 1) for jittering within the pixel
                  double offset_x = RndGen::random_double();
                  double offset_y = RndGen::random_double();

                  // Calculate the direction of the ray for the current pixel
                  Vec3 pixel_center = pixel00_loc + (x + offset_x) * pixel_delta_u + (y + offset_y) * pixel_delta_v;
                  Vec3 ray_direction = pixel_center - camera_center;

                  // Create a ray from the camera center through the pixel
                  Ray ray(camera_center, unit_vector(ray_direction));
                  Color sample(ray_color(ray, scene, max_depth));
                  pixel_color += sample;
               }

               pixel_color /= samples_per_pixel; // Average the samples

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
      cout << endl;
   }

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
                             pixel_delta_v.z(), samples_per_pixel, max_depth, 0, 0, image_width, image_height);

      // Add the CUDA ray count to our atomic counter
      n_rays.fetch_add(cuda_ray_count, std::memory_order_relaxed);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
      auto seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
      auto minutes = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);

      cout << "CUDA rendering completed in ";
      if (minutes.count() > 0)
      {
         cout << minutes.count() << " minutes and " << (seconds.count()) << " seconds" << endl;
      }
      if (seconds.count() > 0)
      {
         cout << seconds.count() << " seconds " << endl;
      }
      else
      {
         cout << (ms.count() % 1000) << " milliseconds" << endl;
      }
   }

   void renderPixelsParallelWithTiming(const Hittable &scene, vector<unsigned char> &image)
   {
      auto start_time = std::chrono::high_resolution_clock::now();

      renderPixelsParallel(scene, image);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

      cout << "Parallel rendering completed in " << duration.count() << " milliseconds" << endl;
   }

   

#ifdef SDL2_FOUND
   void renderPixelsCUDART(vector<unsigned char> &image)
   {
      // Initialize SDL
      if (SDL_Init(SDL_INIT_VIDEO) < 0)
      {
         std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
         return;
      }

      // Create window
      SDL_Window *window = SDL_CreateWindow("CUDA Ray Tracer - Real-time", SDL_WINDOWPOS_UNDEFINED,
                                            SDL_WINDOWPOS_UNDEFINED, image_width, image_height, SDL_WINDOW_SHOWN);
      if (window == nullptr)
      {
         std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
         SDL_Quit();
         return;
      }

      // Create renderer
      SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
      if (renderer == nullptr)
      {
         std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
         SDL_DestroyWindow(window);
         SDL_Quit();
         return;
      }

      // Create texture for the image
      SDL_Texture *texture =
          SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, image_width, image_height);
      if (texture == nullptr)
      {
         std::cerr << "Texture could not be created! SDL_Error: " << SDL_GetError() << std::endl;
         SDL_DestroyRenderer(renderer);
         SDL_DestroyWindow(window);
         SDL_Quit();
         return;
      }

      auto start_time = std::chrono::high_resolution_clock::now();

      // Render in tiles for real-time display
      const int tile_size = 64; // Size of each tile
      const int tiles_x = (image_width + tile_size - 1) / tile_size;
      const int tiles_y = (image_height + tile_size - 1) / tile_size;

      std::cout << "Rendering in " << tiles_x << "x" << tiles_y << " tiles..." << std::endl;

      for (int tile_y = 0; tile_y < tiles_y; ++tile_y)
      {
         for (int tile_x = 0; tile_x < tiles_x; ++tile_x)
         {
            int start_x = tile_x * tile_size;
            int start_y = tile_y * tile_size;
            int end_x = std::min(start_x + tile_size, image_width);
            int end_y = std::min(start_y + tile_size, image_height);

            // Render this tile
            unsigned long long cuda_ray_count = ::renderPixelsCUDA(
                image.data(), image_width, image_height, camera_center.x(), camera_center.y(), camera_center.z(),
                pixel00_loc.x(), pixel00_loc.y(), pixel00_loc.z(), pixel_delta_u.x(), pixel_delta_u.y(),
                pixel_delta_u.z(), pixel_delta_v.x(), pixel_delta_v.y(), pixel_delta_v.z(), samples_per_pixel,
                max_depth, start_x, start_y, end_x, end_y);

            n_rays.fetch_add(cuda_ray_count, std::memory_order_relaxed);

            // Update texture with the new tile
            SDL_UpdateTexture(texture, nullptr, image.data(), image_width * 3);

            // Clear and render
            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, texture, nullptr, nullptr);
            SDL_RenderPresent(renderer);

            // Handle events to keep window responsive
            SDL_Event event;
            while (SDL_PollEvent(&event))
            {
               if (event.type == SDL_QUIT)
               {
                  goto cleanup;
               }
            }

            // Small delay to make the progressive rendering visible
            SDL_Delay(1);
         }
      }

   cleanup:
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

      std::cout << "Real-time CUDA rendering completed in " << duration.count() << " milliseconds" << std::endl;
      std::cout << "Press any key to close the window..." << std::endl;

      // Wait for user to close window
      bool quit = false;
      while (!quit)
      {
         SDL_Event event;
         while (SDL_PollEvent(&event))
         {
            if (event.type == SDL_QUIT)
            {
               quit = true;
            }
         }
         SDL_Delay(1);
      }

      // Cleanup
      SDL_DestroyTexture(texture);
      SDL_DestroyRenderer(renderer);
      SDL_DestroyWindow(window);
      SDL_Quit();
   }
#endif

 private:
   int image_channels = 3; // Number of color channels per pixel (e.g., 3 for RGB)

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
      auto theta = degrees_to_radians(vfov);
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
         return 0.5 * (rec.normal + Color(1, 1, 1));

         // if (rec.isMirror)
         // {
         //    Vec3 reflected = r.direction() - 2 * dot(r.direction(), rec.normal) * rec.normal;
         //    return Color(1, 0.85, .47) * 0.2 + 0.8 * ray_color(Ray(rec.p, unit_vector(reflected)), world, depth - 1);
         // }

         auto new_ray = Vec3::random_in_hemisphere(rec.normal);

         // A ray bounces and keeps only 70% of its color. It's a grey object.
         // If it returns 100%, it's white and if 0% it's black.
         return 0.7 * ray_color(Ray(rec.p, new_ray), world, depth - 1);
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
};