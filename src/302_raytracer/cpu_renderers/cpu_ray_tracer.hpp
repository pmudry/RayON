/**
 * @class CPURayTracer
 * @brief Trait providing CPU-based ray tracing methods
 *
 * This trait provides the core ray tracing algorithms used by CPU renderers.
 * It includes pixel color computation with anti-aliasing and recursive ray tracing.
 */
#pragma once

#include "camera_base.hpp"
#include "color.hpp"
#include "hittable.hpp"
#include "interval.hpp"
#include "material.hpp"
#include "ray.hpp"
#include "rnd_gen.hpp"
#include "vec3.hpp"

class CPURayTracer : virtual public CameraBase
{
 public:
   using CameraBase::CameraBase;

   /**
    * @brief Computes the color for a single pixel using anti-aliasing
    *
    * This method performs supersampling anti-aliasing by averaging multiple
    * randomly jittered samples per pixel.
    *
    * @param scene The hittable scene to render
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
         double offset_x = RndGen::random_double() - 0.5;
         double offset_y = RndGen::random_double() - 0.5;

         // Calculate the direction of the ray for the current pixel
         Vec3 pixel_center = pixel00_loc + (x + offset_x) * pixel_delta_u + (y + offset_y) * pixel_delta_v;
         Vec3 ray_direction = pixel_center - camera_center;

         // Create a ray from the camera center through the pixel
         Ray ray(camera_center, unit_vector(ray_direction));

         // Launch the ray to get the color
         Color sample(ray_color(ray, scene, max_depth));
         pixel_color += sample;
      }

      pixel_color /= samples_per_pixel; // Average the samples
      return pixel_color;
   }

   /**
    * @brief Computes the color seen along a ray by tracing it through the scene
    *
    * This method recursively traces rays through the scene, handling material
    * interactions, reflections, and refractions up to the specified depth.
    *
    * @param r The ray to trace
    * @param world The hittable scene
    * @param depth Maximum recursion depth
    * @return Color The computed color along the ray
    */
   inline Color ray_color(const Ray &r, const Hittable &world, int depth)
   {
      if (depth <= 0)
         return Color(0, 0, 0); // No more light is gathered

      n_rays.fetch_add(1, std::memory_order_relaxed);

      Hit_record rec;

      if (world.hit(r, Interval(0.0001, inf), rec))
      {
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
            // For constant materials, the scattered ray direction is zero
            if (scattered.direction().length() == 0.0)
               return attenuation;
            else
               return attenuation * ray_color(scattered, world, depth - 1);
         }
      }

      // Background: blue to white gradient
      Vec3 unit_direction = unit_vector(r.direction());
      float t = 0.5f * (unit_direction.y() + 1.0f);
      return (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
   }
};
