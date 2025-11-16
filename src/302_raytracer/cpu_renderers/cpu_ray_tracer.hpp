/**
 * @class CPURayTracer
 * @brief Trait providing CPU-based ray tracing methods
 *
 * This trait provides the core ray tracing algorithms used by CPU renderers.
 * It includes pixel color computation with anti-aliasing and recursive ray tracing.
 */
#pragma once

#include <atomic>

#include "camera/camera_frame.hpp"

#include "data_structures/color.hpp"
#include "data_structures/hittable.hpp"
#include "data_structures/interval.hpp"
#include "data_structures/material.hpp"
#include "data_structures/ray.hpp"
#include "data_structures/vec3.hpp"

#include "utils/rnd_gen.hpp"

class CPURayTracer
{
 public:
   static Color computePixelColor(const CameraFrame &frame, const Hittable &scene, int x, int y,
                                  std::atomic<long long> &ray_counter)
   {
      Color pixel_color(0, 0, 0);

      for (int s = 0; s < frame.samples_per_pixel; ++s)
      {
         const double offset_x = RndGen::random_double() - 0.5;
         const double offset_y = RndGen::random_double() - 0.5;

         const Vec3 pixel_center = frame.pixel00_loc + (x + offset_x) * frame.pixel_delta_u +
                                   (y + offset_y) * frame.pixel_delta_v;
         const Vec3 ray_direction = pixel_center - frame.camera_center;

         Ray ray(frame.camera_center, unit_vector(ray_direction));

         pixel_color += ray_color(ray, scene, frame.max_depth, ray_counter);
      }

      pixel_color /= frame.samples_per_pixel;
      return pixel_color;
   }

  private:
   static Color ray_color(const Ray &r, const Hittable &world, int depth, std::atomic<long long> &ray_counter)
   {
      if (depth <= 0)
         return Color(0, 0, 0);

      ray_counter.fetch_add(1, std::memory_order_relaxed);

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
            if (scattered.direction().length() == 0.0)
               return attenuation;
            else
               return attenuation * ray_color(scattered, world, depth - 1, ray_counter);
         }
      }

      
      const Vec3 unit_direction = unit_vector(r.direction());
      const float t = 0.5f * (unit_direction.y() + 1.0f);
      return (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
   }
};
