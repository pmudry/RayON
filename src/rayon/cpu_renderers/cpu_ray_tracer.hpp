/**
 * @class CPURayTracer
 * @brief Trait providing CPU-based ray tracing methods with Multiple Importance Sampling
 *
 * Implements an iterative path tracer with:
 *  - Next Event Estimation (NEE) — direct light sampling via shadow ray
 *  - MIS via the power heuristic to combine NEE and BSDF sampling
 *  - Cosine-weighted hemisphere sampling for Lambertian surfaces
 */
#pragma once

#include <atomic>

#include "camera/camera_frame.hpp"

#include "data_structures/color.hpp"
#include "data_structures/hittable.hpp"
#include "data_structures/hittable_list.hpp"
#include "data_structures/interval.hpp"
#include "data_structures/material.hpp"
#include "data_structures/mis_utils.hpp"
#include "data_structures/ray.hpp"
#include "data_structures/vec3.hpp"

#include "utils/rnd_gen.hpp"

class CPURayTracer
{
 public:
   /**
    * @brief Compute the colour of pixel (x,y) by averaging @p samples_per_pixel estimates.
    *
    * @param frame        Camera frame with resolution, FOV, etc.
    * @param scene        The full scene (all geometry)
    * @param lights       Emissive-only geometry list for NEE light sampling
    * @param x            Pixel column
    * @param y            Pixel row
    * @param ray_counter  Atomic counter incremented once per primary/shadow ray
    */
   static Color computePixelColor(const CameraFrame &frame, const Hittable &scene,
                                  const Hittable_list &lights, int x, int y,
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
         pixel_color += ray_color(ray, scene, lights, frame.max_depth, ray_counter);
      }

      pixel_color /= frame.samples_per_pixel;
      return pixel_color;
   }

 private:
   /**
    * @brief Iterative path tracer with MIS (NEE + BSDF sampling).
    *
    * Algorithm per bounce:
    *  1. Trace ray → record hit.
    *  2. If emissive hit: add throughput * Le * MIS_weight (BSDF-sample path).
    *  3. If material scatters and is not a delta BSDF:
    *       a. Sample a direction toward the lights (NEE).
    *       b. Cast shadow ray — if unoccluded, add NEE contribution with MIS weight.
    *  4. Sample a new direction via BSDF, update throughput, advance ray.
    */
   static Color ray_color(const Ray &r, const Hittable &world, const Hittable_list &lights,
                           int depth, std::atomic<long long> &ray_counter)
   {
      Color  accumulated(0, 0, 0);
      Color  throughput(1, 1, 1);
      Ray    current_ray   = r;
      double prev_bsdf_pdf = 1.0;   // treat camera ray as delta (no prior NEE)
      bool   prev_specular = true;  // so emissive hits at bounce 0 get full weight

      for (int bounce = 0; bounce < depth; bounce++)
      {
         ray_counter.fetch_add(1, std::memory_order_relaxed);

         Hit_record rec;
         if (!world.hit(current_ray, Interval(0.0001, inf), rec))
         {
            // Sky / background gradient
            Vec3  unit_dir = unit_vector(current_ray.direction());
            float t        = 0.5f * static_cast<float>(unit_dir.y() + 1.0);
            Color sky      = (1.0f - t) * Color(1, 1, 1) + t * Color(0.5, 0.7, 1.0);
            accumulated   += throughput * sky;
            break;
         }

         // --- Constant / ShowNormals are debug-only terminal materials ---
         if (auto *c = dynamic_cast<Constant *>(rec.mat_ptr.get()))
         {
            if (bounce == 0)
               return c->color;
            break;
         }
         if (dynamic_cast<ShowNormals *>(rec.mat_ptr.get()))
         {
            Color n = 0.5 * (rec.normal + Vec3_ONES);
            if (bounce == 0)
               return n;
            break;
         }

         // --- 1. Emissive contribution (BSDF-sampled path) with MIS weight ---
         Color Le = rec.mat_ptr->emitted(rec);
         if (Le.length_squared() > 0.0)
         {
            double w = 1.0;
            if (bounce > 0 && !prev_specular && !lights.empty())
            {
               // PDF that the NEE mixture distribution would assign to this direction
               double light_pdf = lights.pdf_value(current_ray.origin(),
                                                   unit_vector(current_ray.direction()));
               if (light_pdf > 0.0)
                  w = power_heuristic(prev_bsdf_pdf, light_pdf);
            }
            accumulated += throughput * Le * w;
         }

         // --- 2. Sample next direction via material ---
         ScatterRecord srec;
         if (!rec.mat_ptr->scatter_mis(current_ray, rec, srec))
            break; // material absorbs or is terminal

         // --- 3. NEE: direct light sampling (skip for delta BSDFs) ---
         if (!srec.is_specular && !lights.empty())
         {
            Vec3   to_light  = lights.random_direction(rec.p);
            double light_pdf = lights.pdf_value(rec.p, to_light);

            if (light_pdf > 1e-8)
            {
               to_light = unit_vector(to_light);
               Ray shadow(rec.p + 0.0001 * rec.normal, to_light);

               ray_counter.fetch_add(1, std::memory_order_relaxed);

               Hit_record shadow_rec;
               if (world.hit(shadow, Interval(0.0001, inf), shadow_rec))
               {
                  Color Le_nee = shadow_rec.mat_ptr->emitted(shadow_rec);
                  if (Le_nee.length_squared() > 0.0)
                  {
                     Color  f        = rec.mat_ptr->eval_bsdf(current_ray, rec, to_light);
                     double cos_th   = std::max(0.0, dot(to_light, rec.normal));
                     double p_mat    = rec.mat_ptr->scatter_pdf(current_ray, rec, to_light);
                     double w_nee    = power_heuristic(light_pdf, p_mat);
                     accumulated    += throughput * f * Le_nee * cos_th * w_nee / light_pdf;
                  }
               }
            }
         }

         // --- 4. Update throughput and advance path ---
         Vec3 scatter_dir = unit_vector(srec.scattered.direction());
         if (srec.is_specular)
         {
            throughput   *= srec.bsdf_value;
            prev_bsdf_pdf = 1.0;
            prev_specular = true;
         }
         else
         {
            double cos_th = std::max(0.0, dot(scatter_dir, rec.normal));
            if (srec.pdf > 1e-10)
               throughput *= srec.bsdf_value * cos_th / srec.pdf;
            prev_bsdf_pdf = srec.pdf;
            prev_specular = false;
         }

         current_ray = srec.scattered;
      }

      return accumulated;
   }
};
