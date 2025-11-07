// Common device-side utilities and shading routines shared by CUDA shaders
#pragma once
#include "../cuda_float3.cuh"
#include "../cuda_scene.cuh"
#include "../cuda_utils.cuh"


#include <cfloat>
#include <cmath>
#include <curand_kernel.h>

// Extern declarations for device-side global constants (defined once in renderer_cuda.cu)
extern __constant__ float g_light_intensity;
extern __constant__ float g_background_intensity;
extern __constant__ float g_metal_fuzziness;

// Forward declarations for golf-ball helpers implemented in shader_golf.cu
struct ray_simple;
struct hit_record_simple;
__device__ float3_simple fibonacci_point(int i, int n);
__device__ bool hit_golf_ball_sphere(float3_simple center, float radius, const ray_simple &r, float t_min,
                                     float t_max, hit_record_simple &rec);

//==============================================================================
// RAY TRACING DATA STRUCTURES
//==============================================================================

struct ray_simple
{
   float3_simple orig, dir;
   __device__ ray_simple() {}
   __device__ ray_simple(const float3_simple &origin, const float3_simple &direction) : orig(origin), dir(direction) {}
   __device__ float3_simple at(float t) const { return orig + t * dir; }
};

enum LegacyMaterialType
{
   LAMBERTIAN = 0,
   MIRROR = 1,
   GLASS = 2,
   LIGHT = 3,
   ROUGH_MIRROR = 4
};

struct hit_record_simple
{
   float3_simple p, normal;
   float t;
   bool front_face;
   LegacyMaterialType material;
   float3_simple color;
   float refractive_index;
   float3_simple emission;
   float roughness;
};

//==============================================================================
// OPTICAL PHYSICS FUNCTIONS
//==============================================================================

__device__ inline float3_simple reflect(const float3_simple &v, const float3_simple &n) { return v - 2 * dot(v, n) * n; }

__device__ inline float3_simple refract(const float3_simple &uv, const float3_simple &n, float etai_over_etat)
{
   float cos_theta = fminf(dot(-uv, n), 1.0f);
   float3_simple r_out_perp = etai_over_etat * (uv + cos_theta * n);
   float3_simple r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
   return r_out_perp + r_out_parallel;
}

__device__ inline float reflectance(float cosine, float ref_idx)
{
   float r0 = (1 - ref_idx) / (1 + ref_idx);
   r0 = r0 * r0;
   return r0 + (1 - r0) * powf((1 - cosine), 5);
}

__device__ inline float3_simple reflect_fuzzy(const float3_simple &v, const float3_simple &n, float roughness,
                                              curandState *state)
{
   float3_simple random_in_sphere;
   do
   {
      random_in_sphere = 2.0f * float3_simple(rand_float(state), rand_float(state), rand_float(state)) -
                         float3_simple(1.0f, 1.0f, 1.0f);
   } while (random_in_sphere.length_squared() >= 1.0f);

   float3_simple perturbed_normal = unit_vector(n + roughness * random_in_sphere);
   return reflect(v, perturbed_normal);
}

__device__ inline float3_simple random_in_hemisphere(const float3_simple &normal, curandState *state)
{
   float3_simple in_unit_sphere;
   do
   {
      in_unit_sphere = 2.0f * float3_simple(rand_float(state), rand_float(state), rand_float(state)) -
                       float3_simple(1.0f, 1.0f, 1.0f);
   } while (in_unit_sphere.length_squared() >= 1.0f);

   if (dot(in_unit_sphere, normal) > 0.0f)
      return in_unit_sphere;
   else
      return float3_simple(-in_unit_sphere.x, -in_unit_sphere.y, -in_unit_sphere.z);
}

__device__ inline float smoothstep(float edge0, float edge1, float x)
{
   float t = fmaxf(0.0f, fminf(1.0f, (x - edge0) / (edge1 - edge0)));
   return t * t * (3.0f - 2.0f * t);
}

//==============================================================================
// INTERSECTIONS AND PROCEDURAL UTILS
//==============================================================================

__device__ inline bool hit_sphere(const float3_simple &center, float radius, const ray_simple &r, float t_min,
                                  float t_max, hit_record_simple &rec)
{
   float3_simple oc = r.orig - center;
   float a = dot(r.dir, r.dir);
   float half_b = dot(oc, r.dir);
   float c = dot(oc, oc) - radius * radius;
   float discriminant = half_b * half_b - a * c;
   if (discriminant < 0)
      return false;
   float sqrtd = sqrtf(discriminant);
   float root = (-half_b - sqrtd) / a;
   if (root < t_min || t_max < root)
   {
      root = (-half_b + sqrtd) / a;
      if (root < t_min || t_max < root)
         return false;
   }
   rec.t = root;
   rec.p = r.at(rec.t);
   float3_simple outward_normal = (rec.p - center) / radius;
   rec.front_face = dot(r.dir, outward_normal) < 0;
   rec.normal = rec.front_face ? outward_normal : float3_simple(-outward_normal.x, -outward_normal.y, -outward_normal.z);
   return true;
}

__device__ inline bool hit_rectangle(const float3_simple &corner, const float3_simple &u, const float3_simple &v,
                                     const ray_simple &r, float t_min, float t_max, hit_record_simple &rec)
{
   float3_simple normal = unit_vector(float3_simple(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x));
   float denom = dot(normal, r.dir);
   if (fabsf(denom) < 1e-8f)
      return false;
   float t = dot(normal, corner - r.orig) / denom;
   if (t < t_min || t > t_max)
      return false;
   float3_simple intersection = r.at(t);
   float3_simple p = intersection - corner;
   float alpha = dot(p, u) / dot(u, u);
   float beta = dot(p, v) / dot(v, v);
   if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f)
      return false;
   rec.t = t;
   rec.p = intersection;
   rec.front_face = dot(r.dir, normal) < 0;
   rec.normal = rec.front_face ? normal : float3_simple(-normal.x, -normal.y, -normal.z);
   return true;
}


//==============================================================================
// SCENE & MATERIAL APPLICATION
//==============================================================================

__device__ __forceinline__ bool intersect_geometry(const CudaScene::Geometry &geom, const ray_simple &r, float t_min,
                                                   float t_max, hit_record_simple &rec)
{
   using namespace CudaScene;
   switch (geom.type)
   {
   case GeometryType::SPHERE:
      return hit_sphere(geom.data.sphere.center, geom.data.sphere.radius, r, t_min, t_max, rec);
   case GeometryType::RECTANGLE:
      return hit_rectangle(geom.data.rectangle.corner, geom.data.rectangle.u, geom.data.rectangle.v, r, t_min, t_max, rec);
   case GeometryType::DISPLACED_SPHERE:
      return hit_golf_ball_sphere(geom.data.displaced_sphere.center, geom.data.displaced_sphere.radius, r, t_min, t_max, rec);
   default:
      return false;
   }
}

__device__ inline float nearestAngularDistanceFibonacci(float3_simple dir, int N)
{
   float3_simple q = unit_vector(dir);
   float max_dp = -1.0f;
   for (int i = 0; i < N; ++i)
   {
      float3_simple c = fibonacci_point(i, N);
      float d = dot(q, c);
      if (d > max_dp)
         max_dp = d;
   }
   max_dp = fmaxf(fminf(max_dp, 1.0f), -1.0f);
   return acosf(max_dp);
}


__device__ inline float3_simple apply_procedural_pattern(CudaScene::ProceduralPattern pattern, const float3_simple &base_color,
                                                         const float3_simple &pattern_color, float param1, float param2,
                                                         const float3_simple &surface_point, const float3_simple &geometry_center)
{
   using namespace CudaScene;
   switch (pattern)
   {
   case ProceduralPattern::FIBONACCI_DOTS:
   {
      float3_simple local = float3_simple(surface_point.x - geometry_center.x, surface_point.y - geometry_center.y,
                                          surface_point.z - geometry_center.z);
      float3_simple dir = unit_vector(local);
      int dot_count = static_cast<int>(param1);
      float dot_radius = param2;
      float ang = nearestAngularDistanceFibonacci(dir, dot_count);
      float mask = ang < dot_radius ? 0.0f : 1.0f;
      return float3_simple(base_color.x * mask + pattern_color.x * (1.0f - mask),
                           base_color.y * mask + pattern_color.y * (1.0f - mask),
                           base_color.z * mask + pattern_color.z * (1.0f - mask));
   }
   case ProceduralPattern::NONE:
   default:
      return base_color;
   }
}

__device__ __forceinline__ void apply_material(const CudaScene::Material &mat, hit_record_simple &rec,
                                               const float3_simple &geometry_center)
{
   using namespace CudaScene;
   switch (mat.type)
   {
   case MaterialType::LAMBERTIAN:
      rec.material = LAMBERTIAN; rec.color = mat.albedo; break;
   case MaterialType::METAL:
   case MaterialType::MIRROR:
      rec.material = MIRROR; rec.color = mat.albedo; break;
   case MaterialType::ROUGH_MIRROR:
      rec.material = ROUGH_MIRROR; rec.color = mat.albedo; rec.roughness = mat.roughness; break;
   case MaterialType::GLASS:
   case MaterialType::DIELECTRIC:
      rec.material = GLASS; rec.refractive_index = mat.refractive_index; break;
   case MaterialType::LIGHT:
      rec.material = LIGHT; rec.emission = mat.emission; break;
   case MaterialType::CONSTANT:
      rec.material = LAMBERTIAN; rec.color = mat.albedo; break;
   case MaterialType::SHOW_NORMALS:
      rec.material = LAMBERTIAN; rec.color = float3_simple(1, 1, 1); break;
   case MaterialType::SDF_MATERIAL:
      rec.material = LAMBERTIAN; rec.color = mat.albedo; break;
   }
   if (mat.pattern != CudaScene::ProceduralPattern::NONE)
   {
      rec.color = apply_procedural_pattern(mat.pattern, rec.color, mat.pattern_color, mat.pattern_param1,
                                           mat.pattern_param2, rec.p, geometry_center);
   }
}

__device__ inline bool hit_scene(const CudaScene::Scene &scene, const ray_simple &r, float t_min, float t_max,
                                 hit_record_simple &rec)
{
   hit_record_simple temp_rec;
   bool hit_anything = false;
   float closest_so_far = t_max;
   int closest_material_id = -1;
   int closest_geom_idx = -1;

#pragma unroll 4
   for (int i = 0; i < scene.num_geometries; ++i)
   {
      const CudaScene::Geometry &geom = scene.geometries[i];
      if (intersect_geometry(geom, r, t_min, closest_so_far, temp_rec))
      {
         hit_anything = true;
         closest_so_far = temp_rec.t;
         rec = temp_rec;
         closest_material_id = geom.material_id;
         closest_geom_idx = i;
      }
   }
   if (hit_anything && closest_material_id >= 0 && closest_material_id < scene.num_materials)
   {
      float3_simple geom_center(0, 0, 0);
      if (closest_geom_idx >= 0)
      {
         const CudaScene::Geometry &geom = scene.geometries[closest_geom_idx];
         if (geom.type == CudaScene::GeometryType::SPHERE || geom.type == CudaScene::GeometryType::DISPLACED_SPHERE)
         {
            geom_center = geom.data.sphere.center;
         }
      }
      apply_material(scene.materials[closest_material_id], rec, geom_center);
   }
   return hit_anything;
}

__device__ inline float3_simple ray_color(const ray_simple &r, const CudaScene::Scene &scene, curandState *state,
                                          int depth, int &local_ray_count)
{
   float3_simple accumulated_color(0, 0, 0);
   float3_simple accumulated_attenuation(1.0f, 1.0f, 1.0f);
   ray_simple current_ray = r;

   for (int bounce = 0; bounce < depth; bounce++)
   {
      local_ray_count++;
      hit_record_simple rec;
      if (hit_scene(scene, current_ray, 0.001f, FLT_MAX, rec))
      {
         if (rec.material == LIGHT)
         {
            accumulated_color =
                accumulated_color + float3_simple(accumulated_attenuation.x * rec.emission.x * g_light_intensity,
                                                  accumulated_attenuation.y * rec.emission.y * g_light_intensity,
                                                  accumulated_attenuation.z * rec.emission.z * g_light_intensity);
            return accumulated_color;
         }
         float3_simple scattered_direction;
         float3_simple attenuation;
         if (rec.material == LAMBERTIAN)
         {
            float3_simple target = rec.p + rec.normal + random_in_hemisphere(rec.normal, state);
            scattered_direction = target - rec.p;
            attenuation = rec.color;
         }
         else if (rec.material == MIRROR)
         {
            scattered_direction = reflect(unit_vector(current_ray.dir), rec.normal);
            attenuation = rec.color;
         }
         else if (rec.material == ROUGH_MIRROR)
         {
            scattered_direction = reflect_fuzzy(unit_vector(current_ray.dir), rec.normal, rec.roughness * g_metal_fuzziness, state);
            attenuation = rec.color;
         }
         else if (rec.material == GLASS)
         {
            float3_simple unit_direction = unit_vector(current_ray.dir);
            float refraction_ratio = rec.front_face ? (1.0f / rec.refractive_index) : rec.refractive_index;
            float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
            float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
            bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > rand_float(state))
            {
               scattered_direction = reflect(unit_direction, rec.normal);
            }
            else
            {
               scattered_direction = refract(unit_direction, rec.normal, refraction_ratio);
            }
            attenuation = float3_simple(1.0f, 1.0f, 1.0f);
         }
         else
         {
            return accumulated_color;
         }
         current_ray = ray_simple(rec.p, scattered_direction);
         accumulated_attenuation = float3_simple(accumulated_attenuation.x * attenuation.x,
                                                 accumulated_attenuation.y * attenuation.y,
                                                 accumulated_attenuation.z * attenuation.z);
      }
      else
      {
         float3_simple unit_direction = unit_vector(current_ray.dir);
         float t = 0.5f * (unit_direction.y + 1.0f);
         float3_simple sky_color = (1.0f - t) * float3_simple(1.0f, 1.0f, 1.0f) + t * float3_simple(0.5f, 0.7f, 1.0f);
         accumulated_color = accumulated_color + float3_simple(accumulated_attenuation.x * sky_color.x * g_background_intensity,
                                                               accumulated_attenuation.y * sky_color.y * g_background_intensity,
                                                               accumulated_attenuation.z * sky_color.z * g_background_intensity);
         return accumulated_color;
      }
   }
   return accumulated_color;
}
