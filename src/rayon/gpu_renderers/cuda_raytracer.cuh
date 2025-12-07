// Common device-side utilities and shading routines shared by CUDA shaders
#pragma once
#include "cuda_float3.cuh"
#include "cuda_scene.cuh"
#include "cuda_utils.cuh"

#include <cfloat>
#include <cmath>
#include <curand_kernel.h>

// Extern declarations for device-side global constants (defined once in renderer_cuda_device.cu)
extern __constant__ float g_light_intensity;
extern __constant__ float g_background_intensity;
extern __constant__ float g_metal_fuzziness;
extern __constant__ float g_glass_refraction_index;

// Depth of Field parameters
extern __constant__ bool g_dof_enabled;
extern __constant__ float g_dof_aperture;
extern __constant__ float g_dof_focus_distance;

// Forward declarations for golf-ball helpers implemented in shader_golf.cu
struct ray_simple;
struct hit_record_simple;
__device__ f3 fibonacci_point(int i, int n);
__device__ bool hit_golf_ball_sphere(f3 center, float radius, const ray_simple &r, float t_min, float t_max,
                                     hit_record_simple &rec);

//==============================================================================
// RAY TRACING DATA STRUCTURES
//==============================================================================

struct ray_simple
{
   f3 orig, dir;
   __device__ ray_simple() {}
   __device__ ray_simple(const f3 &origin, const f3 &direction) : orig(origin), dir(direction) {}
   __device__ f3 at(float t) const { return orig + t * dir; }
};

enum LegacyMaterialType
{
   LAMBERTIAN = 0,
   MIRROR = 1,
   GLASS = 2,
   LIGHT = 3,
   ROUGH_MIRROR = 4,
   CONSTANT = 5,
   SHOW_NORMALS = 6
};

struct hit_record_simple
{
   f3 p, normal;
   float t;
   bool front_face;
   LegacyMaterialType material;
   f3 color;
   float refractive_index;
   f3 emission;
   float roughness;
};

#include "materials/material_dispatcher.cuh"

//==============================================================================
// OPTICAL PHYSICS FUNCTIONS
//==============================================================================

__device__ __forceinline__ f3 reflect(const f3 &v, const f3 &n) { return v - 2 * dot(v, n) * n; }

__device__ __forceinline__ f3 refract(const f3 &uv, const f3 &n, float etai_over_etat)
{
   float cos_theta = fminf(dot(-uv, n), 1.0f);
   f3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
   f3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
   return r_out_perp + r_out_parallel;
}

__device__ __forceinline__ float reflectance(float cosine, float ref_idx)
{
   float r0 = (1 - ref_idx) / (1 + ref_idx);
   r0 = r0 * r0;
   return r0 + (1 - r0) * powf((1 - cosine), 5);
}

__device__ __forceinline__ f3 reflect_fuzzy(const f3 &v, const f3 &n, float roughness, curandState *state)
{
   f3 perturbed_normal = normalize(n + roughness * randOnUnitSphere(state));
   return reflect(v, perturbed_normal);
}

__device__ inline float smoothstep(float edge0, float edge1, float x)
{
   float t = fmaxf(0.0f, fminf(1.0f, (x - edge0) / (edge1 - edge0)));
   return t * t * (3.0f - 2.0f * t);
}

/**
 * @brief Generate a random point in the unit disk for DOF
 * @param state Random state
 * @return Random 2D point in unit disk
 */
__device__ inline f2 random_in_unit_disk(curandState *state)
{
   f2 p;
   do
   {
      p = 2.0f * f2(rand_float(state), rand_float(state)) - f2(1.0f, 1.0f);
   } while (p.x * p.x + p.y * p.y >= 1.0f);
   return p;
}

/**
 * @brief Sample a point on the aperture disk for DOF
 * @param cam_u Camera u basis vector
 * @param cam_v Camera v basis vector
 * @param state Random state
 * @return Offset on aperture disk
 */
__device__ inline f3 sample_aperture_disk(const f3 &cam_u, const f3 &cam_v, curandState *state)
{
   f2 disk = random_in_unit_disk(state);
   return g_dof_aperture * (disk.x * cam_u + disk.y * cam_v);
}

//==============================================================================
// BVH / AABB INTERSECTION
//==============================================================================

/**
 * @brief Ray-AABB intersection test using slab method
 * @param r Ray to test
 * @param box_min AABB minimum corner
 * @param box_max AABB maximum corner
 * @param t_min Minimum ray parameter
 * @param t_max Maximum ray parameter
 * @return true if ray intersects AABB in range [t_min, t_max]
 */
__device__ inline bool hit_aabb(const ray_simple &r, const f3 &box_min, const f3 &box_max, float t_min, float t_max)
{
   // Compute inverse ray direction once
   float inv_dir_x = 1.0f / r.dir.x;
   float inv_dir_y = 1.0f / r.dir.y;
   float inv_dir_z = 1.0f / r.dir.z;

   // X slab
   float t0_x = (box_min.x - r.orig.x) * inv_dir_x;
   float t1_x = (box_max.x - r.orig.x) * inv_dir_x;
   if (inv_dir_x < 0.0f)
   {
      float temp = t0_x;
      t0_x = t1_x;
      t1_x = temp;
   }

   // Y slab
   float t0_y = (box_min.y - r.orig.y) * inv_dir_y;
   float t1_y = (box_max.y - r.orig.y) * inv_dir_y;
   if (inv_dir_y < 0.0f)
   {
      float temp = t0_y;
      t0_y = t1_y;
      t1_y = temp;
   }

   // Z slab
   float t0_z = (box_min.z - r.orig.z) * inv_dir_z;
   float t1_z = (box_max.z - r.orig.z) * inv_dir_z;
   if (inv_dir_z < 0.0f)
   {
      float temp = t0_z;
      t0_z = t1_z;
      t1_z = temp;
   }

   // Compute intersection interval
   float t_enter = fmaxf(fmaxf(t0_x, t0_y), t0_z);
   float t_exit = fminf(fminf(t1_x, t1_y), t1_z);

   // Check if ray intersects AABB
   return t_enter <= t_exit && t_exit >= t_min && t_enter <= t_max;
}

//==============================================================================
// INTERSECTIONS AND PROCEDURAL UTILS
//==============================================================================

__device__ inline bool hit_sphere(const f3 &center, float radius, const ray_simple &r, float t_min, float t_max,
                                  hit_record_simple &rec)
{
   f3 oc = r.orig - center;
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
   f3 outward_normal = (rec.p - center) / radius;
   rec.front_face = dot(r.dir, outward_normal) < 0;
   rec.normal = rec.front_face ? outward_normal : f3(-outward_normal.x, -outward_normal.y, -outward_normal.z);
   return true;
}

__device__ inline bool hit_rectangle(const f3 &corner, const f3 &u, const f3 &v, const ray_simple &r, float t_min,
                                     float t_max, hit_record_simple &rec)
{
   // Compute rectangle normal via cross product (u × v)
   f3 normal = normalize(f3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x));

   // Check if ray is parallel to rectangle plane
   float denom = dot(normal, r.dir);
   if (fabsf(denom) < 1e-8f)
      return false;

   // Compute ray parameter t at plane intersection
   float t = dot(normal, corner - r.orig) / denom;
   if (t < t_min || t > t_max)
      return false;

   // Find intersection point and convert to rectangle's local coordinates
   f3 intersection = r.at(t);
   f3 p = intersection - corner;

   // Project onto u and v vectors to get parametric coordinates
   float alpha = dot(p, u) / dot(u, u);
   float beta = dot(p, v) / dot(v, v);

   // Check if intersection is within rectangle bounds [0,1] × [0,1]
   if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f)
      return false;

   // Fill hit record with intersection data
   rec.t = t;
   rec.p = intersection;
   rec.front_face = dot(r.dir, normal) < 0;
   rec.normal = rec.front_face ? normal : f3(-normal.x, -normal.y, -normal.z);
   return true;
}

__device__ inline bool hit_triangle(const f3 &v0, const f3 &v1, const f3 &v2, const f3 &n0, const f3 &n1, const f3 &n2,
                                    bool has_normals, const ray_simple &r, float t_min, float t_max,
                                    hit_record_simple &rec)
{
   // Möller–Trumbore intersection algorithm (or barrycentric method ;)
   f3 e1 = v1 - v0;
   f3 e2 = v2 - v0;

   
   f3 p = cross(r.dir, e2);

   
   float det = dot(e1, p);

   // If det is near zero, ray is parallel to triangle - property of the determinant
   if (fabsf(det) < 1e-8f)
      return false;

   float inv_det = 1.0f / det;

   f3 t_vec = r.orig - v0;

   float u = dot(t_vec, p) * inv_det;
   if (u < 0.0f || u > 1.0f)
      return false;

 
   f3 q = cross(t_vec, e1);


   float v = dot(r.dir, q) * inv_det;
   if (v < 0.0f || u + v > 1.0f)
      return false;

   float t = dot(e2, q) * inv_det;

   if (t < t_min || t > t_max)
      return false;

   rec.t = t;
   rec.p = r.at(t);

   f3 outward_normal;
   if (has_normals)
   {
      // Barycentric interpolation of normals
      float w = 1.0f - u - v;
      outward_normal = normalize(w * n0 + u * n1 + v * n2);
   }
   else
   {
      // Flat normal: normalize(E1 x E2)
      outward_normal = normalize(cross(e1, e2));
   }

   rec.front_face = dot(r.dir, outward_normal) < 0;
   rec.normal = rec.front_face ? outward_normal : -outward_normal;
   return true;
}

//==============================================================================
// SCENE & MATERIAL APPLICATION
//==============================================================================

__device__ inline f3 rotate_vector(f3 v, f3 angles)
{
   // Rotation order: X, then Y, then Z
   if (angles.x != 0.0f) { float c = cosf(angles.x), s = sinf(angles.x); f3 t = v; v.y = t.y*c - t.z*s; v.z = t.y*s + t.z*c; }
   if (angles.y != 0.0f) { float c = cosf(angles.y), s = sinf(angles.y); f3 t = v; v.x = t.x*c + t.z*s; v.z = -t.x*s + t.z*c; }
   if (angles.z != 0.0f) { float c = cosf(angles.z), s = sinf(angles.z); f3 t = v; v.x = t.x*c - t.y*s; v.y = t.x*s + t.y*c; }
   return v;
}

__device__ inline f3 rotate_vector_inverse(f3 v, f3 angles)
{
   // Inverse order: Z, then Y, then X (negative angles)
   if (angles.z != 0.0f) { float c = cosf(-angles.z), s = sinf(-angles.z); f3 t = v; v.x = t.x*c - t.y*s; v.y = t.x*s + t.y*c; }
   if (angles.y != 0.0f) { float c = cosf(-angles.y), s = sinf(-angles.y); f3 t = v; v.x = t.x*c + t.z*s; v.z = -t.x*s + t.z*c; }
   if (angles.x != 0.0f) { float c = cosf(-angles.x), s = sinf(-angles.x); f3 t = v; v.y = t.y*c - t.z*s; v.z = t.y*s + t.z*c; }
   return v;
}

__device__ inline bool hit_mesh_triangle(const CudaScene::MeshTriangle &tri, const ray_simple &r, float t_min, float t_max, hit_record_simple &rec)
{
   return hit_triangle(tri.v0, tri.v1, tri.v2, tri.n0, tri.n1, tri.n2, tri.has_normals, r, t_min, t_max, rec);
}

__device__ inline bool hit_mesh(const CudaScene::Mesh &mesh, const f3 &translation, const f3 &rotation, const f3 &scale,
                                const ray_simple &r_world, float t_min, float t_max, hit_record_simple &rec)
{
   f3 orig_local = (r_world.orig - translation);
   f3 orig_rot = rotate_vector_inverse(orig_local, rotation);
   orig_local = f3(orig_rot.x / scale.x, orig_rot.y / scale.y, orig_rot.z / scale.z);
   
   f3 dir_rot = rotate_vector_inverse(r_world.dir, rotation);
   f3 dir_local = f3(dir_rot.x / scale.x, dir_rot.y / scale.y, dir_rot.z / scale.z);
   
   ray_simple r_local(orig_local, dir_local);

   hit_record_simple temp_rec;
   bool hit_anything = false;
   float closest_so_far = t_max;

   if (mesh.bvh_nodes && mesh.bvh_root_idx >= 0)
   {
       int stack[32];
       int stack_ptr = 0;
       stack[stack_ptr++] = mesh.bvh_root_idx;
       while (stack_ptr > 0)
       {
           int node_idx = stack[--stack_ptr];
           const CudaScene::BVHNode &node = mesh.bvh_nodes[node_idx];
           if (!hit_aabb(r_local, node.bounds_min, node.bounds_max, t_min, closest_so_far)) continue;

           if (node.is_leaf)
           {
               int first = node.data.leaf.first_geom_idx;
               int count = node.data.leaf.geom_count;
               for (int i = 0; i < count; ++i)
               {
                   if (hit_mesh_triangle(mesh.triangles[first + i], r_local, t_min, closest_so_far, temp_rec))
                   {
                       hit_anything = true;
                       closest_so_far = temp_rec.t;
                       rec = temp_rec;
                   }
               }
           }
           else
           {
               int left = node.data.interior.left_child;
               int right = node.data.interior.right_child;
               if (stack_ptr < 32) stack[stack_ptr++] = right;
               if (stack_ptr < 32) stack[stack_ptr++] = left;
           }
       }
   }
   else
   {
       for (int i = 0; i < mesh.num_triangles; ++i)
       {
           if (hit_mesh_triangle(mesh.triangles[i], r_local, t_min, closest_so_far, temp_rec))
           {
               hit_anything = true;
               closest_so_far = temp_rec.t;
               rec = temp_rec;
           }
       }
   }

   if (hit_anything)
   {
       rec.p = r_world.at(rec.t); // Use t directly (valid because we scaled direction)
       f3 n_local = f3(rec.normal.x / scale.x, rec.normal.y / scale.y, rec.normal.z / scale.z);
       rec.normal = normalize(rotate_vector(n_local, rotation));
       if (rec.front_face) rec.normal = dot(r_world.dir, rec.normal) < 0 ? rec.normal : -rec.normal;
       else rec.normal = dot(r_world.dir, rec.normal) < 0 ? rec.normal : -rec.normal;
   }
   return hit_anything;
}

__device__ __forceinline__ bool intersect_geometry(const CudaScene::Scene &scene, const CudaScene::Geometry &geom, const ray_simple &r, float t_min,
                                                   float t_max, hit_record_simple &rec)
{
   using namespace CudaScene;
   switch (geom.type)
   {
   case GeometryType::SPHERE:
      return hit_sphere(geom.data.sphere.center, geom.data.sphere.radius, r, t_min, t_max, rec);
   case GeometryType::RECTANGLE:
      return hit_rectangle(geom.data.rectangle.corner, geom.data.rectangle.u, geom.data.rectangle.v, r, t_min, t_max,
                           rec);
   case GeometryType::TRIANGLE:
      return hit_triangle(geom.data.triangle.v0, geom.data.triangle.v1, geom.data.triangle.v2, geom.data.triangle.n0,
                          geom.data.triangle.n1, geom.data.triangle.n2, geom.data.triangle.has_normals, r, t_min, t_max,
                          rec);
   case GeometryType::DISPLACED_SPHERE:
      return hit_golf_ball_sphere(geom.data.displaced_sphere.center, geom.data.displaced_sphere.radius, r, t_min, t_max,
                                  rec);
   case GeometryType::TRIANGLE_MESH:
      if (geom.data.mesh_instance.mesh_id >= 0 && geom.data.mesh_instance.mesh_id < scene.num_meshes)
      {
          return hit_mesh(scene.meshes[geom.data.mesh_instance.mesh_id], 
                          geom.data.mesh_instance.translation,
                          geom.data.mesh_instance.rotation,
                          geom.data.mesh_instance.scale,
                          r, t_min, t_max, rec);
      }
      return false;
   default:
      return false;
   }
}

__device__ inline float nearestAngularDistanceFibonacci(f3 dir, int N)
{
   f3 q = normalize(dir);
   float max_dp = -1.0f;
   for (int i = 0; i < N; ++i)
   {
      f3 c = fibonacci_point(i, N);
      float d = dot(q, c);
      if (d > max_dp)
         max_dp = d;
   }
   max_dp = fmaxf(fminf(max_dp, 1.0f), -1.0f);
   return acosf(max_dp);
}

__device__ inline f3 apply_procedural_pattern(CudaScene::ProceduralPattern pattern, const f3 &base_color,
                                              const f3 &pattern_color, float param1, float param2,
                                              const f3 &surface_point, const f3 &geometry_center)
{
   using namespace CudaScene;
   switch (pattern)
   {
   case ProceduralPattern::FIBONACCI_DOTS:
   {
      f3 local = f3(surface_point.x - geometry_center.x, surface_point.y - geometry_center.y,
                    surface_point.z - geometry_center.z);
      f3 dir = normalize(local);
      int dot_count = static_cast<int>(param1);
      float dot_radius = param2;
      float ang = nearestAngularDistanceFibonacci(dir, dot_count);
      float mask = ang < dot_radius ? 0.0f : 1.0f;
      return f3(base_color.x * mask + pattern_color.x * (1.0f - mask),
                base_color.y * mask + pattern_color.y * (1.0f - mask),
                base_color.z * mask + pattern_color.z * (1.0f - mask));
   }
   case ProceduralPattern::NONE:
   default:
      return base_color;
   }
}

__device__ __forceinline__ void apply_material(const CudaScene::Material &mat, hit_record_simple &rec,
                                               const f3 &geometry_center)
{
   using namespace CudaScene;
   switch (mat.type)
   {
   case MaterialType::LAMBERTIAN:
      rec.material = LAMBERTIAN;
      rec.color = mat.albedo;
      break;
   case MaterialType::METAL:
   case MaterialType::MIRROR:
      rec.material = MIRROR;
      rec.color = mat.albedo;
      break;
   case MaterialType::ROUGH_MIRROR:
      rec.material = ROUGH_MIRROR;
      rec.color = mat.albedo;
      rec.roughness = mat.roughness;
      break;
   case MaterialType::GLASS:
   case MaterialType::DIELECTRIC:
      rec.material = GLASS;
      rec.refractive_index = mat.refractive_index;
      break;
   case MaterialType::LIGHT:
      rec.material = LIGHT;
      rec.emission = mat.emission;
      break;
   case MaterialType::CONSTANT:
      rec.material = CONSTANT;
      rec.color = mat.albedo;
      break;
   case MaterialType::SHOW_NORMALS:
      rec.material = SHOW_NORMALS;
      break;
   case MaterialType::SDF_MATERIAL: // TODO: Implement SDF materials
      rec.material = LAMBERTIAN;
      rec.color = mat.albedo;
      break;
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

   // Use BVH if available, otherwise linear scan
   if (scene.use_bvh && scene.bvh_root_idx >= 0)
   {
      // Stack-based BVH traversal (iterative to avoid recursion)
      int stack[32];
      int stack_ptr = 0;
      stack[stack_ptr++] = scene.bvh_root_idx;

      while (stack_ptr > 0)
      {
         int node_idx = stack[--stack_ptr];
         const CudaScene::BVHNode &node = scene.bvh_nodes[node_idx];

         // Test ray against node's AABB
         if (!hit_aabb(r, node.bounds_min, node.bounds_max, t_min, closest_so_far))
            continue;

         if (node.is_leaf)
         {
            // Leaf node: test all geometries
            int first = node.data.leaf.first_geom_idx;
            int count = node.data.leaf.geom_count;

            for (int i = 0; i < count; ++i)
            {
               const CudaScene::Geometry &geom = scene.geometries[first + i];
               if (intersect_geometry(scene, geom, r, t_min, closest_so_far, temp_rec))
               {
                  hit_anything = true;
                  closest_so_far = temp_rec.t;
                  rec = temp_rec;
                  closest_material_id = geom.material_id;
                  closest_geom_idx = first + i;
               }
            }
         }
         else
         {
            // Interior node: push children onto stack
            // Push farther child first for better traversal order
            int left_child = node.data.interior.left_child;
            int right_child = node.data.interior.right_child;

            // Simple heuristic: test which child is closer
            f3 left_center = (scene.bvh_nodes[left_child].bounds_min + scene.bvh_nodes[left_child].bounds_max) * 0.5f;
            f3 right_center =
                (scene.bvh_nodes[right_child].bounds_min + scene.bvh_nodes[right_child].bounds_max) * 0.5f;

            float dist_left = (left_center - r.orig).length_squared();
            float dist_right = (right_center - r.orig).length_squared();

            if (dist_left < dist_right)
            {
               // Right is farther, push it first
               if (stack_ptr < 32)
                  stack[stack_ptr++] = right_child;
               if (stack_ptr < 32)
                  stack[stack_ptr++] = left_child;
            }
            else
            {
               // Left is farther, push it first
               if (stack_ptr < 32)
                  stack[stack_ptr++] = left_child;
               if (stack_ptr < 32)
                  stack[stack_ptr++] = right_child;
            }
         }
      }
   }
   else
   {
      // Linear scan fallback
#pragma unroll 4
      for (int i = 0; i < scene.num_geometries; ++i)
      {
         const CudaScene::Geometry &geom = scene.geometries[i];
         if (intersect_geometry(scene, geom, r, t_min, closest_so_far, temp_rec))
         {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
            closest_material_id = geom.material_id;
            closest_geom_idx = i;
         }
      }
   }

   if (hit_anything && closest_material_id >= 0 && closest_material_id < scene.num_materials)
   {
      f3 geom_center(0, 0, 0);
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

/**
 * @brief Ray color computation using new material system
 *
 * This version uses compile-time material dispatch via CRTP templates.
 * The compiler generates optimized code for each material type with zero overhead.
 */
__device__ inline f3 ray_color(const ray_simple &r, const CudaScene::Scene &scene, curandState *state, int depth,
                               int &local_ray_count)
{
   using namespace Materials;

   f3 accumulated_color(0.0f, 0.0f, 0.0f);
   f3 accumulated_attenuation(1.0f, 1.0f, 1.0f);
   ray_simple current_ray = r;

   for (int bounce = 0; bounce < depth; bounce++)
   {
      local_ray_count++;
      hit_record_simple rec;

      if (hit_scene(scene, current_ray, 0.001f, FLT_MAX, rec))
      {
         // Create material descriptor from hit record
         // TODO: This is a temporary adapter - ideally hit_scene would return MaterialDescriptor directly
         MaterialDescriptor mat_desc;

         switch (rec.material)
         {
         case LAMBERTIAN:
            mat_desc = MaterialDescriptor::makeLambertian(rec.color);
            break;
         case MIRROR:
            mat_desc = MaterialDescriptor::makeMirror(rec.color);
            break;
         case ROUGH_MIRROR:
            mat_desc = MaterialDescriptor::makeRoughMirror(rec.color, rec.roughness);
            break;
         case GLASS:
            mat_desc = MaterialDescriptor::makeGlass(rec.refractive_index);
            break;
         case LIGHT:
            mat_desc = MaterialDescriptor::makeLight(rec.emission);
            break;
         case CONSTANT:
            mat_desc = MaterialDescriptor::makeConstant(rec.color);
            break;
         case SHOW_NORMALS:
            mat_desc = MaterialDescriptor::makeShowNormals(rec.normal);
            break;
         }

         // Dispatch to appropriate material using compile-time template magic
         bool scattered = dispatch_material_bool(
             mat_desc,
             [&](auto material) -> bool
             {
                // Check if emissive first
                f3 emitted = material.emission();
                if (emitted.length_squared() > 0.0f)
                {
                   accumulated_color = accumulated_color + f3(accumulated_attenuation.x * emitted.x,
                                                              accumulated_attenuation.y * emitted.y,
                                                              accumulated_attenuation.z * emitted.z);
                   return false; // Light materials don't scatter
                }

                // Scatter the ray
                f3 attenuation;
                ray_simple scattered_ray;
                if (material.scatter(current_ray, rec, attenuation, scattered_ray, state))
                {
                   current_ray = scattered_ray;
                   accumulated_attenuation =
                       f3(accumulated_attenuation.x * attenuation.x, accumulated_attenuation.y * attenuation.y,
                          accumulated_attenuation.z * attenuation.z);
                   return true;
                }
                return false;
             });

         if (!scattered)
         {
            return accumulated_color;
         }

         // Russian Roulette path termination (after minimum bounces)
         if (bounce > 3)
         {
            float max_component =
                fmaxf(accumulated_attenuation.x, fmaxf(accumulated_attenuation.y, accumulated_attenuation.z));
            float survival_prob = fminf(max_component, 0.95f);

            if (rand_float(state) > survival_prob)
            {
               return accumulated_color;
            }

            accumulated_attenuation = accumulated_attenuation / survival_prob;
         }
      }
      else
      {
         // Sky/background
         f3 unit_direction = normalize(current_ray.dir);
         float t = 0.5f * (unit_direction.y + 1.0f);
         f3 sky_color = (1.0f - t) * f3(1.0f, 1.0f, 1.0f) + t * f3(0.5f, 0.7f, 1.0f);
         accumulated_color = accumulated_color + accumulated_attenuation * sky_color * g_background_intensity;
         return accumulated_color;
      }
   }
   return accumulated_color;
}