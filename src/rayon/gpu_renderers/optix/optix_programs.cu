// OptiX device programs: raygen, miss, closest-hit, intersection
// Compiled to PTX at build time, loaded by host at runtime.

#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "optix_params.h"

// Launch parameters in constant memory
extern "C"
{
   __constant__ OptixLaunchParams params;
}

//==============================================================================
// RANDOM NUMBER GENERATION (PCG-based, same as CUDA renderer)
//==============================================================================

__device__ __forceinline__ unsigned int pcg_hash(unsigned int input)
{
   unsigned int state = input * 747796405u + 2891336453u;
   unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
   return (word >> 22u) ^ word;
}

__device__ __forceinline__ float rand_float(unsigned int &seed)
{
   seed = pcg_hash(seed);
   return (float)seed / (float)0xFFFFFFFFu;
}

// Direct uniform unit vector — no rejection loop, no warp divergence.
// Uses spherical coordinates: z = 2*u-1, (x,y) from azimuth angle.
__device__ __forceinline__ float3 rand_unit_vector(unsigned int &seed)
{
   float u = rand_float(seed);
   float v = rand_float(seed);
   float z = 2.0f * u - 1.0f;
   float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
   float phi = 6.283185307f * v; // 2*PI
   return make_float3(r * __cosf(phi), r * __sinf(phi), z);
}

// Random point in unit sphere via direct method (used for rough mirror perturbation)
__device__ __forceinline__ float3 rand_unit_sphere(unsigned int &seed)
{
   float3 dir = rand_unit_vector(seed);
   float t = cbrtf(rand_float(seed)); // cube root for uniform volume distribution
   return make_float3(dir.x * t, dir.y * t, dir.z * t);
}

__device__ __forceinline__ float2 rand_in_unit_disk(unsigned int &seed)
{
   float2 p;
   do
   {
      p = make_float2(2.0f * rand_float(seed) - 1.0f, 2.0f * rand_float(seed) - 1.0f);
   } while (p.x * p.x + p.y * p.y >= 1.0f);
   return p;
}

//==============================================================================
// VECTOR HELPERS
//==============================================================================

__device__ __forceinline__ float3 operator+(const float3 &a, const float3 &b)
{
   return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float3 operator-(const float3 &a, const float3 &b)
{
   return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ float3 operator*(float s, const float3 &a) { return make_float3(s * a.x, s * a.y, s * a.z); }
__device__ __forceinline__ float3 operator*(const float3 &a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float3 operator*(const float3 &a, const float3 &b)
{
   return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__device__ __forceinline__ float3 operator-(const float3 &a) { return make_float3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ float3 operator/(const float3 &a, float s)
{
   return make_float3(a.x / s, a.y / s, a.z / s);
}

__device__ __forceinline__ float dot3(const float3 &a, const float3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__device__ __forceinline__ float length3(const float3 &a) { return sqrtf(dot3(a, a)); }

// Use rsqrtf (hardware intrinsic) — avoids separate sqrt + division
__device__ __forceinline__ float3 normalize3(const float3 &a)
{
   float len_sq = dot3(a, a);
   if (len_sq < 1e-16f)
      return make_float3(0, 0, 0);
   float inv_len = rsqrtf(len_sq);
   return make_float3(a.x * inv_len, a.y * inv_len, a.z * inv_len);
}

__device__ __forceinline__ float3 reflect3(const float3 &v, const float3 &n) { return v - 2.0f * dot3(v, n) * n; }

__device__ __forceinline__ float3 refract3(const float3 &uv, const float3 &n, float etai_over_etat)
{
   float cos_theta = fminf(dot3(-uv, n), 1.0f);
   float3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
   float3 r_out_parallel = -sqrtf(fabsf(1.0f - dot3(r_out_perp, r_out_perp))) * n;
   return r_out_perp + r_out_parallel;
}

// Schlick's approximation — manual power-of-5 expansion avoids powf overhead
__device__ __forceinline__ float reflectance(float cosine, float ref_idx)
{
   float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
   r0 = r0 * r0;
   float x = 1.0f - cosine;
   float x2 = x * x;
   return r0 + (1.0f - r0) * (x2 * x2 * x);
}

//==============================================================================
// PAYLOAD HELPERS — pass PRD pointer via 2 payload slots
//==============================================================================

__device__ __forceinline__ PRDRadiance *getPRD()
{
   unsigned int p0 = optixGetPayload_0();
   unsigned int p1 = optixGetPayload_1();
   unsigned long long ptr = (unsigned long long)p0 | ((unsigned long long)p1 << 32);
   return reinterpret_cast<PRDRadiance *>(ptr);
}

__device__ __forceinline__ void trace(OptixTraversableHandle handle, float3 origin, float3 direction, float tmin,
                                       float tmax, PRDRadiance *prd)
{
   unsigned int p0 = (unsigned int)((unsigned long long)prd);
   unsigned int p1 = (unsigned int)((unsigned long long)prd >> 32);
   optixTrace(handle, origin, direction, tmin, tmax,
              0.0f,                    // rayTime
              OptixVisibilityMask(1),  // visibilityMask
              OPTIX_RAY_FLAG_NONE,     // rayFlags
              0,                       // SBT offset
              1,                       // SBT stride
              0,                       // missSBTIndex
              p0, p1);
}

//==============================================================================
// RAY GENERATION — path tracing loop
//==============================================================================

extern "C" __global__ void __raygen__rg()
{
   const uint3 idx = optixGetLaunchIndex();
   const unsigned int x = idx.x;
   const unsigned int y = idx.y;

   if (x >= params.width || y >= params.height)
      return;

   const unsigned int pixel_idx = y * params.width + x;

   // Initialize seed based on pixel and frame
   unsigned int seed = pcg_hash(pixel_idx ^ (params.frame_seed * 1099511628211u));

   // Load accumulated color
   float4 acc = params.accum_buffer[pixel_idx];
   float3 accumulated = make_float3(acc.x, acc.y, acc.z);

   for (int s = 0; s < params.samples_per_launch; ++s)
   {
      // Jittered pixel sample
      float offset_u = rand_float(seed) - 0.5f;
      float offset_v = rand_float(seed) - 0.5f;

      float3 pixel_center = params.pixel00_loc + ((float)x + offset_u) * params.pixel_delta_u +
                             ((float)y + offset_v) * params.pixel_delta_v;
      float3 ray_direction = pixel_center - params.camera_center;
      float3 ray_origin = params.camera_center;

      // Depth of field
      if (params.dof_enabled && params.dof_aperture > 0.0f)
      {
         float3 normalized_dir = normalize3(ray_direction);
         float3 focus_point = params.camera_center + params.dof_focus_distance * normalized_dir;

         float2 disk = rand_in_unit_disk(seed);
         float3 aperture_offset = params.dof_aperture * (disk.x * params.cam_u + disk.y * params.cam_v);
         ray_origin = params.camera_center + aperture_offset;
         ray_direction = focus_point - ray_origin;
      }

      // Path tracing loop
      float3 color = make_float3(0.0f, 0.0f, 0.0f);
      float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

      float3 cur_origin = ray_origin;
      float3 cur_direction = ray_direction;

      for (int bounce = 0; bounce < params.max_depth; ++bounce)
      {
         PRDRadiance prd;
         prd.seed = seed;
         prd.hit = false;

         trace(params.traversable, cur_origin, cur_direction, 0.001f, 1e16f, &prd);
         seed = prd.seed; // Propagate RNG state

         if (!prd.hit)
         {
            // Sky/background
            float3 unit_dir = normalize3(cur_direction);
            float t = 0.5f * (unit_dir.y + 1.0f);
            float3 sky = (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
            color = color + throughput * sky * params.background_intensity;
            break;
         }

         // Get material data
         if (prd.hit_material_type == OptixMaterialType::LIGHT)
         {
            color = color + throughput * prd.hit_emission;
            break;
         }

         if (prd.hit_material_type == OptixMaterialType::SHOW_NORMALS)
         {
            color = color + throughput * make_float3(0.5f * (prd.hit_normal.x + 1.0f), 0.5f * (prd.hit_normal.y + 1.0f),
                                                      0.5f * (prd.hit_normal.z + 1.0f));
            break;
         }

         if (prd.hit_material_type == OptixMaterialType::CONSTANT)
         {
            color = color + throughput * prd.hit_color;
            break;
         }

         // Scatter based on material
         float3 scatter_dir;
         float3 attenuation;
         bool did_scatter = false;

         if (prd.hit_material_type == OptixMaterialType::LAMBERTIAN ||
             prd.hit_material_type == OptixMaterialType::SDF_MATERIAL)
         {
            // Lambertian: cosine-weighted hemisphere sampling
            scatter_dir = prd.hit_normal + rand_unit_vector(seed);
            // Catch degenerate direction
            if (fabsf(scatter_dir.x) < 1e-8f && fabsf(scatter_dir.y) < 1e-8f && fabsf(scatter_dir.z) < 1e-8f)
               scatter_dir = prd.hit_normal;
            attenuation = prd.hit_color;
            did_scatter = true;
         }
         else if (prd.hit_material_type == OptixMaterialType::MIRROR ||
                  prd.hit_material_type == OptixMaterialType::METAL)
         {
            float3 unit_dir = normalize3(cur_direction);
            scatter_dir = reflect3(unit_dir, prd.hit_normal);
            attenuation = prd.hit_color;
            did_scatter = (dot3(scatter_dir, prd.hit_normal) > 0.0f);
         }
         else if (prd.hit_material_type == OptixMaterialType::ROUGH_MIRROR)
         {
            float3 unit_dir = normalize3(cur_direction);
            float3 perturbed_n = normalize3(prd.hit_normal + prd.hit_roughness * rand_unit_sphere(seed));
            scatter_dir = reflect3(unit_dir, perturbed_n);
            attenuation = prd.hit_color;
            did_scatter = (dot3(scatter_dir, prd.hit_normal) > 0.0f);
         }
         else if (prd.hit_material_type == OptixMaterialType::GLASS ||
                  prd.hit_material_type == OptixMaterialType::DIELECTRIC)
         {
            attenuation = make_float3(1.0f, 1.0f, 1.0f);
            float ri = prd.front_face ? (1.0f / prd.hit_refractive_index) : prd.hit_refractive_index;

            float3 unit_dir = normalize3(cur_direction);
            float cos_theta = fminf(dot3(-unit_dir, prd.hit_normal), 1.0f);
            float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

            bool cannot_refract = ri * sin_theta > 1.0f;
            if (cannot_refract || reflectance(cos_theta, ri) > rand_float(seed))
            {
               scatter_dir = reflect3(unit_dir, prd.hit_normal);
            }
            else
            {
               scatter_dir = refract3(unit_dir, prd.hit_normal, ri);
            }
            did_scatter = true;
         }

         if (!did_scatter)
         {
            break;
         }

         throughput = throughput * attenuation;
         cur_origin = prd.hit_point;
         cur_direction = scatter_dir;

         // Russian Roulette (from bounce 1)
         if (bounce > 0)
         {
            float max_comp = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            float survival_prob = fminf(max_comp, 0.95f);
            if (rand_float(seed) > survival_prob)
            {
               break;
            }
            throughput = throughput / survival_prob;
         }
      }

      accumulated = accumulated + color;
   }

   params.accum_buffer[pixel_idx] = make_float4(accumulated.x, accumulated.y, accumulated.z, 0.0f);
}

//==============================================================================
// MISS — sky background
//==============================================================================

extern "C" __global__ void __miss__ms()
{
   PRDRadiance *prd = getPRD();
   prd->hit = false;
}

//==============================================================================
// CLOSEST HIT — fill PRD with material/geometry info
//==============================================================================

extern "C" __global__ void __closesthit__ch()
{
   PRDRadiance *prd = getPRD();
   prd->hit = true;

   const HitGroupData *sbt_data = reinterpret_cast<const HitGroupData *>(optixGetSbtDataPointer());

   // Get hit normal from intersection program attributes
   prd->hit_normal = make_float3(__int_as_float(optixGetAttribute_0()), __int_as_float(optixGetAttribute_1()),
                                  __int_as_float(optixGetAttribute_2()));

   // Front face test
   float3 ray_dir = optixGetWorldRayDirection();
   prd->front_face = dot3(ray_dir, prd->hit_normal) < 0.0f;
   if (!prd->front_face)
      prd->hit_normal = -prd->hit_normal;

   // Hit point
   float t_hit = optixGetRayTmax();
   float3 ray_origin = optixGetWorldRayOrigin();
   prd->hit_point = ray_origin + t_hit * ray_dir;

   // Look up material from params array
   int mat_idx = sbt_data->material_idx;
   if (mat_idx >= 0 && mat_idx < params.num_materials)
   {
      const OptixMaterialData &mat = params.materials[mat_idx];
      prd->hit_material_type = mat.type;
      prd->hit_color = mat.albedo;
      prd->hit_emission = mat.emission;
      prd->hit_roughness = mat.roughness;
      prd->hit_refractive_index = mat.refractive_index;

      // Apply procedural pattern if present
      if (mat.pattern == 1) // FIBONACCI_DOTS
      {
         // Simplified pattern: use solid color for now
         // Full fibonacci dot pattern requires porting the angular distance function
         prd->hit_color = mat.albedo;
      }
   }
   else
   {
      // Fallback
      prd->hit_material_type = OptixMaterialType::LAMBERTIAN;
      prd->hit_color = make_float3(1.0f, 0.0f, 1.0f); // Magenta = error
   }
}

//==============================================================================
// INTERSECTION — sphere
//==============================================================================

extern "C" __global__ void __intersection__sphere()
{
   const HitGroupData *sbt_data = reinterpret_cast<const HitGroupData *>(optixGetSbtDataPointer());

   const float3 center = sbt_data->center;
   const float radius = sbt_data->radius;

   const float3 ray_orig = optixGetObjectRayOrigin();
   const float3 ray_dir = optixGetObjectRayDirection();
   const float tmin = optixGetRayTmin();
   const float tmax = optixGetRayTmax();

   float3 oc = ray_orig - center;
   float a = dot3(ray_dir, ray_dir);
   float half_b = dot3(oc, ray_dir);
   float c = dot3(oc, oc) - radius * radius;
   float discriminant = half_b * half_b - a * c;

   if (discriminant < 0.0f)
      return;

   float sqrtd = sqrtf(discriminant);
   float root = (-half_b - sqrtd) / a;
   if (root < tmin || root > tmax)
   {
      root = (-half_b + sqrtd) / a;
      if (root < tmin || root > tmax)
         return;
   }

   // Compute outward normal at hit point
   float3 hit_point = ray_orig + root * ray_dir;
   float3 outward_normal = (hit_point - center) / radius;

   // Report intersection via attributes (normal passed as 3 float attributes)
   optixReportIntersection(root, 0, // hit kind
                           __float_as_int(outward_normal.x), __float_as_int(outward_normal.y),
                           __float_as_int(outward_normal.z));
}

//==============================================================================
// INTERSECTION — rectangle
//==============================================================================

extern "C" __global__ void __intersection__rectangle()
{
   const HitGroupData *sbt_data = reinterpret_cast<const HitGroupData *>(optixGetSbtDataPointer());

   const float3 corner = sbt_data->center; // corner stored in center field
   const float3 u = sbt_data->u_vec;
   const float3 v = sbt_data->v_vec;
   const float3 normal = sbt_data->normal;

   const float3 ray_orig = optixGetObjectRayOrigin();
   const float3 ray_dir = optixGetObjectRayDirection();
   const float tmin = optixGetRayTmin();
   const float tmax = optixGetRayTmax();

   float denom = dot3(normal, ray_dir);
   if (fabsf(denom) < 1e-8f)
      return;

   float t = dot3(normal, corner - ray_orig) / denom;
   if (t < tmin || t > tmax)
      return;

   float3 intersection = ray_orig + t * ray_dir;
   float3 p = intersection - corner;

   float alpha = dot3(p, u) / dot3(u, u);
   float beta = dot3(p, v) / dot3(v, v);

   if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f)
      return;

   optixReportIntersection(t, 1, // hit kind = 1 for rectangle
                           __float_as_int(normal.x), __float_as_int(normal.y), __float_as_int(normal.z));
}
