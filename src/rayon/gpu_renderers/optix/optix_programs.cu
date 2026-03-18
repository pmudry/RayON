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

__device__ __forceinline__ float3 cross3(const float3 &a, const float3 &b)
{
   return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

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
// ANISOTROPIC GGX MICROFACET — ported from microfacet_ggx.cuh (PBR Book §9.6)
//==============================================================================

__device__ __forceinline__ float Lambda_GGX_opt(const float3 &w, float alpha_x, float alpha_y)
{
   float wz2 = w.z * w.z;
   if (wz2 < 1e-16f)
      return 0.0f;
   float a2 = (alpha_x * w.x) * (alpha_x * w.x) + (alpha_y * w.y) * (alpha_y * w.y);
   return (sqrtf(1.0f + a2 / wz2) - 1.0f) * 0.5f;
}

__device__ __forceinline__ float G1_GGX_opt(const float3 &w, float alpha_x, float alpha_y)
{
   return 1.0f / (1.0f + Lambda_GGX_opt(w, alpha_x, alpha_y));
}

__device__ __forceinline__ float G_GGX_opt(const float3 &wo, const float3 &wi, float alpha_x, float alpha_y)
{
   return 1.0f / (1.0f + Lambda_GGX_opt(wo, alpha_x, alpha_y) + Lambda_GGX_opt(wi, alpha_x, alpha_y));
}

__device__ __forceinline__ float3 Sample_wm_GGX_opt(const float3 &wo, float alpha_x, float alpha_y,
                                                     float u1, float u2)
{
   float3 wh = normalize3(make_float3(alpha_x * wo.x, alpha_y * wo.y, wo.z));
   if (wh.z < 0.0f)
      wh = -wh;
   float3 T1 = (wh.z < 0.99999f) ? normalize3(cross3(make_float3(0.0f, 0.0f, 1.0f), wh))
                                  : make_float3(1.0f, 0.0f, 0.0f);
   float3 T2 = cross3(wh, T1);
   float r = sqrtf(u1);
   float phi = 6.283185307f * u2;
   float sin_phi, cos_phi;
   __sincosf(phi, &sin_phi, &cos_phi);
   float p_x = r * cos_phi;
   float p_y = r * sin_phi;
   float h = sqrtf(fmaxf(0.0f, 1.0f - p_x * p_x));
   float s = (1.0f + wh.z) * 0.5f;
   p_y = (1.0f - s) * h + s * p_y;
   float pz = sqrtf(fmaxf(0.0f, 1.0f - p_x * p_x - p_y * p_y));
   float3 nh = p_x * T1 + p_y * T2 + pz * wh;
   return normalize3(make_float3(alpha_x * nh.x, alpha_y * nh.y, fmaxf(1e-6f, nh.z)));
}

__device__ __forceinline__ float FrComplex1_opt(float cos2, float sin2, float cos_theta_i,
                                                float eta_ch, float k_ch)
{
   float eta2 = eta_ch * eta_ch;
   float k2 = k_ch * k_ch;
   float t0 = eta2 - k2 - sin2;
   float a2plusb2 = sqrtf(fmaxf(t0 * t0 + 4.0f * eta2 * k2, 0.0f));
   float a = sqrtf(fmaxf((a2plusb2 + t0) * 0.5f, 0.0f));
   float Rs_num = a2plusb2 + cos2 - 2.0f * a * cos_theta_i;
   float Rs_den = a2plusb2 + cos2 + 2.0f * a * cos_theta_i;
   float Rs = Rs_num / fmaxf(Rs_den, 1e-10f);
   float Rp_num = a2plusb2 * cos2 + sin2 * sin2 - 2.0f * a * cos_theta_i * sin2;
   float Rp_den = a2plusb2 * cos2 + sin2 * sin2 + 2.0f * a * cos_theta_i * sin2;
   float Rp = Rs * Rp_num / fmaxf(Rp_den, 1e-10f);
   return (Rs + Rp) * 0.5f;
}

__device__ __forceinline__ float3 FrComplex_opt(float cos_theta_i, const float3 &eta, const float3 &k)
{
   cos_theta_i = fminf(fmaxf(cos_theta_i, 0.0f), 1.0f);
   float cos2 = cos_theta_i * cos_theta_i;
   float sin2 = 1.0f - cos2;
   return make_float3(FrComplex1_opt(cos2, sin2, cos_theta_i, eta.x, k.x),
                      FrComplex1_opt(cos2, sin2, cos_theta_i, eta.y, k.y),
                      FrComplex1_opt(cos2, sin2, cos_theta_i, eta.z, k.z));
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
            float3 sky;
            if (params.use_hdr_env)
            {
               // Equirectangular (lat-long) mapping
               float theta = acosf(fmaxf(-1.0f, fminf(1.0f, unit_dir.y)));
               float phi   = atan2f(-unit_dir.z, unit_dir.x);
               float u     = (phi + 3.14159265f) * (0.5f / 3.14159265f);
               float v     = theta * (1.0f / 3.14159265f);
               float4 samp = tex2D<float4>(params.hdr_env_tex, u, v);
               sky = make_float3(samp.x, samp.y, samp.z);
            }
            else
            {
               float t = 0.5f * (unit_dir.y + 1.0f);
               sky = (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
            }
            color = color + throughput * sky * params.background_intensity;
            break;
         }

         // Get material data
         if (prd.hit_material_type == OptixMaterialType::LIGHT)
         {
            color = color + throughput * prd.hit_emission * params.light_intensity;
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
            float eff_roughness = prd.hit_roughness * params.metal_fuzziness;
            float3 perturbed_n = normalize3(prd.hit_normal + eff_roughness * rand_unit_sphere(seed));
            scatter_dir = reflect3(unit_dir, perturbed_n);
            attenuation = prd.hit_color;
            did_scatter = (dot3(scatter_dir, prd.hit_normal) > 0.0f);
         }
         else if (prd.hit_material_type == OptixMaterialType::GLASS ||
                  prd.hit_material_type == OptixMaterialType::DIELECTRIC)
         {
            attenuation = make_float3(1.0f, 1.0f, 1.0f);
            float eff_ior = prd.hit_refractive_index * params.glass_ior_multiplier;
            float ri = prd.front_face ? (1.0f / eff_ior) : eff_ior;

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
         else if (prd.hit_material_type == OptixMaterialType::ANISOTROPIC_METAL)
         {
            // Full anisotropic GGX microfacet conductor (PBR Book §9.6)
            float aspect = sqrtf(fmaxf(1e-4f, 1.0f - 0.9f * prd.hit_anisotropy));
            float r2 = prd.hit_roughness * prd.hit_roughness;
            float alpha_x = fmaxf(1e-4f, r2 / aspect);
            float alpha_y = fmaxf(1e-4f, r2 * aspect);

            float3 N = prd.hit_normal;

            if (fmaxf(alpha_x, alpha_y) < 1e-3f)
            {
               // Nearly smooth: perfect mirror with complex Fresnel tint
               float3 unit_dir = normalize3(cur_direction);
               scatter_dir = reflect3(unit_dir, N);
               float cos_i = fmaxf(dot3(-unit_dir, N), 0.0f);
               attenuation = prd.hit_color * FrComplex_opt(cos_i, prd.hit_eta, prd.hit_k);
               did_scatter = dot3(scatter_dir, N) > 0.0f;
            }
            else
            {
               // Build TBN frame
               float3 T = normalize3(prd.hit_tangent - dot3(prd.hit_tangent, N) * N);
               float3 B = cross3(N, T);

               // Outgoing direction (toward viewer) in local shading space
               float3 wo_world = normalize3(-cur_direction);
               float3 wo_local = make_float3(dot3(wo_world, T), dot3(wo_world, B), dot3(wo_world, N));

               if (wo_local.z > 0.0f)
               {
                  // Sample microfacet normal via VNDF
                  float3 wm = Sample_wm_GGX_opt(wo_local, alpha_x, alpha_y, rand_float(seed), rand_float(seed));

                  // Reflect incoming direction around microfacet normal to get wi
                  float3 wi_local = reflect3(make_float3(-wo_local.x, -wo_local.y, -wo_local.z), wm);

                  if (wi_local.z > 0.0f)
                  {
                     // Transform wi back to world space
                     float3 wi_world = wi_local.x * T + wi_local.y * B + wi_local.z * N;

                     // Fresnel + VNDF weighting: F * G / G1
                     float cos_theta_i = fmaxf(dot3(wo_local, wm), 0.0f);
                     float3 F = FrComplex_opt(cos_theta_i, prd.hit_eta, prd.hit_k);
                     float G = G_GGX_opt(wo_local, wi_local, alpha_x, alpha_y);
                     float G1 = G1_GGX_opt(wo_local, alpha_x, alpha_y);

                     attenuation = prd.hit_color * F * (G / fmaxf(G1, 1e-6f));
                     scatter_dir = wi_world;
                     did_scatter = true;
                  }
               }
            }
         }
         else if (prd.hit_material_type == OptixMaterialType::THIN_FILM)
         {
            // Full Airy thin-film interference (soap bubbles / oil slicks).
            // Stochastic reflect/transmit sampled from per-wavelength reflectance.
            float3 unit_dir = normalize3(cur_direction);
            float cos_i = fmaxf(fabsf(dot3(-unit_dir, prd.hit_normal)), 0.001f);

            float n0 = prd.hit_refractive_index; // exterior medium (air ≈ 1.0)
            float n1 = prd.hit_film_ior;         // film (soap/water ≈ 1.33)
            float d  = prd.hit_film_thickness;   // film thickness in nm

            float sin_i = sqrtf(fmaxf(0.0f, 1.0f - cos_i * cos_i));
            float sin_t_sq = (n0 / n1) * (n0 / n1) * (1.0f - cos_i * cos_i);
            float cos_t = (sin_t_sq >= 1.0f) ? 0.0f : sqrtf(1.0f - sin_t_sq);
            (void)sin_i;

            float r01_v = (n0 - n1) / (n0 + n1); r01_v *= r01_v;
            float x01 = 1.0f - cos_i;
            float R01 = r01_v + (1.0f - r01_v) * (x01 * x01 * x01 * x01 * x01);

            float r12_v = (n1 - n0) / (n1 + n0); r12_v *= r12_v;
            float x12 = 1.0f - cos_t;
            float R12 = r12_v + (1.0f - r12_v) * (x12 * x12 * x12 * x12 * x12);

            float sqrt_R = sqrtf(R01 * R12);

            // Airy formula for a single wavelength lambda (nm)
            auto airy = [&](float lambda) -> float {
               float delta = (4.0f * M_PIf * n1 * d * cos_t) / lambda;
               float c = cosf(delta);
               float num = R01 + R12 + 2.0f * sqrt_R * c;
               float den = 1.0f + R01 * R12 + 2.0f * sqrt_R * c;
               return num / fmaxf(den, 1e-8f);
            };

            float Rr = airy(650.0f), Rg = airy(550.0f), Rb = airy(450.0f);
            float avg_R = (Rr + Rg + Rb) / 3.0f;

            if (rand_float(seed) < avg_R)
            {
               // Reflect — carry iridescent color
               scatter_dir = reflect3(unit_dir, prd.hit_normal);
               attenuation = make_float3(Rr, Rg, Rb) * (1.0f / fmaxf(avg_R, 0.001f));
               did_scatter = (dot3(scatter_dir, prd.hit_normal) > 0.0f);
            }
            else
            {
               // Transmit — pass straight through
               scatter_dir = unit_dir;
               attenuation = make_float3(1.0f - Rr, 1.0f - Rg, 1.0f - Rb)
                             * (1.0f / fmaxf(1.0f - avg_R, 0.001f));
               did_scatter = true;
            }
         }
         else if (prd.hit_material_type == OptixMaterialType::CLEAR_COAT)
         {
            // Two-layer: Schlick Fresnel decides specular coat vs diffuse base
            float3 unit_dir = normalize3(cur_direction);
            float cos_theta = fminf(dot3(-unit_dir, prd.hit_normal), 1.0f);
            float r0 = (1.0f - prd.hit_refractive_index) / (1.0f + prd.hit_refractive_index);
            r0 = r0 * r0;
            float cx = 1.0f - cos_theta;
            float cx2 = cx * cx;
            float fresnel = r0 + (1.0f - r0) * (cx2 * cx2 * cx);

            if (rand_float(seed) < fresnel)
            {
               // Specular reflection through the coat
               float3 perturbed_n = normalize3(prd.hit_normal + prd.hit_roughness * rand_unit_sphere(seed));
               scatter_dir = reflect3(unit_dir, perturbed_n);
               attenuation = make_float3(1.0f, 1.0f, 1.0f); // coat is clear
               did_scatter = (dot3(scatter_dir, prd.hit_normal) > 0.0f);
            }
            else
            {
               // Diffuse base color
               scatter_dir = prd.hit_normal + rand_unit_vector(seed);
               if (fabsf(scatter_dir.x) < 1e-8f && fabsf(scatter_dir.y) < 1e-8f && fabsf(scatter_dir.z) < 1e-8f)
                  scatter_dir = prd.hit_normal;
               attenuation = prd.hit_color;
               did_scatter = true;
            }
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

      // Firefly rejection: clamp per-sample luminance to prevent single HDR texels
      // (e.g., sun disk in outdoor environment maps) from causing permanent white dots.
      // Uses a luminance-preserving scale so hue is maintained.
      constexpr float FIREFLY_CLAMP = 20.0f;
      float sample_lum = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
      if (sample_lum > FIREFLY_CLAMP)
      {
         float scale = FIREFLY_CLAMP / sample_lum;
         color = color * scale;
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
// GOLF BALL DISPLACEMENT HELPERS (ported from shader_golf.cu)
//==============================================================================

__device__ __forceinline__ float3 fibonacci_point_optix(int i, int n)
{
   const float ga = 2.39996323f;
   float k = (float)i + 0.5f;
   float phi = acosf(1.0f - 2.0f * k / (float)n);
   float theta = ga * k;
   float s = sinf(phi);
   return make_float3(cosf(theta) * s, sinf(theta) * s, cosf(phi));
}

__device__ __forceinline__ float distanceToNearestDimple_optix(float3 p)
{
   float3 q = normalize3(p);
   int N = params.golf_dimple_count;
   float max_dot = -1.0f;
   for (int i = 0; i < N; ++i)
   {
      float3 c = fibonacci_point_optix(i, N);
      float d = dot3(q, c);
      if (d > max_dot)
         max_dot = d;
   }
   max_dot = fmaxf(fminf(max_dot, 1.0f), -1.0f);
   return acosf(max_dot);
}

__device__ __forceinline__ float hexagonalDimplePattern_optix(float3 p)
{
   float ang = distanceToNearestDimple_optix(normalize3(p));
   float dimple_radius = params.golf_dimple_radius;
   float dimple_depth  = params.golf_dimple_depth;
   if (ang < dimple_radius)
   {
      float t = ang / dimple_radius;
      // Full half-period cosine (Hann profile): C1-continuous at boundary, eliminating jump artifacts.
      return -dimple_depth * 0.5f * (1.0f + cosf(t * M_PIf));
   }
   return 0.0f;
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

   // Get UV from attributes 3 and 4 (only valid for triangles, hit kind == 2)
   if (optixGetHitKind() == 2)
      prd->hit_uv = make_float2(__int_as_float(optixGetAttribute_3()), __int_as_float(optixGetAttribute_4()));
   else
      prd->hit_uv = make_float2(0.0f, 0.0f);

   // Front face test
   float3 ray_dir = optixGetWorldRayDirection();
   prd->front_face = dot3(ray_dir, prd->hit_normal) < 0.0f;
   if (!prd->front_face)
      prd->hit_normal = -prd->hit_normal;

   // Hit point
   float t_hit = optixGetRayTmax();
   float3 ray_origin = optixGetWorldRayOrigin();
   prd->hit_point = ray_origin + t_hit * ray_dir;

   // --- Displaced sphere: post-intersection position and normal correction ---
   if (sbt_data->geom_type == OptixGeomType::DISPLACED_SPHERE)
   {
      float3 surface_point = prd->hit_point;
      float3 center        = sbt_data->center;

      float3 local          = surface_point - center;
      float3 normalized_loc = normalize3(local);
      float  base_disp      = hexagonalDimplePattern_optix(normalized_loc);

      const float displacement_scale = 0.2f;
      float dimple_depth_param       = params.golf_dimple_depth;
      const float geo_strength       = 0.35f;

      float  d_norm       = fminf(1.0f, fmaxf(0.0f, -base_disp / dimple_depth_param));
      float  outward_push = sbt_data->radius * geo_strength * (1.0f - d_norm);
      float3 displaced_pt = surface_point + outward_push * normalized_loc;

      float3 base_normal = normalize3(displaced_pt - center);
      float3 final_normal;

      if (base_disp < -0.001f)
      {
         // Duff et al. 2017 "Building an Orthonormal Basis, Revisited":
         // continuously varying basis — no seam from a sudden helper-vector switch.
         float  nz_sign = copysignf(1.0f, base_normal.z);
         float  nz_a    = -1.0f / (nz_sign + base_normal.z);
         float  nz_b    = base_normal.x * base_normal.y * nz_a;
         float3 t1      = make_float3(1.0f + nz_sign * base_normal.x * base_normal.x * nz_a,
                                      nz_sign * nz_b,
                                      -nz_sign * base_normal.x);
         float3 t2      = make_float3(nz_b,
                                      nz_sign + base_normal.y * base_normal.y * nz_a,
                                      -base_normal.y);

         const float h = 0.015f;
         float3 p_hat  = base_normal;
         float  d0     = hexagonalDimplePattern_optix(p_hat);
         float  d1     = hexagonalDimplePattern_optix(normalize3(p_hat + h * t1));
         float  d2     = hexagonalDimplePattern_optix(normalize3(p_hat + h * t2));

         float  dd1      = (d1 - d0) / h;
         float  dd2      = (d2 - d0) / h;
         float3 grad_tan = dd1 * t1 + dd2 * t2;
         float3 delta_n  = (-displacement_scale) * grad_tan;

         float max_len = 0.4f;
         float len     = length3(delta_n);
         if (len > max_len && len > 1e-6f)
            delta_n = (max_len / len) * delta_n;

         float3 perturbed = normalize3(base_normal + delta_n);
         if (dot3(perturbed, base_normal) < 0.0f)
            perturbed = -perturbed;
         if (!(perturbed.x == perturbed.x) || !(perturbed.y == perturbed.y) || !(perturbed.z == perturbed.z))
            perturbed = base_normal;
         final_normal = perturbed;
      }
      else
      {
         final_normal = base_normal;
      }

      // Re-determine front face with the perturbed normal and push off surface
      prd->front_face = dot3(ray_dir, final_normal) < 0.0f;
      if (!prd->front_face)
         final_normal = -final_normal;
      prd->hit_normal = final_normal;

      const float surface_epsilon = 1e-3f;
      prd->hit_point = displaced_pt + surface_epsilon * final_normal;
   }

   // Look up material from params array
   int mat_idx = sbt_data->material_idx;

   // Compute surface tangent for anisotropic materials (geometry-specific)
   {
      float3 N = prd->hit_normal;
      float3 T;
      if (sbt_data->geom_type == OptixGeomType::RECTANGLE)
      {
         T = normalize3(sbt_data->u_vec);
      }
      else if (sbt_data->geom_type == OptixGeomType::TRIANGLE)
      {
         float3 edge1 = sbt_data->tri_v1 - sbt_data->tri_v0;
         T = normalize3(edge1 - dot3(edge1, N) * N);
         // fallback if degenerate
         if (dot3(T, T) < 1e-8f)
         {
            float3 helper = (fabsf(N.x) > 0.8f) ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
            T = normalize3(cross3(helper, N));
         }
      }
      else // SPHERE, DISPLACED_SPHERE
      {
         // Azimuthal tangent matching CUDA renderer: cross(up, outward_normal)
         float3 up = make_float3(0.0f, 1.0f, 0.0f);
         T = cross3(up, N);
         if (dot3(T, T) < 1e-6f)
            T = make_float3(1.0f, 0.0f, 0.0f); // degenerate at poles
         T = normalize3(T);
      }
      prd->hit_tangent = T;
   }

   if (mat_idx >= 0 && mat_idx < params.num_materials)
   {
      const OptixMaterialData &mat = params.materials[mat_idx];
      prd->hit_material_type = mat.type;
      prd->hit_color = mat.albedo;
      prd->hit_emission = mat.emission;
      prd->hit_roughness = mat.roughness;
      prd->hit_refractive_index = mat.refractive_index;
      prd->hit_film_thickness = mat.film_thickness;
      prd->hit_film_ior = mat.film_ior;
      prd->hit_anisotropy = mat.anisotropy;
      prd->hit_eta = mat.eta;
      prd->hit_k = mat.k;

      // Texture sampling: override diffuse color with texel if texture is bound
      if (mat.texture_id >= 0 && mat.texture_id < params.num_textures && params.d_textures)
      {
         float4 texel = tex2D<float4>(params.d_textures[mat.texture_id], prd->hit_uv.x, 1.0f - prd->hit_uv.y);
         prd->hit_color = make_float3(texel.x, texel.y, texel.z);
      }

      // Apply procedural pattern if present
      if (mat.pattern == 1) // FIBONACCI_DOTS
      {
         float3 local  = prd->hit_point - sbt_data->center;
         float3 dir    = normalize3(local);
         int    N      = (int)mat.pattern_param1;
         float  dot_rad = mat.pattern_param2;
         float  max_dp  = -1.0f;
         for (int i = 0; i < N; ++i)
         {
            float3 c = fibonacci_point_optix(i, N);
            float  d = dot3(dir, c);
            if (d > max_dp)
               max_dp = d;
         }
         max_dp = fmaxf(fminf(max_dp, 1.0f), -1.0f);
         float ang  = acosf(max_dp);
         float mask = ang < dot_rad ? 0.0f : 1.0f;
         prd->hit_color = mask * mat.albedo + (1.0f - mask) * mat.pattern_color;
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

//==============================================================================
// INTERSECTION — triangle (Möller–Trumbore)
//==============================================================================

extern "C" __global__ void __intersection__triangle()
{
   const HitGroupData *sbt_data = reinterpret_cast<const HitGroupData *>(optixGetSbtDataPointer());

   const float3 v0 = sbt_data->tri_v0;
   const float3 v1 = sbt_data->tri_v1;
   const float3 v2 = sbt_data->tri_v2;

   const float3 ray_orig = optixGetObjectRayOrigin();
   const float3 ray_dir  = optixGetObjectRayDirection();
   const float  tmin     = optixGetRayTmin();
   const float  tmax     = optixGetRayTmax();

   const float3 edge1 = v1 - v0;
   const float3 edge2 = v2 - v0;
   const float3 h = cross3(ray_dir, edge2);
   const float  a = dot3(edge1, h);

   if (fabsf(a) < 1e-8f)
      return; // Ray parallel to triangle

   const float  inv_a = 1.0f / a;
   const float3 s = ray_orig - v0;
   const float  u = inv_a * dot3(s, h);
   if (u < 0.0f || u > 1.0f)
      return;

   const float3 q = cross3(s, edge1);
   const float  v = inv_a * dot3(ray_dir, q);
   if (v < 0.0f || u + v > 1.0f)
      return;

   const float t = inv_a * dot3(edge2, q);
   if (t < tmin || t > tmax)
      return;

   // Compute normal: interpolate per-vertex normals or use face normal
   float3 outward_normal;
   if (sbt_data->tri_has_normals)
   {
      const float w = 1.0f - u - v;
      outward_normal = normalize3(w * sbt_data->tri_n0 + u * sbt_data->tri_n1 + v * sbt_data->tri_n2);
   }
   else
   {
      outward_normal = normalize3(cross3(edge1, edge2));
   }

   // Compute UV: interpolate per-vertex UV coords using barycentric coordinates
   float2 hit_uv = make_float2(0.0f, 0.0f);
   if (sbt_data->tri_has_uvs)
   {
      const float w = 1.0f - u - v;
      hit_uv.x = w * sbt_data->tri_uv0.x + u * sbt_data->tri_uv1.x + v * sbt_data->tri_uv2.x;
      hit_uv.y = w * sbt_data->tri_uv0.y + u * sbt_data->tri_uv1.y + v * sbt_data->tri_uv2.y;
   }

   optixReportIntersection(t, 2, // hit kind = 2 for triangle
                           __float_as_int(outward_normal.x),
                           __float_as_int(outward_normal.y),
                           __float_as_int(outward_normal.z),
                           __float_as_int(hit_uv.x),
                           __float_as_int(hit_uv.y));
}

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
