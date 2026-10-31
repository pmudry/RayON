// OptiX device programs: raygen, miss, closest-hit, intersection
// Compiled to PTX at build time, loaded by host at runtime.

#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "optix_params.h"
#include "../sdf_primitives.cuh"
#include "volume_medium.cuh"
// OptiX PTX is a separate GPU binary; it needs its own copy of the table.
#define SOBOL_DEFINE_DIRECTIONS
#include "../sobol_sampler.cuh"

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

// Uniform in [0, 1): the top 24 bits (seed / 0xFFFFFFFF rounds to exactly 1.0f for the largest
// seeds, and logf(1 - u) in the free-flight sampling would then be -inf)
__device__ __forceinline__ float rand_float(unsigned int &seed)
{
   seed = pcg_hash(seed);
   return (float)(seed >> 8) * (1.0f / 16777216.0f);
}

//------------------------------------------------------------------------------
// Sobol-aware rand_float for OptiX.
// Routes through the scrambled Sobol sequence when params.use_sobol is true,
// incrementing ss.dim_idx automatically.  Falls back to PCG when the
// dimension budget is exceeded.
//
// The Sobol state is raygen-local: it never crosses the optixTrace() boundary,
// so it stays in registers instead of the payload struct in local memory.
//------------------------------------------------------------------------------
struct SobolState
{
   uint32_t sample_idx; ///< Gray-code sample index for Sobol path
   uint32_t dim_idx;    ///< Dimension counter — incremented per sobol_rand_float call
   uint32_t pixel_hash; ///< Per-pixel stable hash for Owen scrambling
};

__device__ __forceinline__ float sobol_rand_float(SobolState &ss, unsigned int &seed)
{
   if (params.use_sobol)
   {
      if (ss.dim_idx < (uint32_t)SOBOL_MAX_DIM)
      {
         float v = sobol_float(ss.sample_idx, (int)ss.dim_idx, ss.pixel_hash);
         ss.dim_idx++;
         return v;
      }
      // PCG fallback for dims beyond SOBOL_MAX_DIM
      ss.dim_idx++;
      return rand_float(seed);
   }
   return rand_float(seed);
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
   const unsigned long long uptr = reinterpret_cast<unsigned long long>(prd);
   unsigned int p0 = static_cast<unsigned int>(uptr & 0xFFFFFFFFULL);
   unsigned int p1 = static_cast<unsigned int>(uptr >> 32);
   optixTrace(handle, origin, direction, tmin, tmax,
              0.0f,                         // rayTime
              OptixVisibilityMask(1),       // visibilityMask
              OPTIX_RAY_FLAG_DISABLE_ANYHIT, // no any-hit programs in this pipeline
              0,                            // SBT offset
              1,                            // SBT stride
              OPTIX_MISS_RADIANCE,          // missSBTIndex
              p0, p1);
}

//==============================================================================
// SHADOW RAY — occlusion only. Closest-hit is disabled so a hit costs nothing
// beyond traversal: the payload starts as "occluded" and only __miss__shadow
// clears it.
//==============================================================================

__device__ __forceinline__ bool trace_shadow(OptixTraversableHandle handle, float3 origin,
                                              float3 direction, float tmin, float tmax)
{
   unsigned int occluded = 1u;
   optixTrace(handle, origin, direction, tmin, tmax,
              0.0f, OptixVisibilityMask(1),
              OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
                  OPTIX_RAY_FLAG_DISABLE_ANYHIT,
              0, 1, OPTIX_MISS_SHADOW, occluded);
   return occluded != 0u;
}

//==============================================================================
// NEE / MIS HELPERS — light sampling, BSDF eval/PDF
//==============================================================================

struct OptiXLightSample
{
   float3 direction;
   float3 emission;
   float  pdf;
   float  dist;
};

__device__ __forceinline__ void build_onb(const float3 &n, float3 &u, float3 &v)
{
   float sign = copysignf(1.0f, n.z);
   float a    = -1.0f / (sign + n.z);
   float b    = n.x * n.y * a;
   u = make_float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
   v = make_float3(b, sign + n.y * n.y * a, -n.y);
}

/// Artistic distance falloff of the lights (params.light_falloff): the geometry already gives the
/// physical inverse square, so the factor is 1 at exponent 2 and d^(2 - falloff) otherwise,
/// pivoting at one scene unit. Applied to every light contribution past the camera ray, both the
/// NEE samples and the emitter hits, so MIS stays consistent.
__device__ __forceinline__ float light_falloff_factor(float dist)
{
   return (params.light_falloff == 2.0f) ? 1.0f : powf(fmaxf(dist, 1e-3f), 2.0f - params.light_falloff);
}

__device__ OptiXLightSample sample_light_optix(const float3 &shading_p, float u_sel, float u1, float u2)
{
   OptiXLightSample ls;
   ls.pdf = 0.0f;

   if (params.num_lights == 0) return ls;

   // CDF-based light selection
   int idx = 0;
   for (int i = 0; i < params.num_lights; ++i)
   {
      if (u_sel < params.light_cdfs[i + 1]) { idx = i; break; }
      idx = i;
   }

   const OptixLightData &light = params.lights[idx];
   float select_pdf = params.light_cdfs[idx + 1] - params.light_cdfs[idx];
   if (select_pdf < 1e-8f) return ls;

   if (light.geom_type == OptixLightGeomType::RECTANGLE)
   {
      float3 sampled_pt = light.corner + u1 * light.u_vec + u2 * light.v_vec;
      float3 to_light   = sampled_pt - shading_p;
      float  dist       = length3(to_light);
      if (dist < 1e-5f) return ls;

      float3 dir   = to_light / dist;
      float  cos_l = fabsf(dot3(-dir, light.normal));
      if (cos_l < 1e-6f) return ls;

      float area_pdf = 1.0f / fmaxf(light.area, 1e-8f);
      ls.direction   = dir;
      ls.emission    = light.emission * params.light_intensity;
      ls.dist        = dist;
      ls.pdf         = select_pdf * area_pdf * (dist * dist) / cos_l;
   }
   else // SPHERE
   {
      float3 to_center = light.center - shading_p;
      float  dist_sq   = dot3(to_center, to_center);

      if (dist_sq <= light.radius * light.radius)
      {
         // Shading point is inside the sphere — sample uniformly over 4π steradians.
         float z    = 1.0f - 2.0f * u1;
         float r_xy = sqrtf(fmaxf(0.0f, 1.0f - z * z));
         float phi  = 2.0f * M_PIf * u2;
         float3 dir = make_float3(r_xy * cosf(phi), r_xy * sinf(phi), z);

         // Interior ray-sphere intersection: take the far root.
         float3 oc  = shading_p - light.center;
         float  b   = dot3(oc, dir);
         float  c   = dot3(oc, oc) - light.radius * light.radius;
         float  disc = b * b - c;
         if (disc <= 0.0f) return ls;
         float  t   = -b + sqrtf(disc);
         if (t <= 1e-5f) return ls;

         ls.direction = dir;
         ls.emission  = light.emission * params.light_intensity;
         ls.dist      = t;
         ls.pdf       = select_pdf / (4.0f * M_PIf);
      }
      else
      {
         float  dist          = sqrtf(dist_sq);
         float  cos_theta_max = sqrtf(fmaxf(0.0f, 1.0f - (light.radius * light.radius) / dist_sq));
         float  solid_angle   = 2.0f * M_PIf * (1.0f - cos_theta_max);
         if (solid_angle < 1e-8f) return ls;

         float3 w = to_center / dist;
         float3 u_basis, v_basis;
         build_onb(w, u_basis, v_basis);

         float cos_theta = 1.0f - u1 * (1.0f - cos_theta_max);
         float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
         float phi       = 2.0f * 3.14159265f * u2;

         ls.direction = normalize3(sin_theta * cosf(phi) * u_basis + sin_theta * sinf(phi) * v_basis + cos_theta * w);
         ls.emission  = light.emission * params.light_intensity;
         // Shadow ray length: distance to the sphere's surface along the sampled direction (near
         // root), not to its centre, or the shadow ray would hit the emitter itself.
         {
            float3 oc   = shading_p - light.center;
            float  b    = dot3(oc, ls.direction);
            float  disc = b * b - (dist_sq - light.radius * light.radius);
            ls.dist     = (disc > 0.0f) ? (-b - sqrtf(disc)) : dist * cos_theta;
         }
         ls.pdf       = select_pdf / solid_angle;
      }
   }
   ls.emission = ls.emission * light_falloff_factor(ls.dist);
   return ls;
}

/// Compute the solid-angle PDF the NEE sampler would assign to hitting a specific light
__device__ float light_dir_pdf_optix(int light_idx, const float3 &prev_p, const float3 &dir)
{
   if (light_idx < 0 || params.num_lights == 0) return 0.0f;

   const OptixLightData &light = params.lights[light_idx];
   float select_pdf = params.light_cdfs[light_idx + 1] - params.light_cdfs[light_idx];
   if (select_pdf < 1e-8f) return 0.0f;

   if (light.geom_type == OptixLightGeomType::RECTANGLE)
   {
      float denom = dot3(dir, light.normal);
      if (fabsf(denom) < 1e-8f) return 0.0f;
      float t = dot3(light.corner - prev_p, light.normal) / denom;
      if (t < 0.0f) return 0.0f;
      float cos_l = fabsf(dot3(-dir, light.normal));
      if (cos_l < 1e-6f) return 0.0f;
      return select_pdf * (t * t) / (cos_l * fmaxf(light.area, 1e-8f));
   }
   else // SPHERE
   {
      float3 to_center   = light.center - prev_p;
      float  dist_sq     = dot3(to_center, to_center);
      float  r_sq        = light.radius * light.radius;

      // If the shading point is inside (or on) the sphere, the visible solid
      // angle is the full 4π steradians.  Guard against dist_sq ≈ 0 as well.
      const float eps_dist2    = 1e-8f;
      const float inv_four_pi  = 0.07957747154f; // 1 / (4π)
      if (dist_sq <= r_sq + eps_dist2)
         return select_pdf * inv_four_pi;

      float  cos_max     = sqrtf(fmaxf(0.0f, 1.0f - r_sq / dist_sq));
      float  solid_angle = 2.0f * 3.14159265f * (1.0f - cos_max);
      if (solid_angle < 1e-8f) return 0.0f;
      return select_pdf / solid_angle;
   }
}

//------------------------------------------------------------------------------
// HDR sky importance sampling (tables built by buildEnvSamplingTables() in optix_renderer.cu).
// The sky is sampled proportionally to luminance x sin(theta) over a downsampled lat-long grid:
// params.env_marginal_cdf picks a row, params.env_conditional_cdf a cell of that row, and the
// direction is uniform inside the cell. A cell's probability is exactly the product of the two
// CDF steps, so the sampler and env_pdf_optix() (the pdf of a BSDF-sampled ray that escapes)
// agree, which is what the MIS weights need. Without these tables the sun of a clear-sky HDRI
// (a few hundred texels holding half the sky's energy) is only ever hit by chance: one hit
// in ~40 000 diffuse bounces, each worth tens of thousands — speckle that never converges.
//------------------------------------------------------------------------------

/// True when the sky is sampled directly: an HDR map with its tables, visible, and MIS on
__device__ __forceinline__ bool env_sampling_active()
{
   return params.mis_enabled && params.use_hdr_env && params.env_marginal_cdf != nullptr &&
          params.background_intensity > 0.0f;
}

/// Probability that a NEE sample targets the sky rather than a scene light
__device__ __forceinline__ float env_select_prob()
{
   if (!env_sampling_active()) return 0.0f;
   return params.num_lights > 0 ? 0.5f : 1.0f;
}

/// Something to run NEE against: scene lights, the sampled sky, or both
__device__ __forceinline__ bool nee_available()
{
   return params.num_lights > 0 || env_sampling_active();
}

/// Lat-long mapping shared by the sky lookup, the sampler and the pdf: u = azimuth, v = polar angle
__device__ __forceinline__ float2 env_dir_to_uv(const float3 &d)
{
   float theta = acosf(fmaxf(-1.0f, fminf(1.0f, d.y)));
   float phi   = atan2f(-d.z, d.x);
   return make_float2((phi + M_PIf) * (0.5f / M_PIf), theta * (1.0f / M_PIf));
}

__device__ __forceinline__ float3 env_uv_to_dir(float u, float v)
{
   float st, ct, sp, cp;
   sincosf(v * M_PIf, &st, &ct);
   sincosf(u * 2.0f * M_PIf - M_PIf, &sp, &cp);
   return make_float3(st * cp, ct, -st * sp);
}

/// Largest i in [0, n-1] with cdf[i] <= u (cdf holds n+1 entries, cdf[0] = 0, cdf[n] = 1, u < 1)
__device__ __forceinline__ int env_cdf_search(const float *cdf, int n, float u)
{
   int lo = 0, hi = n;
   while (hi - lo > 1)
   {
      const int mid = (lo + hi) >> 1;
      if (cdf[mid] <= u) lo = mid; else hi = mid;
   }
   return lo;
}

/// Solid-angle pdf of the sky sampler for a unit direction (0 where the map is black)
__device__ float env_pdf_optix(const float3 &dir)
{
   const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - dir.y * dir.y));
   if (sin_theta <= 1e-6f) return 0.0f;
   const int    W  = params.env_w, H = params.env_h;
   const float2 uv = env_dir_to_uv(dir);
   const int    iu = min(max((int)(uv.x * (float)W), 0), W - 1);
   const int    iv = min(max((int)(uv.y * (float)H), 0), H - 1);
   const float  pv = params.env_marginal_cdf[iv + 1] - params.env_marginal_cdf[iv];
   const float *row = params.env_conditional_cdf + (size_t)iv * (W + 1);
   const float  pu = row[iu + 1] - row[iu];
   return pv * pu * (float)(W * H) / (2.0f * M_PIf * M_PIf * sin_theta);
}

/// Sample a sky direction from the luminance tables: emission already scaled by the sky intensity,
/// pdf per solid angle, dist "infinite" for the shadow ray. pdf = 0 on failure.
__device__ OptiXLightSample sample_env_optix(float u1, float u2)
{
   OptiXLightSample ls;
   ls.pdf = 0.0f;
   const int W = params.env_w, H = params.env_h;

   const int   iv = env_cdf_search(params.env_marginal_cdf, H, u1);
   const float pv = params.env_marginal_cdf[iv + 1] - params.env_marginal_cdf[iv];
   if (pv <= 0.0f) return ls;
   const float *row = params.env_conditional_cdf + (size_t)iv * (W + 1);
   const int    iu  = env_cdf_search(row, W, u2);
   const float  pu  = row[iu + 1] - row[iu];
   if (pu <= 0.0f) return ls;

   // Uniform inside the cell, re-using what is left of the variates (keeps the Sobol stratification)
   const float dv = fminf((u1 - params.env_marginal_cdf[iv]) / pv, 0.99999f);
   const float du = fminf((u2 - row[iu]) / pu, 0.99999f);
   const float v  = ((float)iv + dv) / (float)H;
   const float u  = ((float)iu + du) / (float)W;
   const float sin_theta = sinf(v * M_PIf);
   if (sin_theta <= 1e-6f) return ls;

   const float4 s = tex2D<float4>(params.hdr_env_tex, u, v);
   ls.direction = env_uv_to_dir(u, v);
   ls.emission  = make_float3(s.x, s.y, s.z) * params.background_intensity;
   ls.dist      = 1e16f;
   ls.pdf       = pv * pu * (float)(W * H) / (2.0f * M_PIf * M_PIf * sin_theta);
   return ls;
}

/// NEE target: the sky with probability env_select_prob(), else a scene light. The pdf carries the
/// selection probability; light_dir_pdf_optix() / env_pdf_optix() callers must apply the same factor.
__device__ OptiXLightSample sample_nee_optix(const float3 &shading_p, float u_sel, float u1, float u2)
{
   const float p_env = env_select_prob();
   if (u_sel < p_env)
   {
      OptiXLightSample ls = sample_env_optix(u1, u2);
      ls.pdf *= p_env;
      return ls;
   }
   OptiXLightSample ls = sample_light_optix(shading_p, (u_sel - p_env) / (1.0f - p_env), u1, u2);
   ls.pdf *= (1.0f - p_env);
   return ls;
}

/// True for materials that are delta distributions (skip NEE)
__device__ __forceinline__ bool is_delta_material_optix(OptixMaterialType mat)
{
   return mat != OptixMaterialType::LAMBERTIAN && mat != OptixMaterialType::SDF_MATERIAL
       && mat != OptixMaterialType::CLEAR_COAT;
}

//==============================================================================
// RAY GENERATION — path tracing loop
//==============================================================================

//==============================================================================
// VOLUME SHADOW RAYS — transmittance along a segment through the scene's media
//==============================================================================

// Ratio tracking: unbiased transmittance estimate over `len` units of medium `m` from p
__device__ float ratio_track_optix(const OptixMaterialData &m, float3 p, float3 dir, float len,
                                   unsigned int &seed)
{
   const float sigma_max = volume_majorant(m.volume);
   if (sigma_max <= 0.0f)
      return 1.0f;
   float T = 1.0f, t = 0.0f;
   for (int k = 0; k < VOLUME_MAX_NULL_STEPS; ++k)
   {
      {
         const float3 pk   = p + t * dir;
         const float  skip = volume_empty_skip(m.volume, pk.x, pk.y, pk.z);
         if (skip > 0.0f)
         {
            t += skip;
            if (t >= len)
               break;
            continue;
         }
      }
      t -= logf(1.0f - rand_float(seed)) / sigma_max;
      if (t >= len)
         break;
      const float3 q = p + t * dir;
      T *= 1.0f - volume_density(m.volume, q.x, q.y, q.z) / sigma_max;
      // Russian roulette on a faint estimate: stop early, unbiased (survivors carry the ceiling)
      constexpr float RR_T = 0.1f;
      if (T < RR_T)
      {
         if (rand_float(seed) * RR_T >= T)
            return 0.0f;
         T = RR_T;
      }
   }
   return T;
}

/// Reflectance of a thin film (Airy formula at 650, 550 and 450 nm) for the cosine of incidence.
/// Its transmission goes straight on (the film has the same medium on both sides): 1 - R.
__device__ __forceinline__ float3 thin_film_reflectance_optix(const OptixMaterialData &mat, float cos_i)
{
   cos_i = fmaxf(cos_i, 0.001f);

   float n0 = mat.refractive_index; // exterior medium (air ≈ 1.0)
   float n1 = mat.film_ior;         // film (soap/water ≈ 1.33)
   float d  = mat.film_thickness;   // film thickness in nm

   float sin_t_sq = (n0 / n1) * (n0 / n1) * (1.0f - cos_i * cos_i);
   float cos_t = (sin_t_sq >= 1.0f) ? 0.0f : sqrtf(1.0f - sin_t_sq);

   float r01_v = (n0 - n1) / (n0 + n1); r01_v *= r01_v;
   float x01 = 1.0f - cos_i;
   float R01 = r01_v + (1.0f - r01_v) * (x01 * x01 * x01 * x01 * x01);

   float r12_v = (n1 - n0) / (n1 + n0); r12_v *= r12_v;
   float x12 = 1.0f - cos_t;
   float R12 = r12_v + (1.0f - r12_v) * (x12 * x12 * x12 * x12 * x12);

   // Product of the two amplitude coefficients r01 * r12. For a film between two media of
   // the same index (soap in air) r12 = -r01, so it is negative: the reflection on the
   // denser side carries a pi phase shift and a vanishing film turns black.
   float rr = -sqrtf(R01 * R12);

   // Airy formula for a single wavelength lambda (nm)
   auto airy = [&](float lambda) -> float {
      float delta = (4.0f * M_PIf * n1 * d * cos_t) / lambda;
      float c = cosf(delta);
      float num = R01 + R12 + 2.0f * rr * c;
      float den = 1.0f + R01 * R12 + 2.0f * rr * c;
      return num / fmaxf(den, 1e-8f);
   };

   return make_float3(airy(650.0f), airy(550.0f), airy(450.0f));
}

// Transmittance from origin over `dist` units along dir: 1 when clear, 0 when an opaque surface
// is in the way, in between through volumes and thin films (soap bubbles let 1 - R through, per
// channel, on a straight line). Without either in the scene this is the plain occlusion test (one
// any-hit style trace). films_only: what a camera ray sees through, visible films and nothing else
// (the caustic pass's join to the camera).
__device__ float3 shadow_transmittance_optix(float3 origin, float3 dir, float dist, unsigned int &seed,
                                             bool films_only = false)
{
   constexpr int SHADOW_MAX_FILMS = 8; // bubbles crossed (two faces each) before giving up
   const float3 ZERO = make_float3(0.0f, 0.0f, 0.0f);
   if ((films_only || !params.has_volume) && !params.has_thin_film)
      return trace_shadow(params.traversable, origin, dir, 0.0001f, dist) ? ZERO : make_float3(1.0f, 1.0f, 1.0f);
   float3 T  = make_float3(1.0f, 1.0f, 1.0f);
   float  t0 = 0.0f;
   for (int i = 0; i < 2 * VOLUME_MAX_PASSES + 2 * SHADOW_MAX_FILMS; ++i)
   {
      if (dist - t0 <= 0.0002f)
         return T;
      const float3 from = origin + t0 * dir;
      PRDRadiance h;
      h.hit = false;
      trace(params.traversable, from, dir, 0.0001f, dist - t0, &h);
      if (!h.hit)
         return T;
      const OptixMaterialType type =
          (h.material_idx < 0) ? OptixMaterialType::LAMBERTIAN : params.materials[h.material_idx].type;
      const float3 d_hit = h.hit_point - from;
      const float  t_hit = t0 + sqrtf(dot3(d_hit, d_hit));
      if (type == OptixMaterialType::THIN_FILM && (h.visible || !films_only))
      {
         const float3 R = thin_film_reflectance_optix(params.materials[h.material_idx], fabsf(dot3(dir, h.hit_normal)));
         T  = T * make_float3(1.0f - R.x, 1.0f - R.y, 1.0f - R.z);
         t0 = t_hit + 0.001f;
         continue;
      }
      if (type != OptixMaterialType::VOLUME || films_only)
         return ZERO; // opaque surface (or, for films_only, anything the camera ray does not see through)
      const OptixMaterialData &m = params.materials[h.material_idx];
      if (h.front_face)
      {
         // Entering a medium: track up to its far side, or to the end of the segment
         PRDRadiance ex;
         ex.hit = false;
         if (dist - t_hit > 0.0002f)
            trace(params.traversable, h.hit_point, dir, 0.0001f, dist - t_hit, &ex);
         if (ex.hit && (ex.material_idx < 0 || params.materials[ex.material_idx].type != OptixMaterialType::VOLUME))
            return ZERO; // opaque surface inside the medium
         float t_exit = dist;
         if (ex.hit)
         {
            const float3 d_ex = ex.hit_point - h.hit_point;
            t_exit = t_hit + sqrtf(dot3(d_ex, d_ex));
         }
         T = T * ratio_track_optix(m, h.hit_point, dir, t_exit - t_hit, seed);
         if (!ex.hit)
            return T;
         t0 = t_exit + 0.001f;
      }
      else
      {
         // Started inside the medium: track up to this far side
         T = T * ratio_track_optix(m, from, dir, t_hit - t0, seed);
         t0 = t_hit + 0.001f;
      }
      if (fmaxf(T.x, fmaxf(T.y, T.z)) <= 0.0f)
         return ZERO;
   }
   return T;
}

// Transmittance for the NEE of a scattering event at p, inside medium m, over `dist` units along
// dir. A boxed medium (the scene's only one) is left analytically: ratio tracking runs first, up to
// the box face, and in a dense cloud its roulette usually ends the estimate before any ray is
// traced. Only a surviving estimate pays an occlusion-only ray for opaque surfaces inside the box
// (the ground under a cloud), and then the general path from the box face to the light. Each
// factor is an independent estimate of its own segment, so the product stays unbiased.
__device__ float3 volume_nee_transmittance_optix(const OptixMaterialData &m, float3 p, float3 dir, float dist,
                                                 unsigned int &seed)
{
   if (!m.volume.boxed)
      return shadow_transmittance_optix(p, dir, dist, seed);

   // Exit distance of the box along dir (p is inside it)
   const float3 lo = m.volume.box_lo, hi = m.volume.box_hi;
   const float tx = (dir.x > 0.0f) ? (hi.x - p.x) / dir.x : (dir.x < 0.0f) ? (lo.x - p.x) / dir.x : 1e30f;
   const float ty = (dir.y > 0.0f) ? (hi.y - p.y) / dir.y : (dir.y < 0.0f) ? (lo.y - p.y) / dir.y : 1e30f;
   const float tz = (dir.z > 0.0f) ? (hi.z - p.z) / dir.z : (dir.z < 0.0f) ? (lo.z - p.z) / dir.z : 1e30f;
   const float t_box = fmaxf(0.0f, fminf(fminf(tx, ty), tz));
   const float t_med = fminf(t_box, dist);

   const float3 ZERO = make_float3(0.0f, 0.0f, 0.0f);
   const float  T_med = ratio_track_optix(m, p, dir, t_med, seed);
   if (T_med <= 0.0f)
      return ZERO;
   // Opaque surfaces between p and the box face (the face itself stays outside the interval)
   constexpr float FACE_GAP = 0.0005f;
   if (t_med > 2.0f * FACE_GAP && trace_shadow(params.traversable, p, dir, 0.0001f, t_med - FACE_GAP))
      return ZERO;
   // Beyond the box: anything on the way to the light, as usual
   if (dist > t_box + FACE_GAP)
      return shadow_transmittance_optix(p + (t_box + FACE_GAP) * dir, dir, dist - t_box - FACE_GAP, seed) * T_med;
   return make_float3(T_med, T_med, T_med);
}

/// Materials whose scattering is not estimated by NEE (mirrors, metals, glass, thin films): the
/// links of a caustic chain. Clear coat is left out (its diffuse lobe takes NEE).
__device__ __forceinline__ bool is_caustic_specular_optix(OptixMaterialType t)
{
   return t == OptixMaterialType::MIRROR || t == OptixMaterialType::METAL || t == OptixMaterialType::ROUGH_MIRROR ||
          t == OptixMaterialType::GLASS || t == OptixMaterialType::DIELECTRIC ||
          t == OptixMaterialType::ANISOTROPIC_METAL || t == OptixMaterialType::THIN_FILM;
}

/// Scattering of the caustic-specular materials, shared by the camera paths (raygen) and the light
/// paths of the caustic pass. rnd() returns the next uniform variate (Sobol for the camera, PCG for
/// the light); the draws happen in the same order for both. Returns false when the path ends.
template <typename Rng>
__device__ __forceinline__ bool scatter_specular_optix(const OptixMaterialData &mat, const PRDRadiance &prd,
                                                       const float3 &cur_direction, Rng &&rnd, float3 &scatter_dir,
                                                       float3 &attenuation)
{
   bool did_scatter = false;
   if (mat.type == OptixMaterialType::MIRROR)
   {
      float3 unit_dir = normalize3(cur_direction);
      scatter_dir = reflect3(unit_dir, prd.hit_normal);
      attenuation = prd.hit_color;
      did_scatter = (dot3(scatter_dir, prd.hit_normal) > 0.0f);
   }
   else if (mat.type == OptixMaterialType::ROUGH_MIRROR ||
            mat.type == OptixMaterialType::METAL) // METAL honours its roughness, as in CUDA
   {
      float3 unit_dir = normalize3(cur_direction);
      float eff_roughness = mat.roughness * params.metal_fuzziness;
      float u1 = rnd();
      float u2 = rnd();
      float u3 = rnd();  // cube root radial
      float z  = 2.0f * u1 - 1.0f;
      float r  = sqrtf(fmaxf(0.0f, 1.0f - z * z));
      float phi = 6.283185307f * u2;
      float t   = cbrtf(u3);
      float3 perturb = make_float3(r * __cosf(phi) * t, r * __sinf(phi) * t, z * t);
      float3 perturbed_n = normalize3(prd.hit_normal + eff_roughness * perturb);
      scatter_dir = reflect3(unit_dir, perturbed_n);
      attenuation = prd.hit_color;
      did_scatter = (dot3(scatter_dir, prd.hit_normal) > 0.0f);
   }
   else if (mat.type == OptixMaterialType::GLASS ||
            mat.type == OptixMaterialType::DIELECTRIC)
   {
      attenuation = make_float3(1.0f, 1.0f, 1.0f);
      float eff_ior = mat.refractive_index * params.glass_ior_multiplier;
      float ri = prd.front_face ? (1.0f / eff_ior) : eff_ior;

      float3 unit_dir = normalize3(cur_direction);
      float cos_theta = fminf(dot3(-unit_dir, prd.hit_normal), 1.0f);
      float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

      bool cannot_refract = ri * sin_theta > 1.0f;
      if (cannot_refract || reflectance(cos_theta, ri) > rnd())
      {
         scatter_dir = reflect3(unit_dir, prd.hit_normal);
      }
      else
      {
         scatter_dir = refract3(unit_dir, prd.hit_normal, ri);
      }
      did_scatter = true;
   }
   else if (mat.type == OptixMaterialType::ANISOTROPIC_METAL)
   {
      // Full anisotropic GGX microfacet conductor (PBR Book §9.6)
      float aspect = sqrtf(fmaxf(1e-4f, 1.0f - 0.9f * mat.anisotropy));
      float r2 = mat.roughness * mat.roughness;
      float alpha_x = fmaxf(1e-4f, r2 / aspect);
      float alpha_y = fmaxf(1e-4f, r2 * aspect);

      float3 N = prd.hit_normal;

      if (fmaxf(alpha_x, alpha_y) < 1e-3f)
      {
         // Nearly smooth: perfect mirror with complex Fresnel tint
         float3 unit_dir = normalize3(cur_direction);
         scatter_dir = reflect3(unit_dir, N);
         float cos_i = fmaxf(dot3(-unit_dir, N), 0.0f);
         attenuation = prd.hit_color * FrComplex_opt(cos_i, mat.eta, mat.k);
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
            float3 wm = Sample_wm_GGX_opt(wo_local, alpha_x, alpha_y, rnd(), rnd());

            // Reflect incoming direction around microfacet normal to get wi
            float3 wi_local = reflect3(make_float3(-wo_local.x, -wo_local.y, -wo_local.z), wm);

            if (wi_local.z > 0.0f)
            {
               // Transform wi back to world space
               float3 wi_world = wi_local.x * T + wi_local.y * B + wi_local.z * N;

               // Fresnel + VNDF weighting: F * G / G1
               float cos_theta_i = fmaxf(dot3(wo_local, wm), 0.0f);
               float3 F = FrComplex_opt(cos_theta_i, mat.eta, mat.k);
               float G = G_GGX_opt(wo_local, wi_local, alpha_x, alpha_y);
               float G1 = G1_GGX_opt(wo_local, alpha_x, alpha_y);

               attenuation = prd.hit_color * F * (G / fmaxf(G1, 1e-6f));
               scatter_dir = wi_world;
               did_scatter = true;
            }
         }
      }
   }
   else if (mat.type == OptixMaterialType::THIN_FILM)
   {
      // Full Airy thin-film interference (soap bubbles / oil slicks).
      // Stochastic reflect/transmit sampled from per-wavelength reflectance.
      float3 unit_dir = normalize3(cur_direction);
      const float3 R  = thin_film_reflectance_optix(mat, fabsf(dot3(-unit_dir, prd.hit_normal)));
      // Reflection chosen with the largest channel's reflectance: a reflected sample then weighs at
      // most 1 per channel. With the average, a film reflecting mostly red sent its rare reflections
      // with red boosted ~3x — coloured fireflies wherever the film mirrors the lamp or a lit floor.
      const float p_refl = fminf(fmaxf(R.x, fmaxf(R.y, R.z)), 0.999f);

      if (rnd() < p_refl)
      {
         // Reflect — carry iridescent color
         scatter_dir = reflect3(unit_dir, prd.hit_normal);
         attenuation = R * (1.0f / fmaxf(p_refl, 0.001f));
         did_scatter = (dot3(scatter_dir, prd.hit_normal) > 0.0f);
      }
      else
      {
         // Transmit — pass straight through
         scatter_dir = unit_dir;
         attenuation = make_float3(1.0f - R.x, 1.0f - R.y, 1.0f - R.z) * (1.0f / (1.0f - p_refl));
         did_scatter = true;
      }
   }
   return did_scatter;
}

/// A vertex the caustic pass can bounce through: a caustic-specular material whose scattering is a
/// delta distribution, so that tracing it from the light estimates the same thing as from the
/// camera. Glossy versions (roughness > 0) stay with the camera paths.
__device__ __forceinline__ bool is_caustic_delta_optix(const OptixMaterialData &mat)
{
   if (!is_caustic_specular_optix(mat.type))
      return false;
   if (mat.type == OptixMaterialType::ROUGH_MIRROR || mat.type == OptixMaterialType::METAL)
      return mat.roughness * params.metal_fuzziness == 0.0f;
   if (mat.type == OptixMaterialType::ANISOTROPIC_METAL)
   {
      // Same test as the near-smooth branch of scatter_specular_optix()
      const float aspect = sqrtf(fmaxf(1e-4f, 1.0f - 0.9f * mat.anisotropy));
      const float r2     = mat.roughness * mat.roughness;
      return fmaxf(fmaxf(1e-4f, r2 / aspect), fmaxf(1e-4f, r2 * aspect)) < 1e-3f;
   }
   return true;
}

/// A straight transmission through a thin film, which shadow rays cross too (see
/// shadow_transmittance_optix): NEE covers the light seen through it, so MIS and the caustic pass
/// treat the film as absent. The normal faces the incoming ray, a reflection stays on its side.
__device__ __forceinline__ bool is_film_pass_optix(const OptixMaterialData &mat, const PRDRadiance &prd,
                                                   const float3 &scatter_dir)
{
   return params.has_thin_film && mat.type == OptixMaterialType::THIN_FILM && dot3(scatter_dir, prd.hit_normal) < 0.0f;
}

/// Solid-angle pdf of scatter_specular_optix()'s rough-mirror lobe: the normal n + r * p, p uniform
/// in the unit ball, reflects the unit direction d_in to wo. The perturbed normal m is the half
/// vector of (-d_in, wo); its pdf integrates the ball's density 3 / (4 pi r^3) along the ray t * m
/// over the chord inside the ball |t m - n| < r, and the reflection maps it to wo with the usual
/// 1 / (4 |wo.m|). The lobe's weight is its albedo, so f * cos = albedo * pdf (wo above the surface).
__device__ __forceinline__ float glossy_lobe_pdf_optix(const float3 &n, const float3 &d_in, const float3 &wo, float r)
{
   if (r <= 0.0f || dot3(wo, n) <= 0.0f)
      return 0.0f;
   const float3 h = wo - d_in;
   const float  hl = length3(h);
   if (hl < 1e-6f)
      return 0.0f;
   // reflect(d, m) == reflect(d, -m): at grazing incidence the sampled normal faces away from the
   // ray and wo - d_in points along -m, so take the side of the ball around n (one side for r < 1)
   float3 m = h / hl;
   float  c = dot3(m, n);
   if (c < 0.0f)
   {
      m = -m;
      c = -c;
   }
   const float disc = c * c - (1.0f - r * r);
   if (disc < 0.0f)
      return 0.0f;
   const float s  = sqrtf(disc);
   const float t2 = c + s;
   const float t1 = fmaxf(0.0f, c - s);
   const float pdf_m = (t2 * t2 * t2 - t1 * t1 * t1) / (4.0f * 3.14159265f * r * r * r);
   return pdf_m / (4.0f * fmaxf(fabsf(dot3(wo, m)), 1e-6f));
}

/// A glossy lobe that light sampling can serve: rough mirrors and metals, the coat of a clear coat.
/// Their smooth versions are delta distributions and stay with the BSDF. (Anisotropic GGX metals
/// are left out for now: a VNDF-pdf version came out 0.3% darker than BSDF sampling alone.)
__device__ __forceinline__ bool has_glossy_lobe_optix(const OptixMaterialData &mat)
{
   if (mat.type == OptixMaterialType::ROUGH_MIRROR || mat.type == OptixMaterialType::METAL)
      return mat.roughness * params.metal_fuzziness > 0.0f;
   if (mat.type == OptixMaterialType::CLEAR_COAT)
      return mat.roughness > 0.0f;
   return false;
}

/// The glossy lobe of has_glossy_lobe_optix() towards the unit direction wl, for a ray arriving along
/// the unit direction d_in: f * cos and the solid-angle pdf of the lobe's own sampling, its selection
/// probability included (the Fresnel coin of a clear coat). The two describe the same sampler as
/// scatter_specular_optix() / the clear-coat branch of raygen, so the MIS weights pair up: the
/// normal is perturbed in a ball and the sample weighs its albedo (coat: 1), so f * cos =
/// albedo * pdf, the pdf from glossy_lobe_pdf_optix(). Returns false when wl lies outside the lobe.
__device__ bool glossy_lobe_eval_optix(const OptixMaterialData &mat, const PRDRadiance &prd, const float3 &d_in,
                                       const float3 &wl, float3 &f_cos, float &pdf)
{
   const float3 n = prd.hit_normal;
   if (mat.type == OptixMaterialType::CLEAR_COAT)
   {
      // Schlick Fresnel of the coat, as in raygen's clear-coat branch
      const float cos_theta = fminf(dot3(-d_in, n), 1.0f);
      float r0 = (1.0f - mat.refractive_index) / (1.0f + mat.refractive_index);
      r0 = r0 * r0;
      const float cx      = 1.0f - cos_theta;
      const float fresnel = r0 + (1.0f - r0) * (cx * cx * cx * cx * cx);
      pdf   = fresnel * glossy_lobe_pdf_optix(n, d_in, wl, mat.roughness);
      f_cos = make_float3(pdf, pdf, pdf); // the coat is clear
      return pdf > 0.0f;
   }
   pdf   = glossy_lobe_pdf_optix(n, d_in, wl, mat.roughness * params.metal_fuzziness);
   f_cos = prd.hit_color * pdf;
   return pdf > 0.0f;
}

/// Light-list index of a custom-primitive hit, -1 when it is not a NEE light
__device__ __forceinline__ int light_index_of_hit(const PRDRadiance &prd)
{
   const int prim = prd.hit_prim_idx;
   return (prim >= 0 && prim < params.num_geom_entries && params.geom_to_light_map) ? params.geom_to_light_map[prim]
                                                                                     : -1;
}

extern "C" __global__ void __raygen__rg()
{
   // One thread per pixel, looping over the batch's samples. (A width x height x spp launch with
   // one path per thread and atomic accumulation was measured 15-20% slower, so the loop stays.)
   const uint3 idx = optixGetLaunchIndex();
   const unsigned int x = idx.x;
   const unsigned int y = idx.y;

   if (x >= params.width || y >= params.height)
      return;

   const unsigned int pixel_idx = y * params.width + x;

   // Adaptive sampling: converged pixels (negative count) take no more samples, except while
   // they are being re-checked (streak == ADAPTIVE_RECHECK_MARK, see optixAdaptiveUpdateKernel)
   if (params.pixel_sample_counts && params.pixel_sample_counts[pixel_idx] < 0 &&
       !(params.conv_streak && params.conv_streak[pixel_idx] == ADAPTIVE_RECHECK_MARK))
      return;

   // Per-pixel stable hash for Sobol scrambling (computed once, reused every sample).
   const uint32_t pixel_hash = pcg_hash(pixel_idx ^ (0xdeadbeef + params.sobol_sequence_seed * 0x9e3779b9u));

   unsigned int seed; // PCG state, re-seeded at every sample (see the sample loop)

   // Load accumulated color. The w component counts the samples whose primary ray hit any
   // geometry: the adaptive-sampling pass uses it to tell a pixel that only ever sees the
   // (possibly black) background from one whose light simply has not been found yet.
   float4 acc = params.accum_buffer[pixel_idx];
   float3 accumulated = make_float3(acc.x, acc.y, acc.z);
   float  primary_hits = acc.w;
   float  lum_sq_added = 0.0f; // sum of squared sample luminance for this launch (variance estimate)
   // Denoiser guides for this launch: sums of the albedo and camera-space normal seen by each sample
   // (see the guide capture in the bounce loop).
   float3 guide_albedo = make_float3(0.0f, 0.0f, 0.0f);
   float3 guide_normal = make_float3(0.0f, 0.0f, 0.0f);

   // Firefly rejection base level (see the clamp at the end of the sample loop). Mirrored in
   // render_acc_kernel.cu — keep both in sync.
   constexpr float FIREFLY_CLAMP_BASE = 20.0f;
   // Samples this pixel already holds: its own count under adaptive sampling (converged pixels
   // skip launches, so it lags the global count), the global count otherwise.
   const int pixel_samples_before =
       params.pixel_sample_counts ? abs(params.pixel_sample_counts[pixel_idx]) : params.total_samples_so_far;

   for (int s = 0; s < params.samples_per_launch; ++s)
   {
      // total_samples_so_far + s is the global sample number; Gray-code encoding
      // improves progressive convergence (each successive index flips one bit).
      const uint32_t abs_sample = (uint32_t)(params.total_samples_so_far + s);
      const uint32_t sobol_sample_idx = sobol_gray(abs_sample);
      // The PCG stream of a sample depends only on (pixel, sample number, accumulation), never on
      // how the samples were split into launches: the same render gives the same image. The
      // sample number is hashed on its own first, or pixel p of one sample would replay pixel q of
      // another.
      seed = pcg_hash(pixel_idx + pcg_hash(abs_sample ^ (params.sobol_sequence_seed * 0x9e3779b9u)));

      // Jittered pixel sample
      // dims 0, 1 — pixel AA jitter
      float offset_u, offset_v;
      if (params.use_sobol) {
         offset_u = sobol_float(sobol_sample_idx, 0, pixel_hash) - 0.5f;
         offset_v = sobol_float(sobol_sample_idx, 1, pixel_hash) - 0.5f;
      } else {
         offset_u = rand_float(seed) - 0.5f;
         offset_v = rand_float(seed) - 0.5f;
      }

      float3 pixel_center = params.pixel00_loc + ((float)x + offset_u) * params.pixel_delta_u +
                             ((float)y + offset_v) * params.pixel_delta_v;
      float3 ray_direction = pixel_center - params.camera_center;
      float3 ray_origin = params.camera_center;

      // Depth of field
      if (params.dof_enabled && params.dof_aperture > 0.0f)
      {
         float3 normalized_dir = normalize3(ray_direction);
         float3 focus_point = params.camera_center + params.dof_focus_distance * normalized_dir;

         // dims 2, 3 — aperture disk sample
         float2 disk;
         if (params.use_sobol) {
            float du = sobol_float(sobol_sample_idx, 2, pixel_hash) * 2.0f - 1.0f;
            float dv = sobol_float(sobol_sample_idx, 3, pixel_hash) * 2.0f - 1.0f;
            // Uniform disk from (u,v) via Shirley-Chiu concentric mapping (every sample lands in
            // the disk: collapsing the ~21 % outside it to the centre added a pinhole image)
            float r, theta;
            if (du == 0.0f && dv == 0.0f) { r = 0.0f; theta = 0.0f; }
            else if (fabsf(du) > fabsf(dv)) { r = du; theta = 0.785398163f * (dv / du); }
            else { r = dv; theta = 1.570796327f - 0.785398163f * (du / dv); }
            disk = make_float2(r * cosf(theta), r * sinf(theta));
         } else {
            disk = rand_in_unit_disk(seed);
         }
         float3 aperture_offset = params.dof_aperture * (disk.x * params.cam_u + disk.y * params.cam_v);
         ray_origin = params.camera_center + aperture_offset;
         ray_direction = focus_point - ray_origin;
      }
      // Unit directions throughout: tmin is then a distance, the same for every ray
      ray_direction = normalize3(ray_direction);

      // Path tracing loop
      float3 color = make_float3(0.0f, 0.0f, 0.0f);
      float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

      float3 cur_origin = ray_origin;
      float3 cur_direction = ray_direction;
      float3 mis_origin = ray_origin; // where the last direction was sampled (light pdf for MIS)
      bool   mis_from_camera = true;  // no scattering vertex yet (thin films passed through do not count)

      // MIS tracking
      float prev_bsdf_pdf = 1.0f;
      bool  prev_is_delta = true; // camera ray treated as delta

      // Sobol state is raygen-local; dim_idx persists across bounces (dims 0-3 used for AA + DOF above)
      SobolState ss;
      ss.sample_idx = sobol_sample_idx;
      ss.dim_idx    = 4u;
      ss.pixel_hash = pixel_hash;

      bool guide_done = false; // this sample's denoiser guide has been recorded
      int  volume_passes = 0;  // medium back faces passed through (see VOLUME below)
      // Indirect clamp bookkeeping: what the sample holds when it leaves its first diffuse
      // surface (or first scattering event in a medium) is direct light — the emitters and the
      // sky seen by the camera, possibly through mirrors and glass, plus that surface's NEE.
      // Everything gathered after is indirect and gets its own ceiling at the end of the sample.
      bool   indirect_started = false;
      float3 direct_color     = make_float3(0.0f, 0.0f, 0.0f);
      // Caustic pass bookkeeping: 1 = the camera ray met a diffuse surface first, 2 = and only
      // delta surfaces since; 0 = not a path of the pass. A path in state 2 that reaches a NEE light
      // belongs to the caustic pass and is not counted here.
      int  caustic_state  = 0;
      bool passed_through = false; // the camera ray crossed invisible geometry or a medium first
      bool films_only     = true;  // every vertex so far was a straight pass through a thin film

      for (int bounce = 0; bounce < params.max_depth; ++bounce)
      {
         PRDRadiance prd;
         prd.hit = false;

         trace(params.traversable, cur_origin, cur_direction, 0.001f, 1e16f, &prd);

         // Invisible geometry (lights kept out of the picture): camera rays pass through it,
         // bounced rays interact normally
         if (bounce == 0 && prd.hit && !prd.visible)
         {
            passed_through = true; // the caustic pass cannot see through it
            cur_origin = prd.hit_point + 0.01f * normalize3(cur_direction);
            --bounce;
            continue;
         }

         if (bounce == 0 && prd.hit)
            primary_hits += 1.0f;

         if (!prd.hit)
         {
            // Sky/background
            float3 unit_dir = normalize3(cur_direction);
            float3 sky;
            float  w_sky = 1.0f; // MIS weight against the sky sampler (NEE covers the same directions)
            if (params.use_hdr_env)
            {
               const float2 uv   = env_dir_to_uv(unit_dir);
               const float4 samp = tex2D<float4>(params.hdr_env_tex, uv.x, uv.y);
               sky = make_float3(samp.x, samp.y, samp.z);
               if (!prev_is_delta && env_sampling_active())
               {
                  const float lp = env_pdf_optix(unit_dir) * env_select_prob();
                  if (lp > 0.0f)
                  {
                     const float a = prev_bsdf_pdf * prev_bsdf_pdf;
                     w_sky = a / (a + lp * lp);
                  }
               }
            }
            else
            {
               float t = 0.5f * (unit_dir.y + 1.0f);
               sky = (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + t * params.background_color;
            }
            color = color + throughput * sky * (params.background_intensity * w_sky);
            if (!guide_done)
            {
               // Background seen directly (or through specular surfaces): the sky colour is the
               // albedo guide, there is no surface so the normal guide stays zero.
               guide_albedo = guide_albedo + make_float3(fminf(sky.x, 1.0f), fminf(sky.y, 1.0f), fminf(sky.z, 1.0f));
               guide_done   = true;
            }
            break;
         }

         // Material parameters are read straight from the material table — only the
         // index crosses the trace() boundary.
         if (prd.material_idx < 0)
         {
            color = color + throughput * make_float3(1.0f, 0.0f, 1.0f); // Magenta = invalid material
            break;
         }
         const OptixMaterialData &mat = params.materials[prd.material_idx];

         const int caustic_state_before = caustic_state; // restored when a thin film is passed through
         if (params.caustics_active)
         {
            if (bounce == 0 || films_only) // the camera ray's first vertex, seen through any films
            {
               if (mat.type == OptixMaterialType::VOLUME)
                  passed_through = true;
               caustic_state = (!passed_through && (mat.type == OptixMaterialType::LAMBERTIAN ||
                                                    mat.type == OptixMaterialType::SDF_MATERIAL))
                                   ? 1
                                   : 0;
            }
            else if (caustic_state > 0 && is_caustic_delta_optix(mat))
               caustic_state = 2;
            else if (mat.type != OptixMaterialType::LIGHT)
               caustic_state = 0;
         }

         // Denoiser guide: the albedo and normal of the first surface that is not a perfect
         // specular (mirror / glass) — what shows through a window or a mirror is what the
         // denoiser must preserve, not the pane itself. After two specular bounces the surface
         // is recorded anyway so every sample contributes a guide.
         if (!guide_done)
         {
            const bool specular = mat.type == OptixMaterialType::MIRROR || mat.type == OptixMaterialType::GLASS ||
                                  mat.type == OptixMaterialType::DIELECTRIC || mat.type == OptixMaterialType::THIN_FILM;
            if (!specular || bounce >= 2)
            {
               const float3 n = prd.hit_normal;
               guide_albedo = guide_albedo + make_float3(fminf(prd.hit_color.x, 1.0f), fminf(prd.hit_color.y, 1.0f),
                                                         fminf(prd.hit_color.z, 1.0f));
               guide_normal = guide_normal + make_float3(dot3(n, params.cam_u), dot3(n, params.cam_v), dot3(n, params.cam_w));
               guide_done   = true;
            }
         }

         // --- Participating medium: the mesh surface is only the boundary of the volume ---
         if (mat.type == OptixMaterialType::VOLUME)
         {
            float3 dir = normalize3(cur_direction);
            // Entering through a front face: the walk starts at the surface. A back face means the
            // ray already was inside the medium (camera or a surface inside it): the walk starts at
            // the ray origin and this hit is its first far side.
            const bool started_inside = !prd.front_face;
            if (started_inside && ++volume_passes > VOLUME_MAX_PASSES)
               break;
            // Delta tracking through the medium: free flights against the majorant, a step is
            // a real scattering event with probability density / majorant, else a null collision.
            const float sigma_max = volume_majorant(mat.volume);
            float3 origin    = started_inside ? cur_origin : prd.hit_point;
            bool   escaped   = false;
            bool   have_exit = started_inside;
            PRDRadiance ex_known = prd;
            bool   scattered_any  = false;
            float3 scatter_point  = origin; // last scattering point (MIS origin for the exit path)
            float  last_phase_pdf = 1.0f;
            for (int ev = 0; ev < VOLUME_MAX_EVENTS && !escaped; ++ev)
            {
               PRDRadiance ex;
               if (have_exit)
               {
                  ex        = ex_known;
                  have_exit = false;
               }
               else
               {
                  ex.hit = false;
                  trace(params.traversable, origin, dir, 0.0001f, 1e16f, &ex);
               }
               if (!ex.hit)
               {
                  escaped = true; // open mesh: nothing ahead, leave towards the sky
                  break;
               }
               const float3 to_exit = ex.hit_point - origin;
               const float  t_exit  = sqrtf(dot3(to_exit, to_exit));
               bool  scattered = false;
               float t = 0.0f;
               for (int k = 0; k < VOLUME_MAX_NULL_STEPS; ++k)
               {
                  {
                     // Jump over the empty space around the shape (no density there to collide with)
                     const float3 pk   = origin + t * dir;
                     const float  skip = volume_empty_skip(mat.volume, pk.x, pk.y, pk.z);
                     if (skip > 0.0f)
                     {
                        t += skip;
                        if (t >= t_exit)
                           break;
                        continue;
                     }
                  }
                  t -= logf(1.0f - rand_float(seed)) / sigma_max;
                  if (t >= t_exit)
                     break;
                  float3 p = origin + t * dir;
                  if (rand_float(seed) * sigma_max <
                      volume_density(mat.volume, p.x, p.y, p.z))
                  {
                     scattered = true;
                     origin    = p;
                     break;
                  }
               }
               if (!scattered)
               {
                  // Out through the far side of the medium, or stopped by another surface inside
                  // it (left for the next bounce to shade). The next trace starts with tmin 0.001
                  // from a point set back along the ray so that the far face itself falls inside
                  // the skipped band while a surface just behind it (the ground under a cloud)
                  // does not: 0.0002 units beyond the face are skipped, nothing more.
                  const bool far_side = ex.material_idx >= 0 &&
                                        params.materials[ex.material_idx].type == OptixMaterialType::VOLUME &&
                                        !ex.front_face;
                  origin = far_side ? ex.hit_point - 0.0008f * dir : ex.hit_point - 0.0012f * dir;
                  escaped = true;
                  break;
               }
               // Scattering event: attenuate by the albedo
               throughput    = throughput * prd.hit_color;
               scattered_any = true;
               scatter_point = origin;

               // Volumetric NEE: sample a light, weight by the phase function and the transmittance
               // towards it (ratio tracking), MIS against phase-function sampling
               if (params.mis_enabled && nee_available())
               {
                  const float nee_u1 = rand_float(seed), nee_u2 = rand_float(seed), nee_sel = rand_float(seed);
                  OptiXLightSample ls = sample_nee_optix(origin, nee_sel, nee_u1, nee_u2);
                  if (ls.pdf > 0.0f)
                  {
                     const float3 T = volume_nee_transmittance_optix(mat, origin, ls.direction, ls.dist - 0.002f, seed);
                     if (fmaxf(T.x, fmaxf(T.y, T.z)) > 0.0f)
                     {
                        const float ph = hg_phase(mat.anisotropy, dot3(dir, ls.direction));
                        const float a = ls.pdf * ls.pdf, b = ph * ph;
                        const float w = (a + b > 0.0f) ? a / (a + b) : 1.0f;
                        color = color + throughput * ls.emission * T * (ph * w / ls.pdf);
                     }
                  }
               }

               if (!indirect_started)
               {
                  direct_color     = color;
                  indirect_started = true;
               }

               // New direction from the Henyey-Greenstein phase function
               const float cos_t = hg_sample_cos(mat.anisotropy, rand_float(seed));
               const float sin_t = sqrtf(fmaxf(0.0f, 1.0f - cos_t * cos_t));
               const float phi   = 6.283185307f * rand_float(seed);
               float3 a  = fabsf(dir.x) > 0.9f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
               float3 b1 = normalize3(cross3(a, dir));
               float3 b2 = cross3(dir, b1);
               dir = normalize3(sin_t * __cosf(phi) * b1 + sin_t * __sinf(phi) * b2 + cos_t * dir);
               last_phase_pdf = hg_phase(mat.anisotropy, cos_t);
               if (ev >= 8)
               {
                  float max_c = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
                  float surv  = fminf(max_c, 0.95f);
                  if (rand_float(seed) > surv)
                  {
                     escaped = false;
                     break;
                  }
                  throughput = throughput / surv;
               }
            }
            if (!escaped)
               break; // path cut inside the medium (roulette or event budget)
            cur_origin    = origin;
            cur_direction = dir;
            if (scattered_any)
            {
               // An emissive hit on the exit path is MIS-weighted against the phase function
               // sampled at the last scattering point (the exit itself is a pass-through)
               prev_bsdf_pdf = last_phase_pdf;
               prev_is_delta = false;
               mis_origin    = scatter_point;
               mis_from_camera = false;
            }
            else if (++volume_passes <= VOLUME_MAX_PASSES)
               --bounce; // crossed without scattering: not a bounce (keeps the surface behind at depth 0)
            continue;
         }

         if (mat.type == OptixMaterialType::LIGHT)
         {
            // Light -> delta surfaces -> the diffuse surface the camera sees: the caustic pass
            // carries this path
            if (params.caustics_active && caustic_state == 2 && light_index_of_hit(prd) >= 0)
               break;

            // MIS weight for BSDF-sampled emissive hit
            float w = 1.0f;
            if (params.mis_enabled && bounce > 0 && !prev_is_delta && params.num_lights > 0)
            {
               int prim_idx = prd.hit_prim_idx;
               int light_idx = -1;
               if (prim_idx >= 0 && prim_idx < params.num_geom_entries && params.geom_to_light_map)
                  light_idx = params.geom_to_light_map[prim_idx];
               if (light_idx >= 0)
               {
                  float3 hit_dir = normalize3(cur_direction);
                  float lp = light_dir_pdf_optix(light_idx, mis_origin, hit_dir) * (1.0f - env_select_prob());
                  if (lp > 0.0f)
                  {
                     float a = prev_bsdf_pdf * prev_bsdf_pdf;
                     float b = lp * lp;
                     w = a / (a + b);
                  }
               }
            }
            // Camera rays see the emitter as is; lighting of surfaces and media gets the falloff
            const float fall = mis_from_camera ? 1.0f : light_falloff_factor(length3(prd.hit_point - mis_origin));
            color = color + throughput * mat.emission * params.light_intensity * (w * fall);
            break;
         }

         if (mat.type == OptixMaterialType::SHOW_NORMALS)
         {
            color = color + throughput * make_float3(0.5f * (prd.hit_normal.x + 1.0f), 0.5f * (prd.hit_normal.y + 1.0f),
                                                      0.5f * (prd.hit_normal.z + 1.0f));
            break;
         }

         if (mat.type == OptixMaterialType::CONSTANT)
         {
            color = color + throughput * prd.hit_color;
            break;
         }

         // Scatter based on material
         float3 scatter_dir;
         float3 attenuation;
         bool did_scatter = false;

         // Lobe bookkeeping for NEE / MIS. NEE only ever estimates the cosine-weighted
         // diffuse lobe, so it must use that lobe's true weight in the material's BSDF,
         // and only a bounce that actually sampled the diffuse lobe gets MIS-weighted at
         // the next emissive hit. Specular / coat bounces are treated as delta (weight 1,
         // never covered by NEE) — the same split the CUDA renderer uses.
         float diffuse_lobe   = 1.0f; // weight of the diffuse lobe in the BSDF (1 for Lambertian)
         bool  sampled_diffuse = true; // this bounce's direction came from the diffuse lobe

         if (mat.type == OptixMaterialType::LAMBERTIAN ||
             mat.type == OptixMaterialType::SDF_MATERIAL)
         {
            // Lambertian: cosine-weighted hemisphere sampling
            // Uses 2 Sobol dims
            float u1 = sobol_rand_float(ss, seed);
            float u2 = sobol_rand_float(ss, seed);
            float z  = 2.0f * u1 - 1.0f;
            float r  = sqrtf(fmaxf(0.0f, 1.0f - z * z));
            float phi = 6.283185307f * u2;
            float3 rnd_unit = make_float3(r * __cosf(phi), r * __sinf(phi), z);
            scatter_dir = prd.hit_normal + rnd_unit;
            // Catch degenerate direction
            if (fabsf(scatter_dir.x) < 1e-8f && fabsf(scatter_dir.y) < 1e-8f && fabsf(scatter_dir.z) < 1e-8f)
               scatter_dir = prd.hit_normal;
            scatter_dir = normalize3(scatter_dir);
            attenuation = prd.hit_color;
            did_scatter = true;
         }
         else if (is_caustic_specular_optix(mat.type))
         {
            did_scatter = scatter_specular_optix(mat, prd, cur_direction, [&]() { return sobol_rand_float(ss, seed); },
                                                 scatter_dir, attenuation);
         }
         else if (mat.type == OptixMaterialType::CLEAR_COAT)
         {
            // Two-layer: Schlick Fresnel decides specular coat vs diffuse base
            float3 unit_dir = normalize3(cur_direction);
            float cos_theta = fminf(dot3(-unit_dir, prd.hit_normal), 1.0f);
            float r0 = (1.0f - mat.refractive_index) / (1.0f + mat.refractive_index);
            r0 = r0 * r0;
            float cx = 1.0f - cos_theta;
            float cx2 = cx * cx;
            float fresnel = r0 + (1.0f - r0) * (cx2 * cx2 * cx);

            diffuse_lobe = 1.0f - fresnel;
            if (sobol_rand_float(ss, seed) < fresnel)
            {
               // Specular reflection through the coat
               sampled_diffuse = false;
               float u1 = sobol_rand_float(ss, seed);
               float u2 = sobol_rand_float(ss, seed);
               float u3 = sobol_rand_float(ss, seed);
               float z  = 2.0f * u1 - 1.0f;
               float r  = sqrtf(fmaxf(0.0f, 1.0f - z * z));
               float phi_r = 6.283185307f * u2;
               float t   = cbrtf(u3);
               float3 perturb = make_float3(r * __cosf(phi_r) * t, r * __sinf(phi_r) * t, z * t);
               float3 perturbed_n = normalize3(prd.hit_normal + mat.roughness * perturb);
               scatter_dir = reflect3(unit_dir, perturbed_n);
               attenuation = make_float3(1.0f, 1.0f, 1.0f); // coat is clear
               did_scatter = (dot3(scatter_dir, prd.hit_normal) > 0.0f);
            }
            else
            {
               // Diffuse base color
               float u1 = sobol_rand_float(ss, seed);
               float u2 = sobol_rand_float(ss, seed);
               float z  = 2.0f * u1 - 1.0f;
               float r  = sqrtf(fmaxf(0.0f, 1.0f - z * z));
               float phi_r = 6.283185307f * u2;
               float3 rnd_unit = make_float3(r * __cosf(phi_r), r * __sinf(phi_r), z);
               scatter_dir = prd.hit_normal + rnd_unit;
               if (fabsf(scatter_dir.x) < 1e-8f && fabsf(scatter_dir.y) < 1e-8f && fabsf(scatter_dir.z) < 1e-8f)
                  scatter_dir = prd.hit_normal;
               scatter_dir = normalize3(scatter_dir);
               attenuation = prd.hit_color;
               did_scatter = true;
            }
         }

         if (!did_scatter)
         {
            break;
         }

         // Straight through a thin film (a soap bubble): for MIS, the light falloff and the caustic
         // pass the path carries on as if the film were not there — the light pdf at the next
         // emissive hit or sky escape is still measured from the last real scattering vertex.
         const bool film_pass = is_film_pass_optix(mat, prd, scatter_dir);
         if (film_pass)
            caustic_state = caustic_state_before;
         else
            films_only = false;

         // --- NEE: direct light sampling for non-delta materials ---
         bool is_delta = is_delta_material_optix(mat.type);
         // The first bounce carries most of the direct light and always gets NEE; deeper
         // bounces do it on 1 of every nee_stride samples (contribution x stride, unbiased).
         // Applying the stride at bounce 0 shows up as bright speckle at low sample counts.
         bool do_nee = true;
         float nee_scale = 1.0f;
         if (bounce > 0 && params.nee_stride > 1)
         {
            do_nee = ((abs_sample % (uint32_t)params.nee_stride) == 0u);
            nee_scale = (float)params.nee_stride;
         }

         // Glossy NEE: a rough mirror, metal or lacquer coat reflecting a small bright source (the sun
         // of an HDR sky, a lamp) found it only by chance through its own lobe, leaving fireflies that
         // took thousands of samples to fade. Its lobe pdf is known in closed form, so it gets light
         // sampling too, MIS-weighted against that lobe (the weight at the next emissive hit or sky
         // escape uses prev_bsdf_pdf below). Otherwise the vertex is treated as before: metals stay
         // "delta" for the diffuse NEE, the indirect clamp and the caustic pass; a clear coat keeps
         // its diffuse NEE beside this one (one strategy pair per lobe).
         const bool glossy = has_glossy_lobe_optix(mat);
         if (params.mis_enabled && glossy && do_nee && nee_available())
         {
            float max_att = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            if (max_att > 1e-4f)
            {
               float nee_u1  = sobol_rand_float(ss, seed);
               float nee_u2  = sobol_rand_float(ss, seed);
               float nee_sel = rand_float(seed);

               OptiXLightSample ls = sample_nee_optix(prd.hit_point, nee_sel, nee_u1, nee_u2);
               float3 f_cos;
               float  p_mat = 0.0f;
               if (ls.pdf > 0.0f &&
                   glossy_lobe_eval_optix(mat, prd, normalize3(cur_direction), ls.direction, f_cos, p_mat))
               {
                  const float3 T_sh = shadow_transmittance_optix(prd.hit_point + 0.0001f * prd.hit_normal, ls.direction,
                                                                ls.dist - 0.002f, seed);
                  if (fmaxf(T_sh.x, fmaxf(T_sh.y, T_sh.z)) > 0.0f)
                  {
                     const float a     = ls.pdf * ls.pdf;
                     const float w_nee = a / (a + p_mat * p_mat);
                     color = color + throughput * f_cos * ls.emission * T_sh * (w_nee * nee_scale / ls.pdf);
                  }
               }
            }
         }

         if (params.mis_enabled && !is_delta && nee_available())
         {
            if (do_nee)
            {
               float max_att = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
               if (max_att > 1e-4f)
               {
                  float nee_u1  = sobol_rand_float(ss, seed);
                  float nee_u2  = sobol_rand_float(ss, seed);
                  float nee_sel = rand_float(seed);

                  OptiXLightSample ls = sample_nee_optix(prd.hit_point, nee_sel, nee_u1, nee_u2);
                  // A sky sample is drawn over the whole sphere: below the surface it contributes
                  // nothing, so the shadow ray is skipped (half of them on a ground plane)
                  float cos_th = (ls.pdf > 0.0f) ? fmaxf(0.0f, dot3(ls.direction, prd.hit_normal)) : 0.0f;
                  if (cos_th > 0.0f)
                  {
                     // Shadow ray (transmittance through any volume on the way)
                     const float3 T_sh = shadow_transmittance_optix(prd.hit_point + 0.0001f * prd.hit_normal,
                                                                   ls.direction, ls.dist - 0.002f, seed);
                     if (fmaxf(T_sh.x, fmaxf(T_sh.y, T_sh.z)) > 0.0f)
                     {
                        // Diffuse lobe only: f = w_d * albedo / pi, pdf = w_d * cos(theta) / pi
                        float3 f_nee = prd.hit_color * (diffuse_lobe / 3.14159265f);
                        float p_mat  = diffuse_lobe * cos_th / 3.14159265f;
                        float a      = ls.pdf * ls.pdf;
                        float b      = p_mat * p_mat;
                        float w_nee  = (a + b > 0.0f) ? (a / (a + b)) : 1.0f;
                        color = color + throughput * f_nee * ls.emission * T_sh * (cos_th * w_nee * nee_scale / ls.pdf);
                     }
                  }
               }
            }
         }

         if (!is_delta && !indirect_started)
         {
            direct_color     = color;
            indirect_started = true;
         }

         throughput = throughput * attenuation;
         cur_origin = prd.hit_point;
         if (!film_pass)
         {
            mis_origin      = prd.hit_point;
            mis_from_camera = false;
         }
         const float3 cur_direction_in = normalize3(cur_direction);
         cur_direction = scatter_dir;

         // Track BSDF PDF for MIS weighting at the next emissive hit
         if (film_pass)
         {
            // Unchanged: the direction still comes from the last real scattering vertex
         }
         else if (params.mis_enabled && glossy && nee_available() &&
                  !(mat.type == OptixMaterialType::CLEAR_COAT && sampled_diffuse))
         {
            // Glossy lobe sampled, the glossy NEE above covers the same directions (on 1 of every
            // nee_stride samples beyond the first bounce, scaled up: the weight applies to all)
            float3 f_cos;
            prev_bsdf_pdf = 0.0f;
            glossy_lobe_eval_optix(mat, prd, cur_direction_in, scatter_dir, f_cos, prev_bsdf_pdf);
            prev_is_delta = (prev_bsdf_pdf <= 0.0f);
         }
         else if (is_delta || !sampled_diffuse)
         {
            prev_bsdf_pdf = 1.0f;
            prev_is_delta = true;
         }
         else
         {
            prev_bsdf_pdf = diffuse_lobe * fmaxf(0.0f, dot3(normalize3(scatter_dir), prd.hit_normal)) / 3.14159265f;
            prev_is_delta = false;
         }

         // Russian Roulette (from bounce 1)
         if (bounce > 0)
         {
            float max_comp = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            float survival_prob = fminf(max_comp, 0.95f);
            if (sobol_rand_float(ss, seed) > survival_prob)
            {
               break;
            }
            throughput = throughput / survival_prob;
         }
      }

      // Indirect clamp: fixed ceiling on the light gathered after the first diffuse bounce. Those
      // paths reach an emitter or the sun through glass or off a mirror (caustics), which no light
      // sampling can target; under the growing clamp below their speckle only fades as 1/sqrt(N).
      // Biased (caustics get dimmer), but it converges. Hue-preserving, like the clamp below.
      if (params.indirect_clamp > 0.0f && indirect_started)
      {
         float3 indirect = color - direct_color;
         const float ind_lum = 0.2126f * indirect.x + 0.7152f * indirect.y + 0.0722f * indirect.z;
         if (ind_lum > params.indirect_clamp)
            color = direct_color + indirect * (params.indirect_clamp / ind_lum);
      }

      // Firefly rejection: clamp per-sample luminance so that one very bright path (a sun texel
      // in an HDRI, a small light hit by chance from a diffuse surface) cannot leave a permanent
      // white dot. The ceiling grows with the square root of the samples the pixel already holds:
      // the first samples are clamped hard (clean interactive preview), then the clamp relaxes as
      // the mean settles, so the estimator stays consistent. A fixed ceiling would freeze in a
      // bias — with a 12 cm light and no NEE the image converged to a value 30x too dark.
      // While the camera moves each frame is its own short accumulation of a few samples, so the
      // host imposes a low fixed ceiling instead: a sample of a few units would be displayed as a
      // bright dot right away. Hue-preserving scale: only the luminance is reduced.
      const float firefly_clamp = (params.firefly_clamp_ceiling > 0.0f)
                                      ? params.firefly_clamp_ceiling
                                      : FIREFLY_CLAMP_BASE * sqrtf((float)(pixel_samples_before + s + 1));
      float sample_lum = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
      if (sample_lum > firefly_clamp)
      {
         float scale = firefly_clamp / sample_lum;
         color = color * scale;
      }

      accumulated = accumulated + color;
      const float lum601 = 0.299f * color.x + 0.587f * color.y + 0.114f * color.z;
      lum_sq_added += lum601 * lum601;
   }

   params.accum_buffer[pixel_idx] = make_float4(accumulated.x, accumulated.y, accumulated.z, primary_hits);
   if (params.lum_sq_buffer)
      params.lum_sq_buffer[pixel_idx] += lum_sq_added;
   if (params.albedo_buffer)
   {
      float4 ga = params.albedo_buffer[pixel_idx];
      params.albedo_buffer[pixel_idx] = make_float4(ga.x + guide_albedo.x, ga.y + guide_albedo.y, ga.z + guide_albedo.z, 0.0f);
      float4 gn = params.normal_buffer[pixel_idx];
      params.normal_buffer[pixel_idx] = make_float4(gn.x + guide_normal.x, gn.y + guide_normal.y, gn.z + guide_normal.z, 0.0f);
   }
}

//==============================================================================
// CAUSTIC PASS — light tracing (one light path per launch index)
//
// A light path leaves a point of a NEE light towards the delta surfaces (mirrors, glass), follows
// them with the camera paths' own scattering code, and at the first diffuse surface after at least
// one of them is joined to the camera by an occlusion ray: its contribution lands in the pixel the
// surface projects to. These are the paths light -> delta+ -> diffuse -> camera that the camera
// paths would only find by chance (a diffuse bounce through the glass onto the lamp); raygen
// leaves them to this pass (caustic_state 2). Both passes run on the same accumulation buffer,
// this one scaled so that it adds the same number of "samples" per pixel as the camera pass.
//==============================================================================

/// Solid-angle pdf of the target mixture at direction w from p: the average, over the targets, of
/// the uniform-cone pdf of each target's bounding sphere when w lies in its cone
__device__ float caustic_direction_pdf(const float3 &p, const float3 &w)
{
   float sum = 0.0f;
   for (int k = 0; k < params.num_caustic_targets; ++k)
   {
      const float4 t  = params.caustic_targets[k];
      const float3 to = make_float3(t.x, t.y, t.z) - p;
      const float  d2 = dot3(to, to);
      if (d2 <= t.w * t.w)
         sum += 0.07957747154f; // inside the bounding sphere: uniform over all directions
      else
      {
         const float cos_max = sqrtf(fmaxf(0.0f, 1.0f - t.w * t.w / d2));
         if (dot3(w, to) >= cos_max * sqrtf(d2))
            sum += 1.0f / (2.0f * M_PIf * fmaxf(1.0f - cos_max, 1e-7f));
      }
   }
   return sum / (float)params.num_caustic_targets;
}

/// Add the light path vertex x (diffuse, facing n, albedo a, throughput beta) to the pixel it
/// projects to, when the camera sees it directly
__device__ void caustic_splat(const float3 &x, const float3 &n, const float3 &albedo, const float3 &beta)
{
   float3       to_cam = params.camera_center - x;
   const float  dist   = length3(to_cam);
   if (dist < 1e-4f)
      return;
   to_cam = to_cam / dist;
   const float cos_x = dot3(n, to_cam); // the surface must face the camera on the lit side
   if (cos_x <= 0.0f)
      return;

   // Projection onto the pixel lattice: the plane through pixel00 facing the camera
   const float3 fwd    = -params.cam_w;
   const float  plane  = dot3(params.pixel00_loc - params.camera_center, fwd);
   const float  cos_c  = -dot3(to_cam, fwd);
   if (plane <= 0.0f || cos_c <= 1e-6f)
      return;
   const float3 q    = params.camera_center - to_cam * (plane / cos_c) - params.pixel00_loc;
   const float  du2  = dot3(params.pixel_delta_u, params.pixel_delta_u);
   const float  dv2  = dot3(params.pixel_delta_v, params.pixel_delta_v);
   const int    px   = (int)floorf(dot3(q, params.pixel_delta_u) / du2 + 0.5f);
   const int    py   = (int)floorf(dot3(q, params.pixel_delta_v) / dv2 + 0.5f);
   if (px < 0 || py < 0 || px >= (int)params.width || py >= (int)params.height)
      return;
   const unsigned int pixel_idx = (unsigned int)py * params.width + (unsigned int)px;
   // Same skip rule as raygen: a converged pixel takes no samples this launch, so no splats either
   if (params.pixel_sample_counts && params.pixel_sample_counts[pixel_idx] < 0 &&
       !(params.conv_streak && params.conv_streak[pixel_idx] == ADAPTIVE_RECHECK_MARK))
      return;
   // Leave from just above the surface, as NEE shadow rays do: a hit point that rounds below the
   // surface (the 950-unit ground sphere of the default scene) would otherwise see the surface
   // itself and drop the splat, in a moire of rounding cells
   // Thin films on the way let 1 - R through, as the camera paths see this surface through them
   unsigned int  film_seed = 0u; // unused: the join meets no medium
   const float3  T_cam     = shadow_transmittance_optix(x + 0.0001f * n, to_cam, dist - 0.001f, film_seed, true);
   if (fmaxf(T_cam.x, fmaxf(T_cam.y, T_cam.z)) <= 0.0f)
      return;

   // Pixel importance of the pinhole camera, per solid angle: plane^2 / (pixel area cos^3);
   // the geometry term to the camera point is cos_x / dist^2
   const float pixel_area = sqrtf(du2 * dv2);
   const float importance = plane * plane / (pixel_area * cos_c * cos_c * cos_c);
   const float scale = importance * cos_x / (dist * dist) * (1.0f / M_PIf) *
                       ((float)params.samples_per_launch / (float)params.caustic_paths);
   const float3 c = beta * albedo * T_cam * scale;
   float *acc = reinterpret_cast<float *>(params.accum_buffer + pixel_idx);
   atomicAdd(acc + 0, c.x);
   atomicAdd(acc + 1, c.y);
   atomicAdd(acc + 2, c.z);
}

extern "C" __global__ void __raygen__caustics()
{
   const unsigned int path = optixGetLaunchIndex().x;
   if (path >= params.caustic_paths || params.num_lights == 0)
      return;
   // Each sample of the camera pass comes with a fixed block of light paths, numbered globally:
   // the stream of a light path never depends on how the samples were split into launches
   const unsigned long long per_sample = params.caustic_paths / (unsigned int)params.samples_per_launch;
   const unsigned long long global     = (unsigned long long)params.total_samples_so_far * per_sample + path;
   unsigned int seed = pcg_hash((uint32_t)global ^ pcg_hash((uint32_t)(global >> 32) ^ 0x5bd1e995u ^
                                                            (params.sobol_sequence_seed * 0x9e3779b9u)));

   // A point on a light, chosen by power like NEE
   const float u_sel = rand_float(seed);
   int li = params.num_lights - 1;
   for (int i = 0; i < params.num_lights; ++i)
      if (u_sel < params.light_cdfs[i + 1]) { li = i; break; }
   const float p_sel = params.light_cdfs[li + 1] - params.light_cdfs[li];
   if (p_sel <= 0.0f)
      return;
   const OptixLightData &light = params.lights[li];
   float3 y, n_y;
   float  pdf_area;
   if (light.geom_type == OptixLightGeomType::RECTANGLE)
   {
      y        = light.corner + rand_float(seed) * light.u_vec + rand_float(seed) * light.v_vec;
      n_y      = light.normal;
      pdf_area = 1.0f / fmaxf(light.area, 1e-8f);
   }
   else
   {
      n_y      = rand_unit_vector(seed);
      y        = light.center + light.radius * n_y;
      pdf_area = 1.0f / fmaxf(4.0f * M_PIf * light.radius * light.radius, 1e-8f);
   }

   // A direction towards one of the delta targets (pdf: the mixture over all of them)
   if (params.num_caustic_targets == 0)
      return;
   float3       w;
   const int    k  = min((int)(rand_float(seed) * (float)params.num_caustic_targets), params.num_caustic_targets - 1);
   const float4 t  = params.caustic_targets[k];
   const float3 to = make_float3(t.x, t.y, t.z) - y;
   const float  d2 = dot3(to, to);
   if (d2 <= t.w * t.w)
      w = rand_unit_vector(seed);
   else
   {
      const float  cos_max = sqrtf(fmaxf(0.0f, 1.0f - t.w * t.w / d2));
      const float  cos_t   = 1.0f - rand_float(seed) * (1.0f - cos_max);
      const float  sin_t   = sqrtf(fmaxf(0.0f, 1.0f - cos_t * cos_t));
      const float  phi     = 2.0f * M_PIf * rand_float(seed);
      const float3 a       = to * rsqrtf(d2);
      float3 b1, b2;
      build_onb(a, b1, b2);
      w = normalize3(sin_t * cosf(phi) * b1 + sin_t * sinf(phi) * b2 + cos_t * a);
   }
   const float pdf_w = caustic_direction_pdf(y, w);
   float cos_y = dot3(w, n_y);
   if (light.geom_type == OptixLightGeomType::RECTANGLE)
      cos_y = fabsf(cos_y); // the panel emits on both faces
   if (cos_y <= 0.0f || pdf_w <= 0.0f)
      return; // into the sphere light itself: nothing leaves that way

   float3 beta = light.emission * params.light_intensity * (cos_y / (p_sel * pdf_area * pdf_w));
   float3 origin = y, dir = w;
   int    n_delta = 0; // delta vertices passed
   int    n_bent  = 0; // of which change the direction (not a straight pass through a thin film)
   bool   fall_pending = true; // light falloff not applied yet
   float  fall_dist    = 0.0f; // distance from the light so far, through thin films

   // The camera side counts the path when the emitter is reached at bounce n_delta + 1 < max_depth
   for (int v = 0; v <= params.max_depth - 2; ++v)
   {
      PRDRadiance prd;
      prd.hit = false;
      trace(params.traversable, origin, dir, 0.001f, 1e16f, &prd);
      if (!prd.hit || prd.material_idx < 0)
         return;
      // Falloff on the segment next to the light, as the camera side measures it: from the light to
      // the first vertex that is not a straight pass through a thin film
      if (fall_pending)
         fall_dist += length3(prd.hit_point - origin);
      const OptixMaterialData &mat = params.materials[prd.material_idx];

      const bool diffuse = mat.type == OptixMaterialType::LAMBERTIAN || mat.type == OptixMaterialType::SDF_MATERIAL;
      // Light reaching it straight through thin films only is NEE's (shadow rays cross films)
      if (diffuse && prd.visible && n_bent > 0)
      {
         if (fall_pending)
            beta = beta * light_falloff_factor(fall_dist);
         caustic_splat(prd.hit_point, prd.hit_normal, prd.hit_color, beta);
         return;
      }
      if (!is_caustic_delta_optix(mat))
         return; // anything else (diffuse first, glossy, clear coat, medium...) is the camera paths' job

      float3 scatter_dir, attenuation;
      if (!scatter_specular_optix(mat, prd, dir, [&]() { return rand_float(seed); }, scatter_dir, attenuation))
         return;
      beta = beta * attenuation;
      ++n_delta;
      if (!is_film_pass_optix(mat, prd, scatter_dir))
      {
         ++n_bent;
         if (fall_pending)
         {
            beta         = beta * light_falloff_factor(fall_dist);
            fall_pending = false;
         }
      }
      origin = prd.hit_point;
      dir    = scatter_dir;
      if (n_delta > 1)
      {
         const float surv = fminf(fmaxf(beta.x, fmaxf(beta.y, beta.z)), 0.95f);
         if (rand_float(seed) > surv)
            return;
         beta = beta / surv;
      }
   }
}

//==============================================================================
// MISS — sky background
//==============================================================================

extern "C" __global__ void __miss__ms()
{
   PRDRadiance *prd = getPRD();
   prd->hit = false;
}

// Shadow rays carry a single "occluded" flag in payload 0; reaching miss clears it.
extern "C" __global__ void __miss__shadow() { optixSetPayload_0(0u); }

//==============================================================================
// GOLF BALL DISPLACEMENT HELPERS (ported from shader_golf.cu)
//==============================================================================

__device__ __forceinline__ float distanceToNearestDimple_optix(float3 p)
{
   // Dimple directions are precomputed on the host (params.golf_points): the loop is
   // N dot products instead of N sin/cos/acos evaluations, and it runs 4x per hit.
   float3 q = normalize3(p);
   const int N = params.golf_dimple_count;
   const float3 *__restrict__ pts = params.golf_points;
   float max_dot = -1.0f;
   for (int i = 0; i < N; ++i)
   {
      float d = dot3(q, pts[i]);
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
// CLOSEST HIT — fill PRD with geometry info + resolved albedo
//==============================================================================

// Azimuthal tangent matching the CUDA renderer: cross(up, N). Used for spheres and
// triangles (an edge-based triangle tangent differs between triangles of the same
// fan-triangulated polygon and splits flat faces visibly).
__device__ __forceinline__ float3 azimuthal_tangent(const float3 &N)
{
   float3 T = cross3(make_float3(0.0f, 1.0f, 0.0f), N);
   if (dot3(T, T) < 1e-6f)
      T = make_float3(1.0f, 0.0f, 0.0f); // degenerate at poles
   return normalize3(T);
}

// Resolve the material index and the albedo (texture / procedural pattern applied).
// Everything else about the material is read by raygen from params.materials.
__device__ __forceinline__ void resolve_material(PRDRadiance *prd, int mat_idx, const float2 &uv,
                                                 const float3 &pattern_center)
{
   if (mat_idx < 0 || mat_idx >= params.num_materials)
   {
      prd->material_idx = -1;
      prd->hit_color    = make_float3(1.0f, 0.0f, 1.0f);
      return;
   }
   prd->material_idx = mat_idx;

   const OptixMaterialData &mat = params.materials[mat_idx];
   prd->hit_color = mat.albedo;

   // Texture sampling: override diffuse color with texel if texture is bound
   if (mat.texture_id >= 0 && mat.texture_id < params.num_textures && params.d_textures)
   {
      float4 texel = tex2D<float4>(params.d_textures[mat.texture_id], uv.x, 1.0f - uv.y);
      prd->hit_color = make_float3(texel.x, texel.y, texel.z);
   }

   // Apply procedural pattern if present
   if (mat.pattern == 1) // FIBONACCI_DOTS
   {
      float3 dir     = normalize3(prd->hit_point - pattern_center);
      int    N       = (int)mat.pattern_param1;
      float  dot_rad = mat.pattern_param2;
      float  max_dp  = -1.0f;
      const float3 *__restrict__ pts = params.pattern_points + mat.pattern_table_offset; // precomputed on host
      for (int i = 0; i < N; ++i)
      {
         float  d = dot3(dir, pts[i]);
         if (d > max_dp)
            max_dp = d;
      }
      max_dp = fmaxf(fminf(max_dp, 1.0f), -1.0f);
      float ang  = acosf(max_dp);
      float mask = ang < dot_rad ? 0.0f : 1.0f;
      prd->hit_color = mask * mat.albedo + (1.0f - mask) * mat.pattern_color;
   }
}

// Hardware-intersected triangles: no intersection program, attributes are the barycentrics.
extern "C" __global__ void __closesthit__triangle()
{
   PRDRadiance *prd = getPRD();
   prd->hit          = true;
   prd->hit_prim_idx = -1; // triangles are never NEE lights

   const unsigned int tri = optixGetPrimitiveIndex();
   const float2 bary      = optixGetTriangleBarycentrics(); // (u, v) weights of v1, v2
   const float  w         = 1.0f - bary.x - bary.y;
   const OptixTriangleData &td = params.tri_data[tri];
   prd->visible = td.visible != 0;

   // Inside / outside is decided by the face normal: an interpolated normal can disagree with it
   // in sign near silhouettes, which flipped the glass IOR ratio and the volume enter/exit test.
   const float3 *v = params.tri_vertices + 3u * tri;
   float3 face_normal = normalize3(cross3(v[1] - v[0], v[2] - v[0]));
   float3 outward_normal = face_normal;
   if (td.has_normals)
   {
      // Orient the face like the vertex normals (meshes whose winding disagrees with their normals)
      if (dot3(face_normal, td.n0 + td.n1 + td.n2) < 0.0f)
         face_normal = -face_normal;
      const float3 n = normalize3(w * td.n0 + bary.x * td.n1 + bary.y * td.n2);
      outward_normal = (dot3(n, n) > 0.0f) ? n : face_normal; // degenerate vertex normals: flat
   }

   float2 uv = make_float2(0.0f, 0.0f);
   if (td.has_uvs)
      uv = make_float2(w * td.uv0.x + bary.x * td.uv1.x + bary.y * td.uv2.x,
                       w * td.uv0.y + bary.x * td.uv1.y + bary.y * td.uv2.y);

   const float3 ray_dir = optixGetWorldRayDirection();
   prd->front_face = dot3(ray_dir, face_normal) < 0.0f;
   prd->hit_normal = prd->front_face ? outward_normal : -outward_normal;
   // The shading normal must face the ray like the face does; past a silhouette, use the face
   if (dot3(prd->hit_normal, ray_dir) >= 0.0f)
      prd->hit_normal = prd->front_face ? face_normal : -face_normal;
   prd->hit_point  = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;
   prd->hit_tangent = azimuthal_tangent(prd->hit_normal);

   resolve_material(prd, td.material_idx, uv, make_float3(0.0f, 0.0f, 0.0f));
}

// Custom primitives: spheres, displaced spheres, rectangles.
extern "C" __global__ void __closesthit__ch()
{
   PRDRadiance *prd = getPRD();
   prd->hit = true;
   prd->hit_prim_idx = optixGetPrimitiveIndex();

   const HitGroupData *sbt_data = reinterpret_cast<const HitGroupData *>(optixGetSbtDataPointer());
   prd->visible = sbt_data->visible != 0;

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

   // Surface tangent for anisotropic materials (geometry-specific)
   prd->hit_tangent = (sbt_data->geom_type == OptixGeomType::RECTANGLE) ? normalize3(sbt_data->u_vec)
                                                                         : azimuthal_tangent(prd->hit_normal);

   // Custom primitives carry no UVs; the Fibonacci-dot pattern is centred on the sphere.
   resolve_material(prd, sbt_data->material_idx, make_float2(0.0f, 0.0f), sbt_data->center);
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
// INTERSECTION — ray-marched signed distance primitive
//==============================================================================

extern "C" __global__ void __intersection__sdf()
{
   const HitGroupData *sbt_data = reinterpret_cast<const HitGroupData *>(optixGetSbtDataPointer());
   SdfPrim s;
   s.shape = sbt_data->sdf_shape;
   s.cx = sbt_data->center.x; s.cy = sbt_data->center.y; s.cz = sbt_data->center.z;
   s.p0 = sbt_data->u_vec.x;  s.p1 = sbt_data->u_vec.y;  s.p2 = sbt_data->u_vec.z;
   s.rx = sbt_data->v_vec.x;  s.ry = sbt_data->v_vec.y;  s.rz = sbt_data->v_vec.z;

   // Rays are traced with unit directions (no instance transforms), so t is a distance
   const float3 o   = optixGetObjectRayOrigin();
   const float3 d   = optixGetObjectRayDirection();
   const float  len = length3(d);
   if (len <= 0.0f)
      return;
   const float3 dir = d / len;
   const float tmin = optixGetRayTmin() * len, tmax = optixGetRayTmax() * len;
   float t, nx, ny, nz;
   if (!sdf_march(s, o.x, o.y, o.z, dir.x, dir.y, dir.z, tmin, tmax, t, nx, ny, nz))
      return;
   optixReportIntersection(t / len, 2, // hit kind 2 = SDF
                           __float_as_int(nx), __float_as_int(ny), __float_as_int(nz));
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
