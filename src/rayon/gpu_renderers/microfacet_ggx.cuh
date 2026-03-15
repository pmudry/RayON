/**
 * @file microfacet_ggx.cuh
 * @brief Anisotropic GGX (Trowbridge-Reitz) microfacet distribution
 *
 * Implements the physically-based microfacet model from PBR Book 4th Ed §9.6:
 * - Smith masking/shadowing (G1, G) — algebraic form, no trig
 * - VNDF sampling (Sample_wm) — uses __sincosf intrinsic
 * - Complex Fresnel for conductors (FrComplex) — fully unrolled, no arrays
 */

#pragma once
#include "cuda_float3.cuh"
#include "cuda_utils.cuh"
#include <math_constants.h>

//==============================================================================
// SMITH MASKING — Algebraic form (no trig functions)
// Lambda(w) = (-1 + sqrt(1 + ((alpha_x*w.x)^2 + (alpha_y*w.y)^2) / w.z^2)) / 2
//==============================================================================

__device__ __forceinline__ float Lambda_GGX(const f3 &w, float alpha_x, float alpha_y)
{
   float wz2 = w.z * w.z;
   if (wz2 < 1e-16f)
      return 0.0f;
   float a2 = (alpha_x * w.x) * (alpha_x * w.x) + (alpha_y * w.y) * (alpha_y * w.y);
   return (sqrtf(1.0f + a2 / wz2) - 1.0f) * 0.5f;
}

__device__ __forceinline__ float G1_GGX(const f3 &w, float alpha_x, float alpha_y)
{
   return 1.0f / (1.0f + Lambda_GGX(w, alpha_x, alpha_y));
}

/** Height-correlated masking-shadowing — PBR Book Eq. 9.22 */
__device__ __forceinline__ float G_GGX(const f3 &wo, const f3 &wi, float alpha_x, float alpha_y)
{
   return 1.0f / (1.0f + Lambda_GGX(wo, alpha_x, alpha_y) + Lambda_GGX(wi, alpha_x, alpha_y));
}

//==============================================================================
// VNDF SAMPLING — PBR Book §9.6.4 (uses __sincosf intrinsic)
//==============================================================================

__device__ __forceinline__ f3 Sample_wm_GGX(const f3 &wo, float alpha_x, float alpha_y,
                                             float u1, float u2)
{
   // Step 1: Transform wo to hemispherical configuration
   f3 wh = normalize(f3(alpha_x * wo.x, alpha_y * wo.y, wo.z));
   if (wh.z < 0.0f)
      wh = -wh;

   // Step 2: Find orthonormal basis
   f3 T1 = (wh.z < 0.99999f)
               ? normalize(cross(f3(0.0f, 0.0f, 1.0f), wh))
               : f3(1.0f, 0.0f, 0.0f);
   f3 T2 = cross(wh, T1);

   // Step 3: Sample uniform disk — __sincosf computes both in one instruction
   float r = sqrtf(u1);
   float phi = 2.0f * CUDART_PI_F * u2;
   float sin_phi, cos_phi;
   __sincosf(phi, &sin_phi, &cos_phi);
   float p_x = r * cos_phi;
   float p_y = r * sin_phi;

   // Step 4: Warp hemispherical projection
   float h = sqrtf(fmaxf(0.0f, 1.0f - p_x * p_x));
   float s = (1.0f + wh.z) * 0.5f;
   p_y = (1.0f - s) * h + s * p_y;

   // Step 5: Reproject and transform back to ellipsoid
   float pz = sqrtf(fmaxf(0.0f, 1.0f - p_x * p_x - p_y * p_y));
   f3 nh = p_x * T1 + p_y * T2 + pz * wh;
   return normalize(f3(alpha_x * nh.x, alpha_y * nh.y, fmaxf(1e-6f, nh.z)));
}

//==============================================================================
// COMPLEX FRESNEL FOR CONDUCTORS — fully unrolled, register-friendly
//==============================================================================

/// Single-channel conductor Fresnel
__device__ __forceinline__ float FrComplex1(float cos2, float sin2, float cos_theta_i,
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

__device__ __forceinline__ f3 FrComplex(float cos_theta_i, const f3 &eta, const f3 &k)
{
   cos_theta_i = fminf(fmaxf(cos_theta_i, 0.0f), 1.0f);
   float cos2 = cos_theta_i * cos_theta_i;
   float sin2 = 1.0f - cos2;
   return f3(FrComplex1(cos2, sin2, cos_theta_i, eta.x, k.x),
             FrComplex1(cos2, sin2, cos_theta_i, eta.y, k.y),
             FrComplex1(cos2, sin2, cos_theta_i, eta.z, k.z));
}
