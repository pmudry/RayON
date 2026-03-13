/**
 * @file microfacet_ggx.cuh
 * @brief Anisotropic GGX (Trowbridge-Reitz) microfacet distribution
 *
 * Implements the physically-based microfacet model from PBR Book 4th Ed §9.6:
 * - Trowbridge-Reitz NDF (D)
 * - Smith masking/shadowing (G1, G)
 * - VNDF sampling (Sample_wm)
 * - Complex Fresnel for conductors (FrComplex)
 */

#pragma once
#include "cuda_float3.cuh"
#include "cuda_utils.cuh"
#include <math_constants.h>

//==============================================================================
// HELPER TRIG FUNCTIONS (local shading space: z = normal)
//==============================================================================

__device__ __forceinline__ float CosTheta(const f3 &w) { return w.z; }
__device__ __forceinline__ float Cos2Theta(const f3 &w) { return w.z * w.z; }
__device__ __forceinline__ float Sin2Theta(const f3 &w) { return fmaxf(0.0f, 1.0f - Cos2Theta(w)); }
__device__ __forceinline__ float SinTheta(const f3 &w) { return sqrtf(Sin2Theta(w)); }
__device__ __forceinline__ float TanTheta(const f3 &w) { return SinTheta(w) / CosTheta(w); }
__device__ __forceinline__ float Tan2Theta(const f3 &w) { return Sin2Theta(w) / Cos2Theta(w); }

__device__ __forceinline__ float CosPhi(const f3 &w)
{
   float sinTheta = SinTheta(w);
   return (sinTheta == 0.0f) ? 1.0f : fminf(fmaxf(w.x / sinTheta, -1.0f), 1.0f);
}

__device__ __forceinline__ float SinPhi(const f3 &w)
{
   float sinTheta = SinTheta(w);
   return (sinTheta == 0.0f) ? 0.0f : fminf(fmaxf(w.y / sinTheta, -1.0f), 1.0f);
}

//==============================================================================
// TROWBRIDGE-REITZ (GGX) DISTRIBUTION — PBR Book Eq. 9.16
//==============================================================================

/**
 * @brief Anisotropic Trowbridge-Reitz NDF
 * @param wm Microfacet normal in local shading space
 * @param alpha_x Roughness along tangent direction
 * @param alpha_y Roughness along bitangent direction
 */
__device__ __forceinline__ float D_GGX(const f3 &wm, float alpha_x, float alpha_y)
{
   float cos2Theta = Cos2Theta(wm);
   if (cos2Theta < 1e-16f)
      return 0.0f;

   float tan2Theta = Sin2Theta(wm) / cos2Theta;
   if (isinf(tan2Theta))
      return 0.0f;

   float cos4Theta = cos2Theta * cos2Theta;
   float cosPhi = CosPhi(wm);
   float sinPhi = SinPhi(wm);

   float e = tan2Theta * ((cosPhi * cosPhi) / (alpha_x * alpha_x) +
                           (sinPhi * sinPhi) / (alpha_y * alpha_y));

   float denom = CUDART_PI_F * alpha_x * alpha_y * cos4Theta * (1.0f + e) * (1.0f + e);
   return 1.0f / denom;
}

//==============================================================================
// SMITH MASKING — PBR Book Eq. 9.20-9.21
//==============================================================================

/**
 * @brief Lambda function for Smith masking (anisotropic)
 */
__device__ __forceinline__ float Lambda_GGX(const f3 &w, float alpha_x, float alpha_y)
{
   float cos2Theta = Cos2Theta(w);
   if (cos2Theta < 1e-16f)
      return 0.0f;

   float tan2Theta = Sin2Theta(w) / cos2Theta;
   if (isinf(tan2Theta))
      return 0.0f;

   float cosPhi = CosPhi(w);
   float sinPhi = SinPhi(w);
   float alpha2 = (cosPhi * alpha_x) * (cosPhi * alpha_x) +
                   (sinPhi * alpha_y) * (sinPhi * alpha_y);

   return (sqrtf(1.0f + alpha2 * tan2Theta) - 1.0f) * 0.5f;
}

/**
 * @brief Single-direction masking function G1
 */
__device__ __forceinline__ float G1_GGX(const f3 &w, float alpha_x, float alpha_y)
{
   return 1.0f / (1.0f + Lambda_GGX(w, alpha_x, alpha_y));
}

/**
 * @brief Height-correlated masking-shadowing — PBR Book Eq. 9.22
 */
__device__ __forceinline__ float G_GGX(const f3 &wo, const f3 &wi, float alpha_x, float alpha_y)
{
   return 1.0f / (1.0f + Lambda_GGX(wo, alpha_x, alpha_y) + Lambda_GGX(wi, alpha_x, alpha_y));
}

//==============================================================================
// VNDF SAMPLING — PBR Book §9.6.4
//==============================================================================

/**
 * @brief Sample visible microfacet normal (VNDF sampling)
 *
 * Follows the geometric approach from PBR Book:
 * 1. Transform wo to hemispherical configuration
 * 2. Build orthonormal basis
 * 3. Sample uniform disk
 * 4. Warp for visible normal projection
 * 5. Reproject and transform back
 *
 * @param wo Outgoing direction in local shading space (z = normal)
 * @param alpha_x, alpha_y Anisotropic roughness
 * @param u1, u2 Uniform random numbers in [0,1)
 * @return Sampled microfacet normal in local shading space
 */
__device__ __forceinline__ f3 Sample_wm_GGX(const f3 &wo, float alpha_x, float alpha_y,
                                             float u1, float u2)
{
   // Step 1: Transform wo to hemispherical configuration
   f3 wh = normalize(f3(alpha_x * wo.x, alpha_y * wo.y, wo.z));
   if (wh.z < 0.0f)
      wh = -wh;

   // Step 2: Find orthonormal basis for visible normal sampling
   f3 T1 = (wh.z < 0.99999f)
               ? normalize(cross(f3(0.0f, 0.0f, 1.0f), wh))
               : f3(1.0f, 0.0f, 0.0f);
   f3 T2 = cross(wh, T1);

   // Step 3: Generate uniformly distributed points on the unit disk
   float r = sqrtf(u1);
   float phi = 2.0f * CUDART_PI_F * u2;
   float p_x = r * cosf(phi);
   float p_y = r * sinf(phi);

   // Step 4: Warp hemispherical projection for visible normal sampling
   // PBR Book: Lerp((1+wh.z)/2, h, p_y) = (1-s)*h + s*p_y  where s=(1+wh.z)/2
   float h = sqrtf(fmaxf(0.0f, 1.0f - p_x * p_x));
   float s = (1.0f + wh.z) * 0.5f;
   p_y = (1.0f - s) * h + s * p_y;

   // Step 5: Reproject to hemisphere and transform normal to ellipsoid configuration
   float pz = sqrtf(fmaxf(0.0f, 1.0f - p_x * p_x - p_y * p_y));
   f3 nh = p_x * T1 + p_y * T2 + pz * wh;
   return normalize(f3(alpha_x * nh.x, alpha_y * nh.y, fmaxf(1e-6f, nh.z)));
}

//==============================================================================
// COMPLEX FRESNEL FOR CONDUCTORS
//==============================================================================

/**
 * @brief Per-channel complex Fresnel reflectance for conductors
 *
 * Exact conductor Fresnel equations (not Schlick approximation).
 * For metals, the complex IOR is n + ik where eta=n, k=extinction.
 *
 * @param cos_theta_i Cosine of incident angle (relative to microfacet normal)
 * @param eta Real part of complex IOR (per-channel)
 * @param k Imaginary/extinction part of complex IOR (per-channel)
 * @return Per-channel Fresnel reflectance
 */
__device__ __forceinline__ f3 FrComplex(float cos_theta_i, const f3 &eta, const f3 &k)
{
   // Clamp to avoid numerical issues
   cos_theta_i = fminf(fmaxf(cos_theta_i, 0.0f), 1.0f);
   float cos2 = cos_theta_i * cos_theta_i;
   float sin2 = 1.0f - cos2;

   // Per-channel computation using arrays for loop efficiency
   float eta_arr[3] = {eta.x, eta.y, eta.z};
   float k_arr[3] = {k.x, k.y, k.z};
   float result[3];

   for (int ch = 0; ch < 3; ch++)
   {
      float eta2 = eta_arr[ch] * eta_arr[ch];
      float k2 = k_arr[ch] * k_arr[ch];

      float t0 = eta2 - k2 - sin2;
      float a2plusb2 = sqrtf(fmaxf(t0 * t0 + 4.0f * eta2 * k2, 0.0f));
      float a = sqrtf(fmaxf((a2plusb2 + t0) * 0.5f, 0.0f));

      // Rs (s-polarized)
      float Rs_num = a2plusb2 + cos2 - 2.0f * a * cos_theta_i;
      float Rs_den = a2plusb2 + cos2 + 2.0f * a * cos_theta_i;
      float Rs = Rs_num / fmaxf(Rs_den, 1e-10f);

      // Rp (p-polarized)
      float Rp_num = a2plusb2 * cos2 + sin2 * sin2 - 2.0f * a * cos_theta_i * sin2;
      float Rp_den = a2plusb2 * cos2 + sin2 * sin2 + 2.0f * a * cos_theta_i * sin2;
      float Rp = Rs * Rp_num / fmaxf(Rp_den, 1e-10f);

      // Average of both polarizations
      result[ch] = (Rs + Rp) * 0.5f;
   }
   return f3(result[0], result[1], result[2]);
}
