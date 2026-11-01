#pragma once
/**
 * @file spectral.cuh
 * @brief Chromatic dispersion helpers shared by the CUDA and OptiX kernels.
 *
 * A `glass` with `dispersion` > 0 has a refractive index that depends on the wavelength (Cauchy's
 * equation), so white light splits into a rainbow. The renderer is an RGB path tracer: a path that
 * meets such a glass draws one wavelength ("hero wavelength"), keeps it for every later dispersive
 * refraction, and its throughput is weighted by the colour of that wavelength. The weights average
 * to (1, 1, 1) over the sampled range, so a white light stays white in expectation; the rainbow
 * builds up as colour noise that converges like any other.
 *
 * Plain floats only, so both back ends can use it with their own vector types.
 */

#include <cuda_runtime.h>

/// Sampled wavelength range, micrometres (the visible spectrum)
constexpr float SPECTRAL_LAMBDA_MIN = 0.38f;
constexpr float SPECTRAL_LAMBDA_MAX = 0.72f;
/// Reference wavelength of `refractive_index` (the sodium d line, 587.6 nm)
constexpr float SPECTRAL_LAMBDA_D = 0.5876f;

/// Uniform wavelength for a uniform variate u in [0, 1)
__device__ __forceinline__ float spectral_sample_lambda(float u)
{
   return SPECTRAL_LAMBDA_MIN + u * (SPECTRAL_LAMBDA_MAX - SPECTRAL_LAMBDA_MIN);
}

/// Cauchy's equation: n(lambda) = n_d + B (1 / lambda^2 - 1 / lambda_d^2), lambda in micrometres.
/// `dispersion` is B in um^2 (crown glass ~0.0045, flint ~0.013; 0.02 and above is showy).
__device__ __forceinline__ float dispersive_ior(float ior_d, float dispersion, float lambda)
{
   return ior_d + dispersion * (1.0f / (lambda * lambda) - 1.0f / (SPECTRAL_LAMBDA_D * SPECTRAL_LAMBDA_D));
}

/// Throughput weight of a wavelength: three Gaussian bumps standing for the display primaries
/// (blue 455 nm, green 540 nm, red 605 nm), each scaled so that its mean over the sampled range is 1
__device__ __forceinline__ void spectral_weight(float lambda, float &r, float &g, float &b)
{
   const float width = SPECTRAL_LAMBDA_MAX - SPECTRAL_LAMBDA_MIN;
   auto bump = [&](float mu, float sigma) {
      const float d = (lambda - mu) / sigma;
      return __expf(-0.5f * d * d) * width / (sigma * 2.50662827f); // sigma * sqrt(2 pi)
   };
   r = bump(0.605f, 0.040f);
   g = bump(0.540f, 0.035f);
   b = bump(0.455f, 0.035f);
}
