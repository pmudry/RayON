/**
 * @file mis_utils.hpp
 * @brief Multiple Importance Sampling utilities shared by CPU and GPU renderers
 *
 * Provides the power heuristic and the LightSample struct used by the CPU
 * iterative path-tracing loop for Next Event Estimation (NEE).
 */

#pragma once

#include "color.hpp"
#include "vec3.hpp"

//==============================================================================
// POWER HEURISTIC (beta = 2)
// No powf() needed: squaring is sufficient and numerically stable.
//==============================================================================

/// Power heuristic for MIS: returns weight for estimator 'a' given PDF pdf_a
/// when combined with an estimator sampling from PDF pdf_b.
inline double power_heuristic(double pdf_a, double pdf_b)
{
   double a = pdf_a * pdf_a;
   double b = pdf_b * pdf_b;
   return (a + b > 0.0) ? (a / (a + b)) : 0.0;
}

//==============================================================================
// CPU LIGHT SAMPLE
//==============================================================================

/// Result of sampling a point on an area light for Next Event Estimation.
struct LightSample
{
   Vec3  direction; ///< Unit direction from shading point toward the sampled light point
   Color emission;  ///< Emitted radiance at the sampled light point
   double pdf;      ///< Solid-angle PDF of this sample (already accounting for light selection)
   double distance; ///< Distance from shading point to the sampled point (for shadow-ray t_max)
};
