/**
 * @file material_base.cuh
 * @brief Base material template using CRTP (Curiously Recurring Template Pattern)
 * 
 * This header provides zero-overhead struct-based material inheritance using CRTP.
 * Benefits:
 * - No virtual function overhead (static polymorphism)
 * - Compile-time dispatch via templates
 * - Shared utility functions for all materials
 * - Fully inlinable by NVCC optimizer
 */

#pragma once
#include "cuda_float3.cuh"
#include <curand_kernel.h>

namespace Materials {

//==============================================================================
// OPTICAL PHYSICS UTILITY FUNCTIONS (shared by all materials)
//==============================================================================

/**
 * @brief Reflect vector v around normal n
 * @param v Incident vector (pointing towards surface)
 * @param n Surface normal (unit vector)
 * @return Reflected vector
 */
__device__ __forceinline__ f3 reflect(const f3& v, const f3& n) {
    return v - 2.0f * dot(v, n) * n;
}

/**
 * @brief Refract vector through interface using Snell's law
 * @param uv Unit incident vector
 * @param n Surface normal
 * @param etai_over_etat Ratio of refractive indices (incident/transmitted)
 * @return Refracted vector
 */
__device__ __forceinline__ f3 refract(const f3& uv, const f3& n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1.0f);
    f3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    f3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

/**
 * @brief Schlick's approximation for Fresnel reflectance
 * @param cosine Cosine of incident angle
 * @param ref_idx Ratio of refractive indices
 * @return Reflectance probability [0, 1]
 */
__device__ __forceinline__ float reflectance(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

/**
 * @brief Reflect with roughness-based perturbation (microfacet model)
 * @param v Incident vector
 * @param n Surface normal
 * @param roughness Surface roughness [0, 1]
 * @param state Random number generator state
 * @return Reflected vector with random perturbation
 */
__device__ __forceinline__ f3 reflect_fuzzy(const f3& v, const f3& n, float roughness, curandState* state);

//==============================================================================
// MATERIAL PARAMETER STRUCTS (POD types for GPU transfer)
//==============================================================================

/**
 * @brief Lambertian (diffuse) material parameters
 */
struct LambertianParams {
    f3 albedo;  // Base color / reflectance
};

/**
 * @brief Perfect mirror material parameters
 */
struct MirrorParams {
    f3 albedo;  // Tint color
};

/**
 * @brief Rough mirror (microfacet) material parameters
 */
struct RoughMirrorParams {
    f3 albedo;     // Base color
    float roughness;  // Surface roughness [0, 1]
};

/**
 * @brief Glass/dielectric material parameters
 */
struct GlassParams {
    float refractive_index;  // Index of refraction (e.g., 1.5 for glass)
};

/**
 * @brief Emissive light material parameters
 */
struct LightParams {
    f3 emission;  // Emitted radiance
};

/**
 * @brief Constant (non-scattering) material parameters
 */
struct ConstantParams {
    f3 color;  // Fixed output color
};

/**
 * @brief Show normals visualization material parameters
 */
struct ShowNormalsParams {
    f3 normal;  // Surface normal from hit record
};

//==============================================================================
// CRTP BASE TEMPLATE
//==============================================================================

/**
 * @brief Base material template using CRTP pattern
 * 
 * Provides shared interface and utilities for all materials.
 * Derived materials must implement:
 * - bool scatter(ray_simple, hit_record_simple, f3&, ray_simple&, curandState*)
 * - f3 emission()
 * 
 * @tparam Derived The derived material type (CRTP pattern)
 */
template<typename Derived>
struct MaterialBase {
    /**
     * @brief Get reference to derived material (CRTP downcast)
     */
    __device__ __forceinline__ const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }
    
    __device__ __forceinline__ Derived& derived() {
        return static_cast<Derived&>(*this);
    }
    
    // Shared utilities accessible to all derived materials
    // (These delegate to the free functions above for better optimization)
    
    __device__ __forceinline__ f3 do_reflect(const f3& v, const f3& n) const {
        return reflect(v, n);
    }
    
    __device__ __forceinline__ f3 do_refract(const f3& uv, const f3& n, float eta) const {
        return refract(uv, n, eta);
    }
    
    __device__ __forceinline__ float do_reflectance(float cosine, float ref_idx) const {
        return reflectance(cosine, ref_idx);
    }
    
    // Interface contract documentation (not enforced at compile-time yet):
    //
    // __device__ bool scatter(const ray_simple& r_in, 
    //                        const hit_record_simple& rec,
    //                        f3& attenuation, 
    //                        ray_simple& scattered,
    //                        curandState* state) const;
    //
    // __device__ f3 emission() const;
};

} // namespace Materials
