/**
 * @file rough_mirror.cuh
 * @brief Rough mirror (microfacet) material with controllable roughness
 * 
 * Implements reflection with surface roughness using perturbed normals.
 * This approximates microfacet BRDF behavior for metallic surfaces.
 */

#pragma once
#include "../material_base.cuh"
#include "cuda_utils.cuh"

// Forward declarations from shader_common.cuh
__device__ f3 randOnUnitSphere(curandState* state);
extern __constant__ float g_metal_fuzziness;

namespace Materials {

/**
 * @brief Rough mirror material with microfacet-like behavior
 * 
 * Reflects rays with random perturbation proportional to surface roughness.
 * Models metallic surfaces with varying degrees of polish.
 */
struct RoughMirror : public MaterialBase<RoughMirror> {
    RoughMirrorParams params;
    
    /**
     * @brief Construct rough mirror material
     * @param p Material parameters (albedo, roughness)
     */
    __device__ __forceinline__ RoughMirror(const RoughMirrorParams& p) : params(p) {}
    
    /**
     * @brief Scatter incident ray with roughness-perturbed reflection
     * 
     * Uses normal perturbation to simulate microfacet behavior:
     * - roughness = 0: perfect mirror
     * - roughness = 1: highly scattered reflection
     * 
     * @param r_in Incident ray
     * @param rec Hit record containing surface information
     * @param attenuation Output: color attenuation
     * @param scattered Output: scattered ray direction
     * @param state Random number generator state
     * @return true if ray reflects above surface, false if absorbed
     */
    __device__ bool scatter(const ray_simple& r_in,
                           const hit_record_simple& rec,
                           f3& attenuation,
                           ray_simple& scattered,
                           curandState* state) const {
        // Perturb normal based on roughness (microfacet approximation)
        f3 perturbed_normal = unit_vector(
            rec.normal + params.roughness * g_metal_fuzziness * randOnUnitSphere(state)
        );
        
        // Reflect off perturbed normal
        f3 reflected = do_reflect(unit_vector(r_in.dir), perturbed_normal);
        
        scattered = ray_simple(rec.p, reflected);
        attenuation = params.albedo;
        
        // Absorb rays that scatter below the surface
        // This happens more frequently at high roughness values
        return dot(scattered.dir, rec.normal) > 0.0f;
    }
    
    /**
     * @brief Get emitted light (zero for non-emissive surfaces)
     * @return Black (no emission)
     */
    __device__ __forceinline__ f3 emission() const {
        return f3(0.0f, 0.0f, 0.0f);
    }
};

} // namespace Materials
