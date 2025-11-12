/**
 * @file light.cuh
 * @brief Emissive light material
 * 
 * Implements area light sources that emit radiance.
 * Light materials don't scatter rays - they terminate the path.
 */

#pragma once
#include "../material_base.cuh"

// Forward declaration for light intensity override
extern __constant__ float g_light_intensity;

namespace Materials {

/**
 * @brief Emissive light material
 * 
 * Represents area light sources in the scene.
 * When a ray hits a light material, it contributes emission and terminates.
 */
struct Light : public MaterialBase<Light> {
    LightParams params;
    
    /**
     * @brief Construct light material
     * @param p Material parameters (emission color/intensity)
     */
    __device__ __forceinline__ Light(const LightParams& p) : params(p) {}
    
    /**
     * @brief Light materials don't scatter - they absorb rays
     * 
     * @param r_in Incident ray (unused)
     * @param rec Hit record (unused)
     * @param attenuation Output: not set (unused)
     * @param scattered Output: not set (unused)
     * @param state Random number generator state (unused)
     * @return false (light doesn't scatter, terminates path)
     */
    __device__ bool scatter(const ray_simple& r_in,
                           const hit_record_simple& rec,
                           f3& attenuation,
                           ray_simple& scattered,
                           curandState* state) const {
        // Light materials absorb rays and emit light
        // The emission is handled by emission() method
        return false;
    }
    
    /**
     * @brief Get emitted light radiance
     * @return Emitted color scaled by global light intensity
     */
    __device__ __forceinline__ f3 emission() const {
        return params.emission * g_light_intensity;
    }
};

} // namespace Materials
