/**
 * @file cuda_scene.cuh
 * @brief GPU-optimized scene representation for CUDA rendering
 * 
 * This file defines GPU-friendly structures that mirror the host-side
 * SceneDescription but are optimized for device memory layout and access patterns.
 */

#pragma once

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#include "cuda_float3.cuh"
#include <cstdint>

namespace CudaScene {

//==============================================================================
// MATERIAL TYPES
//==============================================================================

enum class MaterialType : uint8_t {
    LAMBERTIAN,
    METAL,
    MIRROR,
    ROUGH_MIRROR,
    GLASS,
    DIELECTRIC,
    LIGHT,
    CONSTANT,
    SHOW_NORMALS,
    SDF_MATERIAL
};

/**
 * @brief Procedural pattern types for material coloring
 */
enum class ProceduralPattern : uint8_t {
    NONE,                    // No pattern, use solid color
    FIBONACCI_DOTS,          // Regularly spaced dots using Fibonacci grid
    CHECKERBOARD,           // Checkerboard pattern
    STRIPES                 // Striped pattern
};

/**
 * @brief GPU material structure - flat, no virtuals, device-friendly
 */
struct Material {
    MaterialType type;
    float3_simple albedo;
    float3_simple emission;
    float roughness;
    float metallic;
    float refractive_index;
    float transmission;
    int texture_id;
    
    // Procedural pattern support
    ProceduralPattern pattern;
    float3_simple pattern_color;
    float pattern_param1;
    float pattern_param2;
    
    __host__ __device__ Material() : type(MaterialType::LAMBERTIAN), 
        albedo(0,0,0), emission(0,0,0),
        roughness(0), metallic(0), refractive_index(1), transmission(0), texture_id(-1),
        pattern(ProceduralPattern::NONE), pattern_color(0,0,0), pattern_param1(0), pattern_param2(0) {}
};

//==============================================================================
// GEOMETRY TYPES
//==============================================================================

enum class GeometryType : uint8_t {
    SPHERE,
    RECTANGLE,
    CUBE,
    DISPLACED_SPHERE,
    TRIANGLE,
    TRIANGLE_MESH,
    SDF_PRIMITIVE
};

enum class SDFType : uint8_t {
    SPHERE,
    BOX,
    TORUS,
    CAPSULE,
    MANDELBULB,
    CUSTOM
};

/**
 * @brief GPU geometry structure - flat union for memory efficiency
 */
struct Geometry {
    GeometryType type;
    int material_id;
    
    union GeomData {
        struct {
            float3_simple center;
            float radius;
        } sphere;
        
        struct {
            float3_simple corner;
            float3_simple u, v;
        } rectangle;
        
        struct {
            float3_simple center;
            float radius;
            float displacement_scale;
            int pattern_type;
        } displaced_sphere;
        
        struct {
            float3_simple v0, v1, v2;
            float3_simple n0, n1, n2;
            bool has_normals;
        } triangle;
        
        struct {
            int bvh_root_idx;
            int mesh_id;
            float3_simple translation;
            float3_simple rotation;
            float3_simple scale;
        } mesh_instance;
        
        struct {
            SDFType sdf_type;
            float3_simple position;
            float3_simple parameters;
            float max_distance;
            float epsilon;
        } sdf;
        
        __host__ __device__ GeomData() {} // Empty constructor for union
    } data;
    
    float3_simple bounds_min, bounds_max;
};

//==============================================================================
// BVH STRUCTURES (for Phase 5)
//==============================================================================

struct BVHNode {
    float3_simple bounds_min, bounds_max;
    
    union NodeData {
        struct {
            int left_child;
            int right_child;
        } interior;
        
        struct {
            int first_geom_idx;
            int geom_count;
        } leaf;
        
        __host__ __device__ NodeData() {} // Empty constructor for union
    } data;
    
    bool is_leaf;
    uint8_t split_axis;
};

//==============================================================================
// COMPLETE GPU SCENE
//==============================================================================

/**
 * @brief Complete GPU scene - pointers to device memory
 * Aligned to 128 bytes for optimal memory access
 */
struct alignas(128) Scene {
    // Device memory pointers (aligned to cache line boundaries)
    Material* materials;
    Geometry* geometries;
    BVHNode* bvh_nodes;
    
    // Counts
    int num_materials;
    int num_geometries;
    int num_bvh_nodes;
    
    // Scene-level BVH
    int bvh_root_idx;
    bool use_bvh;
    
    // Ray marching settings
    int max_ray_march_steps;
    float ray_march_epsilon;
};

} // namespace CudaScene
