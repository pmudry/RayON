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

namespace CudaScene
{

//==============================================================================
// MATERIAL TYPES
//==============================================================================

enum class MaterialType : uint8_t
{
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
enum class ProceduralPattern : uint8_t
{
   NONE,           // No pattern, use solid color
   FIBONACCI_DOTS, // Regularly spaced dots using Fibonacci grid
   CHECKERBOARD,   // Checkerboard pattern
   STRIPES         // Striped pattern
};

/**
 * @brief GPU material structure - flat, no virtuals, device-friendly
 */
struct Material
{
   MaterialType type;
   f3 albedo;
   f3 emission;
   float roughness;
   float metallic;
   float refractive_index;
   float transmission;
   int texture_id;

   // Procedural pattern support
   ProceduralPattern pattern;
   f3 pattern_color;
   float pattern_param1;
   float pattern_param2;

   __host__ __device__ Material()
       : type(MaterialType::LAMBERTIAN), albedo(0, 0, 0), emission(0, 0, 0), roughness(0), metallic(0),
         refractive_index(1), transmission(0), texture_id(-1), pattern(ProceduralPattern::NONE), pattern_color(0, 0, 0),
         pattern_param1(0), pattern_param2(0)
   {
   }
};

//==============================================================================
// GEOMETRY TYPES
//==============================================================================

enum class GeometryType : uint8_t
{
   SPHERE,
   RECTANGLE,
   CUBE,
   DISPLACED_SPHERE,
   TRIANGLE,
   TRIANGLE_MESH,
   SDF_PRIMITIVE
};

enum class SDFType : uint8_t
{
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
struct Geometry
{
   GeometryType type;
   int material_id;

   union GeomData
   {
      struct
      {
         f3 center;
         float radius;
      } sphere;

      struct
      {
         f3 corner;
         f3 u, v;
      } rectangle;

      struct
      {
         f3 center;
         float radius;
         float displacement_scale;
         int pattern_type;
      } displaced_sphere;

      struct
      {
         f3 v0, v1, v2;
         f3 n0, n1, n2;
         bool has_normals;
      } triangle;

      struct
      {
         int bvh_root_idx;
         int mesh_id;
         f3 translation;
         f3 rotation;
         f3 scale;
      } mesh_instance;

      struct
      {
         SDFType sdf_type;
         f3 position;
         f3 parameters;
         float max_distance;
         float epsilon;
      } sdf;

      __host__ __device__ GeomData() {} // Empty constructor for union
   } data;

   f3 bounds_min, bounds_max;
};

//==============================================================================
// BVH STRUCTURES (for Phase 5)
//==============================================================================

/**
 * @brief Cache-line-aligned BVH node (64 bytes)
 *
 * Packed to exactly one 64-byte cache line so that each node fetch loads
 * all needed data in a single memory transaction. Layout:
 *   bytes  0-11: bounds_min (f3)
 *   bytes 12-23: bounds_max (f3)
 *   bytes 24-27: left_child / first_geom_idx
 *   bytes 28-31: right_child / geom_count
 *   byte  32:    is_leaf
 *   byte  33:    split_axis
 *   bytes 34-63: padding (reserved for future use)
 */
struct alignas(64) BVHNode
{
   f3 bounds_min, bounds_max; // 24 bytes

   union NodeData
   {
      struct
      {
         int left_child;
         int right_child;
      } interior;

      struct
      {
         int first_geom_idx;
         int geom_count;
      } leaf;

      __host__ __device__ NodeData() {} // Empty constructor for union
   } data; // 8 bytes

   bool is_leaf;       // 1 byte
   uint8_t split_axis; // 1 byte
   uint8_t _pad[30];   // Pad to 64 bytes
};

//==============================================================================
// COMPLETE GPU SCENE
//==============================================================================

/**
 * @brief Complete GPU scene - pointers to device memory
 * Aligned to 128 bytes for optimal memory access
 */
struct alignas(128) Scene
{
   // Device memory pointers (aligned to cache line boundaries)
   Material *materials;
   Geometry *geometries;
   BVHNode *bvh_nodes;

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
