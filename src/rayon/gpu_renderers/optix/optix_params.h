// Shared data structures for OptiX host and device code
#pragma once

#include <cuda_runtime.h>

#ifdef __CUDACC__
#include <optix.h>
#endif

// Material types matching CudaScene::MaterialType
enum class OptixMaterialType : unsigned char
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

// Geometry types for intersection dispatch
enum class OptixGeomType : unsigned char
{
   SPHERE,
   RECTANGLE,
   DISPLACED_SPHERE,
   TRIANGLE
};

// Per-geometry data stored in SBT hit group record
struct HitGroupData
{
   OptixGeomType geom_type;
   int material_idx; // Index into materials array

   // Geometry parameters (union-like, but flat for simplicity)
   float3 center;  // Sphere center / rectangle corner
   float radius;   // Sphere radius
   float3 u_vec;   // Rectangle edge u
   float3 v_vec;   // Rectangle edge v
   float3 normal;  // Precomputed rectangle normal

   // Triangle vertices and per-vertex normals
   float3 tri_v0, tri_v1, tri_v2;
   float3 tri_n0, tri_n1, tri_n2;
   int    tri_has_normals; // 1 = interpolate per-vertex normals, 0 = use face normal
};

// Material data (flat struct, uploaded as array)
struct OptixMaterialData
{
   OptixMaterialType type;
   float3 albedo;
   float3 emission;
   float roughness;
   float refractive_index;

   // Procedural pattern
   unsigned char pattern; // 0=none, 1=fibonacci_dots
   float3 pattern_color;
   float pattern_param1;
   float pattern_param2;
};

// Launch parameters — passed to all OptiX programs via __constant__ memory
struct OptixLaunchParams
{
   // Output
   float4 *accum_buffer;  // Accumulation buffer (float4 per pixel)
   unsigned int width;
   unsigned int height;

   // Camera
   float3 camera_center;
   float3 pixel00_loc;
   float3 pixel_delta_u;
   float3 pixel_delta_v;
   float3 cam_u;
   float3 cam_v;

   // Rendering
   int samples_per_launch;
   int total_samples_so_far;
   int max_depth;
   unsigned int frame_seed;

   // Scene
   OptixMaterialData *materials;
   int num_materials;

   // Traversal
#ifdef __CUDACC__
   OptixTraversableHandle traversable;
#else
   unsigned long long traversable; // Same size, usable from host
#endif

   // Depth of field
   bool dof_enabled;
   float dof_aperture;
   float dof_focus_distance;

   // Environment
   float background_intensity;
};

// Per-ray data passed through payload pointer.
// Kept minimal to reduce register pressure and stack spill — only fields
// that must cross the trace() boundary (raygen ↔ closesthit/miss).
struct PRDRadiance
{
   float3 hit_normal;
   float3 hit_point;
   float3 hit_color;
   float3 hit_emission;
   float hit_roughness;
   float hit_refractive_index;
   OptixMaterialType hit_material_type;
   unsigned int seed;
   bool hit;
   bool front_face;
};

// SBT record template
template <typename T>
struct alignas(16) SbtRecord
{
   char header[32]; // OPTIX_SBT_RECORD_HEADER_SIZE = 32
   T data;
};

struct RayGenData
{
};
struct MissData
{
};

using RayGenRecord = SbtRecord<RayGenData>;
using MissRecord = SbtRecord<MissData>;
using HitGroupRecord = SbtRecord<HitGroupData>;
