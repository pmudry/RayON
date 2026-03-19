// Shared data structures for OptiX host and device code
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#ifdef __CUDACC__
#include <optix.h>
#endif

// Material types matching Scene::MaterialType — keep in sync with scene_description.hpp
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
   SDF_MATERIAL,
   ANISOTROPIC_METAL, // GGX microfacet metal — approximated as rough mirror in OptiX
   THIN_FILM,         // Thin-film interference — approximated as mirror in OptiX
   CLEAR_COAT         // Dielectric coat over diffuse base (Schlick Fresnel blend)
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

   // Triangle per-vertex UV coordinates
   float2 tri_uv0, tri_uv1, tri_uv2;
   int    tri_has_uvs; // 1 = has UV coords, 0 = no UV
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

   // Extra material parameters
   float anisotropy;     // ANISOTROPIC_METAL: anisotropy ratio
   float3 eta;           // ANISOTROPIC_METAL: complex IOR real part (R,G,B)
   float3 k;             // ANISOTROPIC_METAL: complex IOR imaginary/extinction (R,G,B)
   float film_thickness; // THIN_FILM: thickness in nm
   float film_ior;       // THIN_FILM: film refractive index

   int texture_id; // Diffuse texture index (-1 = none)
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
   bool use_sobol;  ///< Use Sobol' quasi-random sampler instead of PCG (default: true)

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
   cudaTextureObject_t hdr_env_tex;  // equirectangular HDR sky (0 = gradient fallback)
   bool                use_hdr_env;

   // Dynamic material multipliers (set per-frame from GUI sliders)
   float light_intensity;      // Multiplier on emissive materials
   float metal_fuzziness;      // Multiplier on roughness of metallic materials
   float glass_ior_multiplier; // Multiplier on refractive index of glass/dielectric

   // Golf ball dimple parameters (runtime-adjustable via GUI sliders)
   int   golf_dimple_count;  // Number of dimples (Fibonacci sphere distribution)
   float golf_dimple_radius; // Angular radius of each dimple (radians)
   float golf_dimple_depth;  // Depth of each dimple (displacement factor)

   // Textures
   cudaTextureObject_t *d_textures; // Device array of CUDA texture objects
   int num_textures;
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
   float hit_film_thickness;   // THIN_FILM: film thickness in nm
   float hit_film_ior;         // THIN_FILM: film refractive index
   float2 hit_uv;              // Interpolated UV at hit point (for texture sampling)
   float3 hit_tangent;         // Surface tangent (for anisotropic materials)
   float3 hit_eta;             // ANISOTROPIC_METAL: complex IOR real part
   float3 hit_k;               // ANISOTROPIC_METAL: complex IOR imaginary/extinction
   float hit_anisotropy;       // ANISOTROPIC_METAL: anisotropy ratio
   OptixMaterialType hit_material_type;
   unsigned int seed;          ///< PCG seed (stateful fallback RNG)
   uint32_t sobol_sample_idx;  ///< Gray-code sample index for Sobol path
   uint32_t sobol_dim_idx;     ///< Dimension counter — incremented per rand_float call
   uint32_t sobol_pixel_hash;  ///< Per-pixel stable hash for Owen scrambling
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
