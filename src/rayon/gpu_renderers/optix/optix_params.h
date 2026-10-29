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

// Custom-primitive geometry types (triangles go through the built-in intersector)
enum class OptixGeomType : unsigned char
{
   SPHERE,
   RECTANGLE,
   DISPLACED_SPHERE
};

// Per-geometry data stored in SBT hit group record (custom primitives only:
// spheres, displaced spheres, rectangles). Triangles use the hardware triangle
// intersector and a single shared SBT record; their per-primitive data lives in
// OptixTriangleData, indexed by optixGetPrimitiveIndex().
struct HitGroupData
{
   OptixGeomType geom_type;
   int material_idx; // Index into materials array

   float3 center;  // Sphere center / rectangle corner
   float radius;   // Sphere radius
   float3 u_vec;   // Rectangle edge u
   float3 v_vec;   // Rectangle edge v
   float3 normal;  // Precomputed rectangle normal
};

// Per-triangle shading attributes (read once per closest hit, never during traversal)
struct OptixTriangleData
{
   float3 n0, n1, n2;    // Per-vertex normals (valid when has_normals)
   float2 uv0, uv1, uv2; // Per-vertex texture coordinates (valid when has_uvs)
   int material_idx;
   int has_normals; // 1 = interpolate per-vertex normals, 0 = use face normal
   int has_uvs;     // 1 = has UV coords, 0 = no UV
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
   int pattern_table_offset; // FIBONACCI_DOTS: first entry of this material's dot directions in params.pattern_points
};

// Light geometry data for Next Event Estimation (NEE)
enum class OptixLightGeomType : unsigned char
{
   RECTANGLE,
   SPHERE
};

struct OptixLightData
{
   OptixLightGeomType geom_type;
   float3 emission; // Already scaled by material emission
   // Rectangle fields
   float3 corner;
   float3 u_vec;
   float3 v_vec;
   float3 normal;
   float  area;
   // Sphere fields
   float3 center;
   float  radius;
};

// Launch parameters — passed to all OptiX programs via __constant__ memory
// Streak value that marks a converged pixel as "being re-checked" (sampled again for one
// checkpoint). Shared by raygen and the adaptive bookkeeping kernel.
constexpr unsigned char ADAPTIVE_RECHECK_MARK = 255u;

struct OptixLaunchParams
{
   // Output
   float4 *accum_buffer;  // Accumulation buffer (float4 per pixel), render resolution
   unsigned int width;    // Render resolution (may be lower than the display while the camera moves)
   unsigned int height;

   // Adaptive sampling: per-pixel sample counts, negative = converged (skip). Null when disabled.
   const int *pixel_sample_counts;
   // Per-pixel convergence streak (see optixAdaptiveUpdateKernel). A converged pixel whose streak
   // equals ADAPTIVE_RECHECK_MARK is being re-checked and must be sampled despite its negative
   // count. Null when adaptive sampling is disabled.
   const unsigned char *conv_streak;
   // Per-pixel running sum of squared sample luminance, kept by raygen so the adaptive pass can
   // estimate each pixel's variance. Always valid (counts stay accurate while adaptive is off).
   float *lum_sq_buffer;
   // Denoiser guide layers, accumulated like accum_buffer (sums over samples): albedo and
   // camera-space normal of each sample's first non-specular hit (background: sky colour, zero
   // normal). Null when the guides are not collected.
   float4 *albedo_buffer;
   float4 *normal_buffer;

   // Camera
   float3 camera_center;
   float3 pixel00_loc;
   float3 pixel_delta_u;
   float3 pixel_delta_v;
   float3 cam_u;
   float3 cam_v;
   float3 cam_w; ///< Camera basis third axis (u x v, towards the viewer) for camera-space normals

   // Rendering
   int samples_per_launch;
   int total_samples_so_far;
   int max_depth;
   unsigned int frame_seed;
   unsigned int sobol_sequence_seed; ///< Sobol scramble seed, constant within one accumulation
   bool use_sobol;  ///< Use Sobol' quasi-random sampler instead of PCG (default: true)
   float firefly_clamp_ceiling; ///< > 0: fixed per-sample luminance ceiling (camera moving); 0: grows with samples

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
   const float3 *golf_points;    // Precomputed unit dimple directions [golf_dimple_count]
   const float3 *pattern_points; // Precomputed FIBONACCI_DOTS directions for all materials (see pattern_table_offset)

   // Textures
   cudaTextureObject_t *d_textures; // Device array of CUDA texture objects
   int num_textures;

   // Light list for Next Event Estimation (NEE / MIS)
   OptixLightData *lights;     // Device array of emissive geometries
   int             num_lights;
   float          *light_cdfs; // Cumulative area CDF [num_lights+1] for importance selection

   // MIS / NEE runtime controls
   bool mis_enabled;            // Master MIS toggle
   int  nee_stride;             // NEE on 1 of every N samples (contribution x N)

   // Reverse lookup: custom-primitive index → light list index (-1 if not a light)
   int *geom_to_light_map;
   int  num_geom_entries;       // size of geom_to_light_map array

   // Hardware triangles: vertex positions (3 per triangle, same order as the GAS build input)
   // and per-triangle shading attributes, both indexed by optixGetPrimitiveIndex().
   const float3            *tri_vertices;
   const OptixTriangleData *tri_data;
   int                      num_triangles;
};

// Miss SBT indices (see trace() / trace_shadow() in optix_programs.cu)
enum : unsigned int
{
   OPTIX_MISS_RADIANCE = 0,
   OPTIX_MISS_SHADOW   = 1
};

// Per-ray data passed through payload pointer — only what must cross the trace()
// boundary (closesthit → raygen). Material parameters are looked up by raygen from
// params.materials[material_idx]; RNG state never leaves raygen.
struct PRDRadiance
{
   float3 hit_normal;  // Shading normal, flipped to face the ray
   float3 hit_point;
   float3 hit_tangent; // Surface tangent (for anisotropic materials)
   float3 hit_color;   // Albedo after texture / procedural pattern
   int material_idx;   // -1 = invalid material (rendered magenta)
   int hit_prim_idx;   // Custom-primitive index (for light identification in MIS), -1 for triangles
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
