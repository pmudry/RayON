// Host-side OptiX renderer implementation
// Handles: context, module, pipeline, GAS, SBT, launch

#include "optix_params.h"
#include "volume_sdf.cuh"
#include "scene_description.hpp"
#include "tone_mapping.hpp"

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Macro for OptiX error checking
#define OPTIX_CHECK(call)                                                                                              \
   do                                                                                                                  \
   {                                                                                                                   \
      OptixResult res = call;                                                                                          \
      if (res != OPTIX_SUCCESS)                                                                                        \
      {                                                                                                                \
         printf("OptiX error: %s at %s:%d\n", optixGetErrorString(res), __FILE__, __LINE__);                           \
      }                                                                                                                \
   } while (0)

#define CUDA_CHECK(call)                                                                                               \
   do                                                                                                                  \
   {                                                                                                                   \
      cudaError_t err = call;                                                                                          \
      if (err != cudaSuccess)                                                                                          \
      {                                                                                                                \
         printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);                             \
      }                                                                                                                \
   } while (0)

// Log buffer for OptiX
static char optix_log[2048];
static size_t optix_log_size = sizeof(optix_log);

static void optixLogCallback(unsigned int level, const char *tag, const char *message, void *)
{
   if (level <= 3) // Only print warnings and errors
      printf("[OptiX][%s]: %s\n", tag, message);
}

// Persistent OptiX state (created once, reused across frames)
struct OptixState
{
   OptixDeviceContext context = nullptr;
   OptixModule module = nullptr;
   OptixPipeline pipeline = nullptr;
   OptixShaderBindingTable sbt = {};

   OptixProgramGroup raygen_pg = nullptr;
   OptixProgramGroup raygen_caustics_pg = nullptr; // light tracing pass of the caustics
   OptixProgramGroup miss_pg = nullptr;
   OptixProgramGroup miss_shadow_pg = nullptr;
   OptixProgramGroup hitgroup_sphere_pg = nullptr;
   OptixProgramGroup hitgroup_rect_pg = nullptr;
   OptixProgramGroup hitgroup_sdf_pg = nullptr; // ray-marched SDF primitives
   OptixProgramGroup hitgroup_triangle_pg = nullptr; // built-in triangles: closest-hit only

   // Scene traversable: a one-level IAS over (at most) two GASes, because OptiX
   // cannot mix built-in triangles and custom primitives in a single GAS.
   OptixTraversableHandle gas_handle = 0;        // IAS handle passed to optixTrace
   CUdeviceptr d_ias_output = 0;
   CUdeviceptr d_custom_gas_output = 0;          // spheres / displaced spheres / rectangles
   CUdeviceptr d_triangle_gas_output = 0;        // hardware-intersected triangles
   float3 *d_tri_vertices = nullptr;             // 3 vertices per triangle
   OptixTriangleData *d_tri_data = nullptr;      // per-triangle normals / UVs / material
   int num_triangles = 0;

   // Device buffers
   CUdeviceptr d_sbt_raygen = 0;
   CUdeviceptr d_sbt_raygen_caustics = 0;
   CUdeviceptr d_sbt_miss = 0;
   CUdeviceptr d_sbt_hitgroup = 0;
   OptixMaterialData *d_materials = nullptr;
   std::vector<OptixMaterialData> host_materials; // mirror of d_materials for in-place updates
   cudaTextureObject_t *d_textures = nullptr;
   int num_textures = 0;

   // Caustic pass: bounding spheres of the geometry whose material may be a delta surface. The
   // candidates (one per custom primitive, one per triangle material) are kept so that a live
   // material change can re-filter them (uploadCausticTargets).
   struct CausticCandidate
   {
      int    material_id;
      float4 sphere; // xyz centre, w radius
   };
   std::vector<CausticCandidate> caustic_candidates;
   float4 *d_caustic_targets  = nullptr;
   int     num_caustic_targets = 0;
   bool    caustics_enabled    = true;

   // Light list for NEE / MIS
   OptixLightData *d_lights = nullptr;
   float          *d_light_cdfs = nullptr;
   int            *d_geom_to_light = nullptr;
   int             num_lights = 0;
   int             num_geom_entries = 0;

   // MIS / NEE runtime settings (cached between launches)
   bool  mis_enabled          = true;
   int   nee_stride           = 1;
   float light_falloff        = 2.0f;
   float indirect_clamp       = 50.0f;
   bool  has_volume           = false; // scene holds a VOLUME material (shadow rays track transmittance)
   bool  has_thin_film        = false; // scene holds a THIN_FILM material (shadow rays pass through films)

   // Adaptive sampling (converged-pixel skipping): per-pixel sample counts (negative = converged)
   // and the per-pixel sum of squared sample luminance for the convergence test.
   bool   adaptive_enabled     = false;
   int    adaptive_min_samples = 32;
   float  adaptive_threshold   = 3.16e-5f;
   int   *d_pixel_sample_counts = nullptr;
   float *d_lum_sq              = nullptr; // per-pixel sum of squared sample luminance (variance)
   unsigned char *d_conv_streak = nullptr; // consecutive checks that passed the stability test
   int   *d_converged_count     = nullptr;

   // Resolution of the last launch (the display may be larger while the camera moves)
   int render_width  = 0;
   int render_height = 0;
   // Hand-over prior: when a reduced-resolution motion pass ends, its upscaled mean image is kept
   // here and blended into the display with a weight that fades to zero as new samples arrive.
   // Display-only: the accumulation buffer itself stays unbiased.
   float4 *d_prior        = nullptr;
   float4 *d_resolved     = nullptr; // display pass 1 output (render resolution, linear)
   bool    prior_active   = false;
   float   prior_strength = 0.0f; // weight in "samples" at the start of the fade

   // OptiX AI denoiser (motion frames and the first frames after the camera stops). The guides are
   // accumulated by raygen at accumulation resolution (sums), resolved to per-pixel means at render
   // resolution by the resolve pass, then the denoiser reads d_resolved + guides and writes
   // d_denoised, which the gamma pass displays instead of d_resolved.
   float4 *d_albedo_accum = nullptr; // sum of per-sample albedo guides (accum resolution)
   float4 *d_normal_accum = nullptr; // sum of per-sample camera-space normals (accum resolution)
   float4 *d_guide_albedo = nullptr; // per-pixel mean albedo (render resolution)
   float4 *d_guide_normal = nullptr; // per-pixel mean normal (render resolution)
   float4 *d_denoised     = nullptr; // denoiser output (render resolution, linear)
   OptixDenoiser denoiser = nullptr;
   CUdeviceptr d_denoiser_state     = 0;
   CUdeviceptr d_denoiser_scratch   = 0;
   CUdeviceptr d_denoiser_intensity = 0; // one float: log-average intensity of the input (auto-exposure)
   size_t denoiser_state_size   = 0;
   size_t denoiser_scratch_size = 0;
   int  denoiser_max_w   = 0, denoiser_max_h   = 0; // dimensions the state/scratch were sized for
   int  denoiser_setup_w = 0, denoiser_setup_h = 0; // dimensions of the last optixDenoiserSetup
   bool denoiser_failed  = false;                   // creation failed: denoising is skipped
   cudaEvent_t denoise_start = nullptr, denoise_end = nullptr;
   float last_denoise_ms = 0.0f;

   // Persistent device accumulation buffer — stays on GPU across batches
   // to avoid costly per-batch host↔device float3↔float4 round-trips
   float4 *d_accum_buffer = nullptr;
   int accum_width = 0;
   int accum_height = 0;

   // Persistent device launch params — avoids cudaMalloc/cudaFree per batch
   CUdeviceptr d_launch_params = 0;

   // Golf ball dimple parameters (cached between launches, set via optixRendererSetGolfDimples)
   int   golf_dimple_count  = 128;
   float golf_dimple_radius = 0.155f;
   float golf_dimple_depth  = 0.25f;
   float3 *d_golf_points    = nullptr; // precomputed unit dimple directions [golf_dimple_count]
   int     golf_points_count = 0;

   // Precomputed FIBONACCI_DOTS directions for every patterned material (offsets in OptixMaterialData)
   float3 *d_pattern_points = nullptr;

   // HDR environment map managed outside the scene textures to survive scene rebuilds
   cudaArray_t         hdr_cuda_array = nullptr;
   cudaTextureObject_t hdr_tex_obj    = 0;
   // Sky importance-sampling tables of the current HDR map (see buildEnvSamplingTables)
   float *d_env_marginal_cdf    = nullptr;
   float *d_env_conditional_cdf = nullptr;
   int    env_w = 0;
   int    env_h = 0;

   // Dedicated CUDA stream for OptiX launches — avoids blocking the default stream
   // and enables stream-specific synchronization instead of cudaDeviceSynchronize().
   cudaStream_t render_stream = nullptr;

   // Pinned host memory + device display buffer for async gamma correction + D2H pipeline
   unsigned char *pinned_display = nullptr;
   size_t pinned_display_size = 0;
   unsigned char *d_display = nullptr;
   size_t d_display_size = 0;

   bool initialized = false;
};

static OptixState g_state;

// Helper: return the dedicated render stream, or the default stream (0) if not initialized.
static inline cudaStream_t getOptiXStream() { return g_state.render_stream ? g_state.render_stream : 0; }

// Host-side Fibonacci sphere point i of n (same formula the device programs used to evaluate per hit)
static float3 fibonacciPoint(int i, int n)
{
   const float ga = 2.39996323f;
   float k = (float)i + 0.5f;
   float phi = acosf(1.0f - 2.0f * k / (float)n);
   float theta = ga * k;
   float s = sinf(phi);
   return make_float3(cosf(theta) * s, sinf(theta) * s, cosf(phi));
}

// (Re)build the device table of golf dimple directions when the count changes
static void uploadGolfPointTable()
{
   const int n = std::max(1, g_state.golf_dimple_count);
   if (g_state.d_golf_points && g_state.golf_points_count == n)
      return;
   if (g_state.d_golf_points)
      CUDA_CHECK(cudaFree(g_state.d_golf_points));
   std::vector<float3> pts(static_cast<size_t>(n));
   for (int i = 0; i < n; ++i)
      pts[i] = fibonacciPoint(i, n);
   CUDA_CHECK(cudaMalloc(&g_state.d_golf_points, pts.size() * sizeof(float3)));
   CUDA_CHECK(cudaMemcpy(g_state.d_golf_points, pts.data(), pts.size() * sizeof(float3), cudaMemcpyHostToDevice));
   g_state.golf_points_count = n;
   // Launches run on a non-blocking stream that does not wait for legacy-stream DMA: fence here.
   CUDA_CHECK(cudaDeviceSynchronize());
}

// Load PTX from file
static std::string loadPTXFromFile(const char *filename)
{
   std::ifstream file(filename);
   if (!file.is_open())
   {
      printf("Failed to open PTX file: %s\n", filename);
      return "";
   }
   std::stringstream ss;
   ss << file.rdbuf();
   return ss.str();
}

// Find PTX file - try multiple locations
static std::string findAndLoadPTX()
{
   const char *candidates[] = {"optix_programs.ptx", "../optix_programs.ptx", "gpu_renderers/optix/optix_programs.ptx",
                                nullptr};

   for (int i = 0; candidates[i] != nullptr; ++i)
   {
      std::string ptx = loadPTXFromFile(candidates[i]);
      if (!ptx.empty())
      {
         printf("Loaded OptiX PTX from: %s\n", candidates[i]);
         return ptx;
      }
   }

   printf("ERROR: Could not find optix_programs.ptx\n");
   return "";
}

static void initializeOptiX()
{
   if (g_state.initialized)
      return;

   // Initialize CUDA
   CUDA_CHECK(cudaFree(0));

   // Initialize OptiX
   OPTIX_CHECK(optixInit());

   // Create context
   CUcontext cuCtx = 0;
   OptixDeviceContextOptions ctx_options = {};
   ctx_options.logCallbackFunction = &optixLogCallback;
   ctx_options.logCallbackLevel = 4;
   OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &ctx_options, &g_state.context));

   // Load PTX and create module
   std::string ptx = findAndLoadPTX();
   if (ptx.empty())
      return;

   OptixModuleCompileOptions module_options = {};
   module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3; // Maximum optimization
   module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

   OptixPipelineCompileOptions pipeline_options = {};
   pipeline_options.usesMotionBlur = false;
   pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
   pipeline_options.numPayloadValues = 2;   // PRD pointer (2 x uint32); shadow rays use only payload 0
   pipeline_options.numAttributeValues = 3; // Custom primitives report their normal (3 x float)
   pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
   pipeline_options.pipelineLaunchParamsVariableName = "params";

   optix_log_size = sizeof(optix_log);
   OPTIX_CHECK(optixModuleCreate(g_state.context, &module_options, &pipeline_options, ptx.c_str(), ptx.size(),
                                  optix_log, &optix_log_size, &g_state.module));

   // Create program groups
   OptixProgramGroupOptions pg_options = {};

   // Raygen
   OptixProgramGroupDesc raygen_desc = {};
   raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
   raygen_desc.raygen.module = g_state.module;
   raygen_desc.raygen.entryFunctionName = "__raygen__rg";
   optix_log_size = sizeof(optix_log);
   OPTIX_CHECK(
       optixProgramGroupCreate(g_state.context, &raygen_desc, 1, &pg_options, optix_log, &optix_log_size, &g_state.raygen_pg));

   // Raygen of the caustic pass (light tracing), launched with its own SBT raygen record
   OptixProgramGroupDesc raygen_caustics_desc = {};
   raygen_caustics_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
   raygen_caustics_desc.raygen.module = g_state.module;
   raygen_caustics_desc.raygen.entryFunctionName = "__raygen__caustics";
   optix_log_size = sizeof(optix_log);
   OPTIX_CHECK(optixProgramGroupCreate(g_state.context, &raygen_caustics_desc, 1, &pg_options, optix_log,
                                        &optix_log_size, &g_state.raygen_caustics_pg));

   // Miss
   OptixProgramGroupDesc miss_desc = {};
   miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
   miss_desc.miss.module = g_state.module;
   miss_desc.miss.entryFunctionName = "__miss__ms";
   optix_log_size = sizeof(optix_log);
   OPTIX_CHECK(
       optixProgramGroupCreate(g_state.context, &miss_desc, 1, &pg_options, optix_log, &optix_log_size, &g_state.miss_pg));

   // Miss for shadow rays (clears the "occluded" payload flag)
   OptixProgramGroupDesc miss_shadow_desc = {};
   miss_shadow_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
   miss_shadow_desc.miss.module = g_state.module;
   miss_shadow_desc.miss.entryFunctionName = "__miss__shadow";
   optix_log_size = sizeof(optix_log);
   OPTIX_CHECK(optixProgramGroupCreate(g_state.context, &miss_shadow_desc, 1, &pg_options, optix_log, &optix_log_size,
                                        &g_state.miss_shadow_pg));

   // Hit group for spheres (custom intersection)
   OptixProgramGroupDesc hitgroup_sphere_desc = {};
   hitgroup_sphere_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
   hitgroup_sphere_desc.hitgroup.moduleCH = g_state.module;
   hitgroup_sphere_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
   hitgroup_sphere_desc.hitgroup.moduleIS = g_state.module;
   hitgroup_sphere_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
   optix_log_size = sizeof(optix_log);
   OPTIX_CHECK(optixProgramGroupCreate(g_state.context, &hitgroup_sphere_desc, 1, &pg_options, optix_log, &optix_log_size,
                                        &g_state.hitgroup_sphere_pg));

   // Hit group for rectangles (custom intersection)
   OptixProgramGroupDesc hitgroup_rect_desc = {};
   hitgroup_rect_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
   hitgroup_rect_desc.hitgroup.moduleCH = g_state.module;
   hitgroup_rect_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
   hitgroup_rect_desc.hitgroup.moduleIS = g_state.module;
   hitgroup_rect_desc.hitgroup.entryFunctionNameIS = "__intersection__rectangle";
   optix_log_size = sizeof(optix_log);
   OPTIX_CHECK(optixProgramGroupCreate(g_state.context, &hitgroup_rect_desc, 1, &pg_options, optix_log, &optix_log_size,
                                        &g_state.hitgroup_rect_pg));

   OptixProgramGroupDesc hitgroup_sdf_desc = {};
   hitgroup_sdf_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
   hitgroup_sdf_desc.hitgroup.moduleCH = g_state.module;
   hitgroup_sdf_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
   hitgroup_sdf_desc.hitgroup.moduleIS = g_state.module;
   hitgroup_sdf_desc.hitgroup.entryFunctionNameIS = "__intersection__sdf";
   optix_log_size = sizeof(optix_log);
   OPTIX_CHECK(optixProgramGroupCreate(g_state.context, &hitgroup_sdf_desc, 1, &pg_options, optix_log, &optix_log_size,
                                        &g_state.hitgroup_sdf_pg));

   // Hit group for triangles: built-in (RT core) intersection, so no IS program
   OptixProgramGroupDesc hitgroup_triangle_desc = {};
   hitgroup_triangle_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
   hitgroup_triangle_desc.hitgroup.moduleCH = g_state.module;
   hitgroup_triangle_desc.hitgroup.entryFunctionNameCH = "__closesthit__triangle";
   optix_log_size = sizeof(optix_log);
   OPTIX_CHECK(optixProgramGroupCreate(g_state.context, &hitgroup_triangle_desc, 1, &pg_options, optix_log, &optix_log_size,
                                        &g_state.hitgroup_triangle_pg));

   // Create pipeline
   const uint32_t max_trace_depth = 1; // We loop in raygen, single trace per bounce
   OptixProgramGroup program_groups[] = {g_state.raygen_pg,          g_state.raygen_caustics_pg, g_state.miss_pg,
                                          g_state.miss_shadow_pg,     g_state.hitgroup_sphere_pg,
                                          g_state.hitgroup_rect_pg,   g_state.hitgroup_triangle_pg,
                                          g_state.hitgroup_sdf_pg};
   const unsigned int num_program_groups = sizeof(program_groups) / sizeof(program_groups[0]);

   OptixPipelineLinkOptions link_options = {};
   link_options.maxTraceDepth = max_trace_depth;
   optix_log_size = sizeof(optix_log);
   OPTIX_CHECK(optixPipelineCreate(g_state.context, &pipeline_options, &link_options, program_groups,
                                    num_program_groups, optix_log, &optix_log_size, &g_state.pipeline));

   // Compute stack sizes
   OptixStackSizes stack_sizes = {};
   for (auto &pg : program_groups)
      OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes, g_state.pipeline));

   uint32_t dc_from_traversal, dc_from_state, continuation;
   OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth, 0, 0, &dc_from_traversal, &dc_from_state,
                                           &continuation));
   // Traversable graph depth 2: one IAS level over the GASes
   OPTIX_CHECK(optixPipelineSetStackSize(g_state.pipeline, dc_from_traversal, dc_from_state, continuation, 2));

   // Create raygen SBT record
   RayGenRecord rg_record;
   OPTIX_CHECK(optixSbtRecordPackHeader(g_state.raygen_pg, &rg_record));
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_state.d_sbt_raygen), sizeof(RayGenRecord)));
   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(g_state.d_sbt_raygen), &rg_record, sizeof(RayGenRecord),
                          cudaMemcpyHostToDevice));

   // Create miss SBT records: [OPTIX_MISS_RADIANCE, OPTIX_MISS_SHADOW]
   MissRecord ms_records[2];
   OPTIX_CHECK(optixSbtRecordPackHeader(g_state.miss_pg, &ms_records[OPTIX_MISS_RADIANCE]));
   OPTIX_CHECK(optixSbtRecordPackHeader(g_state.miss_shadow_pg, &ms_records[OPTIX_MISS_SHADOW]));
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_state.d_sbt_miss), sizeof(ms_records)));
   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(g_state.d_sbt_miss), ms_records, sizeof(ms_records),
                          cudaMemcpyHostToDevice));

   RayGenRecord rg_caustics_record;
   OPTIX_CHECK(optixSbtRecordPackHeader(g_state.raygen_caustics_pg, &rg_caustics_record));
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_state.d_sbt_raygen_caustics), sizeof(RayGenRecord)));
   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(g_state.d_sbt_raygen_caustics), &rg_caustics_record,
                          sizeof(RayGenRecord), cudaMemcpyHostToDevice));

   g_state.sbt.raygenRecord = g_state.d_sbt_raygen;
   g_state.sbt.missRecordBase = g_state.d_sbt_miss;
   g_state.sbt.missRecordStrideInBytes = sizeof(MissRecord);
   g_state.sbt.missRecordCount = 2;

   // Create a dedicated CUDA stream for OptiX launches — enables stream-specific
   // synchronization instead of cudaDeviceSynchronize(), and allows overlap with
   // display/gamma correction work.
   if (g_state.render_stream == nullptr)
      CUDA_CHECK(cudaStreamCreateWithFlags(&g_state.render_stream, cudaStreamNonBlocking));

   g_state.initialized = true;
   printf("OptiX renderer initialized successfully\n");
}

/// Materials that can make a delta surface (see is_caustic_specular_optix in optix_programs.cu)
static bool isCausticSpecular(OptixMaterialType t)
{
   return t == OptixMaterialType::MIRROR || t == OptixMaterialType::METAL || t == OptixMaterialType::ROUGH_MIRROR ||
          t == OptixMaterialType::GLASS || t == OptixMaterialType::DIELECTRIC ||
          t == OptixMaterialType::ANISOTROPIC_METAL || t == OptixMaterialType::THIN_FILM;
}

// Upload the bounding spheres of the candidates whose material is currently caustic-specular: the
// directions the caustic pass aims its light paths at. Called after a scene build and after every
// live material change (a mesh swapped to glass must start receiving light paths).
static void uploadCausticTargets()
{
   std::vector<float4> targets;
   for (const auto &c : g_state.caustic_candidates)
      if (c.material_id >= 0 && c.material_id < (int)g_state.host_materials.size() &&
          isCausticSpecular(g_state.host_materials[c.material_id].type))
         targets.push_back(c.sphere);
   if (g_state.d_caustic_targets)
   {
      CUDA_CHECK(cudaFree(g_state.d_caustic_targets));
      g_state.d_caustic_targets = nullptr;
   }
   g_state.num_caustic_targets = static_cast<int>(targets.size());
   if (!targets.empty())
   {
      CUDA_CHECK(cudaMalloc(&g_state.d_caustic_targets, targets.size() * sizeof(float4)));
      CUDA_CHECK(cudaMemcpy(g_state.d_caustic_targets, targets.data(), targets.size() * sizeof(float4),
                             cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaDeviceSynchronize()); // the render stream does not wait for legacy-stream copies
   }
}

// Build GAS from scene description
// One-time warning flag for unsupported geometry types in OptiX GAS build
static bool g_optix_warned_unsupported_geometry = false;

static void freeDevicePtr(CUdeviceptr &p)
{
   if (p)
   {
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(p)));
      p = 0;
   }
}

// Build one GAS (with compaction) from a single build input. Returns the handle; the
// output buffer is stored in `out_buffer`.
static OptixTraversableHandle buildSingleGAS(const OptixBuildInput &build_input, CUdeviceptr &out_buffer)
{
   OptixAccelBuildOptions accel_options = {};
   accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
   accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

   OptixAccelBufferSizes gas_sizes;
   OPTIX_CHECK(optixAccelComputeMemoryUsage(g_state.context, &accel_options, &build_input, 1, &gas_sizes));

   CUdeviceptr d_temp;
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp), gas_sizes.tempSizeInBytes));

   CUdeviceptr d_output;
   size_t compacted_offset = ((gas_sizes.outputSizeInBytes + 7ull) / 8ull) * 8ull;
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_output), compacted_offset + 8));

   OptixAccelEmitDesc emit_desc = {};
   emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
   emit_desc.result = d_output + compacted_offset;

   OptixTraversableHandle handle = 0;
   OPTIX_CHECK(optixAccelBuild(g_state.context, 0, &accel_options, &build_input, 1, d_temp, gas_sizes.tempSizeInBytes,
                                d_output, gas_sizes.outputSizeInBytes, &handle, &emit_desc, 1));
   CUDA_CHECK(cudaDeviceSynchronize());

   // Compact
   size_t compacted_size;
   CUDA_CHECK(cudaMemcpy(&compacted_size, reinterpret_cast<void *>(d_output + compacted_offset), sizeof(size_t),
                          cudaMemcpyDeviceToHost));
   if (compacted_size < gas_sizes.outputSizeInBytes)
   {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&out_buffer), compacted_size));
      OPTIX_CHECK(optixAccelCompact(g_state.context, 0, handle, out_buffer, compacted_size, &handle));
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_output)));
   }
   else
   {
      out_buffer = d_output;
   }

   CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp)));
   return handle;
}

static inline float3 toFloat3(const Vec3 &v)
{
   return make_float3(static_cast<float>(v.x()), static_cast<float>(v.y()), static_cast<float>(v.z()));
}

// Build the scene traversable from the scene description:
//   * triangles  → one GAS with the built-in (RT core) triangle intersector, a single shared
//                  SBT record, and per-triangle attributes in params.tri_data
//   * spheres / displaced spheres / rectangles → one GAS of custom primitives, one SBT record each
//   * both GASes are referenced by a one-level IAS (OptiX forbids mixing the two in one GAS)
static void buildGAS(const Scene::SceneDescription &scene)
{
   // Free previous resources
   freeDevicePtr(g_state.d_ias_output);
   freeDevicePtr(g_state.d_custom_gas_output);
   freeDevicePtr(g_state.d_triangle_gas_output);
   freeDevicePtr(g_state.d_sbt_hitgroup);
   if (g_state.d_tri_vertices) { CUDA_CHECK(cudaFree(g_state.d_tri_vertices)); g_state.d_tri_vertices = nullptr; }
   if (g_state.d_tri_data)     { CUDA_CHECK(cudaFree(g_state.d_tri_data));     g_state.d_tri_data = nullptr; }
   g_state.num_triangles = 0;
   g_state.gas_handle = 0;

   // Volume materials with a soft edge: build the mesh's signed distance grid now, and trace a
   // padded box around the mesh instead of the mesh itself, so the noise can grow lumps outside
   // the surface. The materials upload below binds the grid; the box faces carry the material.
   struct VolumeBox
   {
      VolumeParams vp;
      float3       lo, hi;
   };
   std::map<int, VolumeBox> volume_boxes;
   for (int i = 0; i < static_cast<int>(scene.materials.size()); ++i)
   {
      const auto &m = scene.materials[i];
      if (m.type != Scene::MaterialType::VOLUME)
         continue;
      if (!m.grid_file.empty())
      {
         // Density grid from data: the grid's own box, unpadded (the loader's box mesh is not traced)
         VolumeBox box{};
         box.lo = toFloat3(m.grid_min);
         box.hi = toFloat3(m.grid_max);
         if (loadVolumeGrid(m.grid_file, box.lo, box.hi, box.vp))
            volume_boxes[i] = box;
         continue;
      }
      if (m.edge <= 0.0f)
         continue;
      std::vector<float3> tris;
      for (const auto &g : scene.geometries)
         if (g.type == Scene::GeometryType::TRIANGLE && g.material_id == i)
            for (const Vec3 *v : {&g.data.triangle.v0, &g.data.triangle.v1, &g.data.triangle.v2})
               tris.push_back(toFloat3(*v));
      VolumeBox box{};
      if (!buildVolumeSDF(tris, 64, VOLUME_BOX_PAD, box.vp))
         continue;
      box.lo = tris[0];
      box.hi = tris[0];
      for (const float3 &v : tris)
      {
         box.lo = make_float3(std::min(box.lo.x, v.x), std::min(box.lo.y, v.y), std::min(box.lo.z, v.z));
         box.hi = make_float3(std::max(box.hi.x, v.x), std::max(box.hi.y, v.y), std::max(box.hi.z, v.z));
      }
      box.lo = make_float3(box.lo.x - VOLUME_BOX_PAD, box.lo.y - VOLUME_BOX_PAD, box.lo.z - VOLUME_BOX_PAD);
      box.hi = make_float3(box.hi.x + VOLUME_BOX_PAD, box.hi.y + VOLUME_BOX_PAD, box.hi.z + VOLUME_BOX_PAD);
      volume_boxes[i] = box;
   }

   // Partition geometry: custom primitives (own SBT record each) vs. hardware triangles.
   // Unsupported types (CUBE, TRIANGLE_MESH) are skipped with a one-time warning.
   std::vector<int> supported_indices; // custom primitives, in SBT order
   std::vector<int> triangle_indices;
   for (int i = 0; i < static_cast<int>(scene.geometries.size()); ++i)
   {
      switch (scene.geometries[i].type)
      {
      case Scene::GeometryType::SPHERE:
      case Scene::GeometryType::RECTANGLE:
      case Scene::GeometryType::DISPLACED_SPHERE:
      case Scene::GeometryType::SDF_PRIMITIVE:
         supported_indices.push_back(i);
         break;
      case Scene::GeometryType::TRIANGLE:
         if (!volume_boxes.count(scene.geometries[i].material_id)) // boxed volumes: mesh not traced
            triangle_indices.push_back(i);
         break;
      default:
         if (!g_optix_warned_unsupported_geometry)
         {
            std::cerr << "[OptiX] Warning: skipping unsupported geometry type(s) in GAS build. "
                         "They will not be rendered by the OptiX backend.\n";
            g_optix_warned_unsupported_geometry = true;
         }
         break;
      }
   }

   const int num_geoms = static_cast<int>(supported_indices.size());
   const int num_tris  = static_cast<int>(triangle_indices.size()) + 12 * static_cast<int>(volume_boxes.size());

   // Caustic pass candidates: every custom primitive, and the triangles grouped by material, each
   // bounded by the sphere around its box (whatever the material now: a live swap may change it)
   g_state.caustic_candidates.clear();
   {
      auto sphere_of = [](float3 lo, float3 hi, float margin) {
         const float3 c = make_float3(0.5f * (lo.x + hi.x), 0.5f * (lo.y + hi.y), 0.5f * (lo.z + hi.z));
         const float  r = 0.5f * sqrtf((hi.x - lo.x) * (hi.x - lo.x) + (hi.y - lo.y) * (hi.y - lo.y) +
                                       (hi.z - lo.z) * (hi.z - lo.z));
         return make_float4(c.x, c.y, c.z, r * margin + 1e-3f);
      };
      for (int i : supported_indices)
      {
         const auto &g = scene.geometries[i];
         // Displaced spheres move their hit points outwards (up to 0.35 radius): extra margin
         const float margin = (g.type == Scene::GeometryType::DISPLACED_SPHERE) ? 1.4f : 1.01f;
         g_state.caustic_candidates.push_back({g.material_id, sphere_of(toFloat3(g.bounds_min), toFloat3(g.bounds_max), margin)});
      }
      std::map<int, std::pair<float3, float3>> mesh_boxes;
      for (int i : triangle_indices)
      {
         const auto &g = scene.geometries[i];
         const float3 lo = toFloat3(g.bounds_min), hi = toFloat3(g.bounds_max);
         auto it = mesh_boxes.find(g.material_id);
         if (it == mesh_boxes.end())
            mesh_boxes[g.material_id] = {lo, hi};
         else
         {
            float3 &a = it->second.first, &b = it->second.second;
            a = make_float3(std::min(a.x, lo.x), std::min(a.y, lo.y), std::min(a.z, lo.z));
            b = make_float3(std::max(b.x, hi.x), std::max(b.y, hi.y), std::max(b.z, hi.z));
         }
      }
      for (const auto &kv : mesh_boxes)
         g_state.caustic_candidates.push_back({kv.first, sphere_of(kv.second.first, kv.second.second, 1.01f)});
   }
   if (num_geoms == 0 && num_tris == 0)
      return;

   // SBT layout: [0, num_geoms) custom primitives, then one shared triangle record (if any).
   const int num_hit_records = num_geoms + (num_tris > 0 ? 1 : 0);
   std::vector<HitGroupRecord> hitgroup_records(num_hit_records);

   // ── Custom-primitive GAS ──────────────────────────────────────────────────
   OptixTraversableHandle custom_handle = 0;
   if (num_geoms > 0)
   {
      std::vector<OptixAabb> aabbs(num_geoms);
      std::vector<uint32_t> sbt_indices(num_geoms);
      for (int i = 0; i < num_geoms; ++i)
      {
         const auto &g = scene.geometries[supported_indices[i]];
         aabbs[i].minX = static_cast<float>(g.bounds_min.x());
         aabbs[i].minY = static_cast<float>(g.bounds_min.y());
         aabbs[i].minZ = static_cast<float>(g.bounds_min.z());
         aabbs[i].maxX = static_cast<float>(g.bounds_max.x());
         aabbs[i].maxY = static_cast<float>(g.bounds_max.y());
         aabbs[i].maxZ = static_cast<float>(g.bounds_max.z());
         sbt_indices[i] = static_cast<uint32_t>(i);
      }

      CUdeviceptr d_aabbs, d_sbt_indices;
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_aabbs), num_geoms * sizeof(OptixAabb)));
      CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_aabbs), aabbs.data(), num_geoms * sizeof(OptixAabb),
                             cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_sbt_indices), num_geoms * sizeof(uint32_t)));
      CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_sbt_indices), sbt_indices.data(),
                             num_geoms * sizeof(uint32_t), cudaMemcpyHostToDevice));

      std::vector<uint32_t> flags(num_geoms, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);

      OptixBuildInput build_input = {};
      build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
      build_input.customPrimitiveArray.aabbBuffers = &d_aabbs;
      build_input.customPrimitiveArray.numPrimitives = num_geoms;
      build_input.customPrimitiveArray.flags = flags.data();
      build_input.customPrimitiveArray.numSbtRecords = num_geoms;
      build_input.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_indices;
      build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
      build_input.customPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

      custom_handle = buildSingleGAS(build_input, g_state.d_custom_gas_output);

      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_aabbs)));
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_sbt_indices)));

      // Per-geometry SBT hit group records
      for (int i = 0; i < num_geoms; ++i)
      {
         const auto &g = scene.geometries[supported_indices[i]];
         HitGroupRecord &rec = hitgroup_records[i];
         OptixProgramGroup pg = (g.type == Scene::GeometryType::RECTANGLE)     ? g_state.hitgroup_rect_pg
                                : (g.type == Scene::GeometryType::SDF_PRIMITIVE) ? g_state.hitgroup_sdf_pg
                                                                                  : g_state.hitgroup_sphere_pg;
         OPTIX_CHECK(optixSbtRecordPackHeader(pg, &rec));

         rec.data.material_idx = g.material_id;
         rec.data.visible      = g.visible ? 1 : 0;

         switch (g.type)
         {
         case Scene::GeometryType::SPHERE:
            rec.data.geom_type = OptixGeomType::SPHERE;
            rec.data.center = toFloat3(g.data.sphere.center);
            rec.data.radius = static_cast<float>(g.data.sphere.radius);
            break;

         case Scene::GeometryType::DISPLACED_SPHERE:
            rec.data.geom_type = OptixGeomType::DISPLACED_SPHERE;
            rec.data.center = toFloat3(g.data.displaced_sphere.center);
            rec.data.radius = static_cast<float>(g.data.displaced_sphere.radius);
            break;

         case Scene::GeometryType::SDF_PRIMITIVE:
            rec.data.geom_type = OptixGeomType::SDF;
            rec.data.center    = toFloat3(g.data.sdf.position);
            rec.data.u_vec     = toFloat3(g.data.sdf.parameters);
            rec.data.v_vec     = toFloat3(g.data.sdf.rotation);
            rec.data.sdf_shape = static_cast<int>(g.data.sdf.sdf_type);
            break;

         case Scene::GeometryType::RECTANGLE:
         {
            rec.data.geom_type = OptixGeomType::RECTANGLE;
            rec.data.center = toFloat3(g.data.rectangle.corner);
            float3 u = toFloat3(g.data.rectangle.u);
            float3 v = toFloat3(g.data.rectangle.v);
            rec.data.u_vec = u;
            rec.data.v_vec = v;
            // Precompute normal = normalize(u x v)
            float3 n = make_float3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
            float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
            rec.data.normal = (len > 1e-8f) ? make_float3(n.x / len, n.y / len, n.z / len) : make_float3(0, 1, 0);
            break;
         }

         default:
            break; // unreachable: filtered above
         }
      }
   }

   // ── Triangle GAS (built-in intersection) ──────────────────────────────────
   OptixTraversableHandle triangle_handle = 0;
   if (num_tris > 0)
   {
      std::vector<float3> vertices(static_cast<size_t>(num_tris) * 3);
      std::vector<OptixTriangleData> tri_data(num_tris);
      // Volume boxes: 12 outward-wound triangles per box, flat shading, the volume material
      int next_tri = static_cast<int>(triangle_indices.size());
      for (const auto &kv : volume_boxes)
      {
         const float3 lo = kv.second.lo, hi = kv.second.hi;
         const float3 c[8] = {make_float3(lo.x, lo.y, lo.z), make_float3(hi.x, lo.y, lo.z),
                              make_float3(hi.x, hi.y, lo.z), make_float3(lo.x, hi.y, lo.z),
                              make_float3(lo.x, lo.y, hi.z), make_float3(hi.x, lo.y, hi.z),
                              make_float3(hi.x, hi.y, hi.z), make_float3(lo.x, hi.y, hi.z)};
         // Faces as quads (a, b, c, d) wound counter-clockwise seen from outside
         const int faces[6][4] = {{0, 3, 2, 1},  // -z
                                  {4, 5, 6, 7},  // +z
                                  {0, 1, 5, 4},  // -y
                                  {3, 7, 6, 2},  // +y
                                  {0, 4, 7, 3},  // -x
                                  {1, 2, 6, 5}}; // +x
         for (const auto &f : faces)
            for (int k = 0; k < 2; ++k)
            {
               const int ia = f[0], ib = k == 0 ? f[1] : f[2], ic = k == 0 ? f[2] : f[3];
               vertices[3 * next_tri + 0] = c[ia];
               vertices[3 * next_tri + 1] = c[ib];
               vertices[3 * next_tri + 2] = c[ic];
               OptixTriangleData &td = tri_data[next_tri];
               td              = OptixTriangleData{};
               td.material_idx = kv.first;
               td.visible      = 1;
               ++next_tri;
            }
      }
      for (int i = 0; i < static_cast<int>(triangle_indices.size()); ++i)
      {
         const auto &t = scene.geometries[triangle_indices[i]].data.triangle;
         vertices[3 * i + 0] = toFloat3(t.v0);
         vertices[3 * i + 1] = toFloat3(t.v1);
         vertices[3 * i + 2] = toFloat3(t.v2);

         OptixTriangleData &td = tri_data[i];
         td.material_idx = scene.geometries[triangle_indices[i]].material_id;
         td.visible      = scene.geometries[triangle_indices[i]].visible ? 1 : 0;
         td.has_normals  = t.has_normals ? 1 : 0;
         td.has_uvs      = t.has_uvs ? 1 : 0;
         td.n0 = toFloat3(t.n0);
         td.n1 = toFloat3(t.n1);
         td.n2 = toFloat3(t.n2);
         td.uv0 = make_float2(static_cast<float>(t.uv0.x()), static_cast<float>(t.uv0.y()));
         td.uv1 = make_float2(static_cast<float>(t.uv1.x()), static_cast<float>(t.uv1.y()));
         td.uv2 = make_float2(static_cast<float>(t.uv2.x()), static_cast<float>(t.uv2.y()));
      }

      CUDA_CHECK(cudaMalloc(&g_state.d_tri_vertices, vertices.size() * sizeof(float3)));
      CUDA_CHECK(cudaMemcpy(g_state.d_tri_vertices, vertices.data(), vertices.size() * sizeof(float3),
                             cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMalloc(&g_state.d_tri_data, tri_data.size() * sizeof(OptixTriangleData)));
      CUDA_CHECK(cudaMemcpy(g_state.d_tri_data, tri_data.data(), tri_data.size() * sizeof(OptixTriangleData),
                             cudaMemcpyHostToDevice));
      g_state.num_triangles = num_tris;

      CUdeviceptr d_vertices = reinterpret_cast<CUdeviceptr>(g_state.d_tri_vertices);
      uint32_t tri_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

      OptixBuildInput build_input = {};
      build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
      build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
      build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
      build_input.triangleArray.numVertices = static_cast<unsigned int>(vertices.size());
      build_input.triangleArray.vertexBuffers = &d_vertices;
      build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE; // non-indexed: 3 consecutive vertices
      build_input.triangleArray.flags = &tri_flags;
      build_input.triangleArray.numSbtRecords = 1; // one shared record; per-triangle data via primitive index

      triangle_handle = buildSingleGAS(build_input, g_state.d_triangle_gas_output);

      HitGroupRecord &rec = hitgroup_records[num_geoms];
      OPTIX_CHECK(optixSbtRecordPackHeader(g_state.hitgroup_triangle_pg, &rec));
      rec.data = HitGroupData{};
      rec.data.material_idx = -1; // unused: triangles resolve their material from tri_data
   }

   // ── Instance acceleration structure over the two GASes ────────────────────
   {
      std::vector<OptixInstance> instances;
      const float identity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
      auto add_instance = [&](OptixTraversableHandle handle, unsigned int sbt_offset) {
         OptixInstance inst = {};
         memcpy(inst.transform, identity, sizeof(identity));
         inst.instanceId        = static_cast<unsigned int>(instances.size());
         inst.sbtOffset         = sbt_offset;
         inst.visibilityMask    = 255;
         inst.flags             = OPTIX_INSTANCE_FLAG_NONE;
         inst.traversableHandle = handle;
         instances.push_back(inst);
      };
      if (custom_handle)
         add_instance(custom_handle, 0);
      if (triangle_handle)
         add_instance(triangle_handle, static_cast<unsigned int>(num_geoms));

      CUdeviceptr d_instances;
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_instances), instances.size() * sizeof(OptixInstance)));
      CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_instances), instances.data(),
                             instances.size() * sizeof(OptixInstance), cudaMemcpyHostToDevice));

      OptixBuildInput build_input = {};
      build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
      build_input.instanceArray.instances = d_instances;
      build_input.instanceArray.numInstances = static_cast<unsigned int>(instances.size());

      OptixAccelBuildOptions accel_options = {};
      accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
      accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

      OptixAccelBufferSizes ias_sizes;
      OPTIX_CHECK(optixAccelComputeMemoryUsage(g_state.context, &accel_options, &build_input, 1, &ias_sizes));

      CUdeviceptr d_temp;
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp), ias_sizes.tempSizeInBytes));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_state.d_ias_output), ias_sizes.outputSizeInBytes));

      OPTIX_CHECK(optixAccelBuild(g_state.context, 0, &accel_options, &build_input, 1, d_temp,
                                   ias_sizes.tempSizeInBytes, g_state.d_ias_output, ias_sizes.outputSizeInBytes,
                                   &g_state.gas_handle, nullptr, 0));
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp)));
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_instances)));
   }

   // ── Upload hit group SBT ──────────────────────────────────────────────────
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_state.d_sbt_hitgroup), num_hit_records * sizeof(HitGroupRecord)));
   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(g_state.d_sbt_hitgroup), hitgroup_records.data(),
                          num_hit_records * sizeof(HitGroupRecord), cudaMemcpyHostToDevice));

   g_state.sbt.hitgroupRecordBase = g_state.d_sbt_hitgroup;
   g_state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
   g_state.sbt.hitgroupRecordCount = num_hit_records;

   // Upload materials (release the volume SDF textures of the previous scene first)
   for (auto &hm : g_state.host_materials)
      freeVolumeSDF(hm.volume);
   g_state.host_materials.clear();
   if (g_state.d_materials)
   {
      CUDA_CHECK(cudaFree(g_state.d_materials));
      g_state.d_materials = nullptr;
   }

   int num_materials = static_cast<int>(scene.materials.size());
   if (num_materials > 0)
   {
      std::vector<OptixMaterialData> materials(num_materials);
      std::vector<float3> pattern_points; // FIBONACCI_DOTS directions, all materials concatenated
      g_state.has_volume    = false;
      g_state.has_thin_film = false;
      const int num_volume_materials = static_cast<int>(
          std::count_if(scene.materials.begin(), scene.materials.end(),
                        [](const Scene::MaterialDesc &m) { return m.type == Scene::MaterialType::VOLUME; }));
      for (int i = 0; i < num_materials; ++i)
      {
         const auto &m = scene.materials[i];
         if (m.type == Scene::MaterialType::VOLUME)
            g_state.has_volume = true;
         if (m.type == Scene::MaterialType::THIN_FILM)
            g_state.has_thin_film = true;
         materials[i].pattern_table_offset = 0;
         if (static_cast<int>(m.pattern) == 1) // FIBONACCI_DOTS: precompute the dot directions once
         {
            const int n = std::max(1, static_cast<int>(m.pattern_param1));
            materials[i].pattern_table_offset = static_cast<int>(pattern_points.size());
            for (int k = 0; k < n; ++k)
               pattern_points.push_back(fibonacciPoint(k, n));
         }
         materials[i].type = static_cast<OptixMaterialType>(static_cast<uint8_t>(m.type));
         materials[i].albedo = make_float3(static_cast<float>(m.albedo.x()), static_cast<float>(m.albedo.y()),
                                            static_cast<float>(m.albedo.z()));
         materials[i].emission = make_float3(static_cast<float>(m.emission.x()), static_cast<float>(m.emission.y()),
                                              static_cast<float>(m.emission.z()));
         materials[i].roughness = m.roughness;
         materials[i].refractive_index = m.refractive_index;
         materials[i].pattern = static_cast<unsigned char>(m.pattern);
         materials[i].pattern_color = make_float3(static_cast<float>(m.pattern_color.x()),
                                                   static_cast<float>(m.pattern_color.y()),
                                                   static_cast<float>(m.pattern_color.z()));
         materials[i].pattern_param1 = m.pattern_param1;
         materials[i].pattern_param2 = m.pattern_param2;
         materials[i].anisotropy = m.anisotropy;
         materials[i].eta = make_float3(static_cast<float>(m.eta.x()), static_cast<float>(m.eta.y()),
                                         static_cast<float>(m.eta.z()));
         materials[i].k = make_float3(static_cast<float>(m.k.x()), static_cast<float>(m.k.y()),
                                       static_cast<float>(m.k.z()));
         materials[i].film_thickness = m.film_thickness;
         materials[i].film_ior = m.film_ior;
         materials[i].volume.density        = m.density;
         materials[i].volume.noise_scale    = m.noise_scale;
         materials[i].volume.noise_strength = m.noise_strength;
         materials[i].volume.edge           = m.edge;
         materials[i].volume.lumps        = m.lumps;
         materials[i].volume.billow         = m.billow;
         materials[i].volume.sdf            = 0;
         materials[i].volume.sdf_array      = nullptr;
         materials[i].volume.field          = 0;
         materials[i].volume.field_array    = nullptr;
         materials[i].volume.field_dims     = make_int3(0, 0, 0);
         auto vb = volume_boxes.find(i);
         if (vb != volume_boxes.end())
         {
            // The analytic box of the NEE shortcut is exactly the traced one; only valid when this
            // is the scene's only medium (another box could overlap it)
            materials[i].volume.box_lo = vb->second.lo;
            materials[i].volume.box_hi = vb->second.hi;
            materials[i].volume.boxed  = (num_volume_materials == 1) ? 1 : 0;
            // Bind the distance grid built with the box geometry above (or the data grid)
            materials[i].volume.sdf            = vb->second.vp.sdf;
            materials[i].volume.sdf_array      = vb->second.vp.sdf_array;
            materials[i].volume.sdf_min        = vb->second.vp.sdf_min;
            materials[i].volume.sdf_inv_extent = vb->second.vp.sdf_inv_extent;
            materials[i].volume.field          = vb->second.vp.field;
            materials[i].volume.field_array    = vb->second.vp.field_array;
            materials[i].volume.field_dims     = vb->second.vp.field_dims;
            if (materials[i].volume.sdf != 0)
               bakeVolumeField(materials[i].volume, VOLUME_FIELD_CELLS);
         }
         materials[i].texture_id = m.texture_id;
      }
      CUDA_CHECK(cudaMalloc(&g_state.d_materials, num_materials * sizeof(OptixMaterialData)));
      CUDA_CHECK(cudaMemcpy(g_state.d_materials, materials.data(), num_materials * sizeof(OptixMaterialData),
                             cudaMemcpyHostToDevice));
      g_state.host_materials = materials;

      if (g_state.d_pattern_points)
      {
         CUDA_CHECK(cudaFree(g_state.d_pattern_points));
         g_state.d_pattern_points = nullptr;
      }
      if (!pattern_points.empty())
      {
         CUDA_CHECK(cudaMalloc(&g_state.d_pattern_points, pattern_points.size() * sizeof(float3)));
         CUDA_CHECK(cudaMemcpy(g_state.d_pattern_points, pattern_points.data(),
                                pattern_points.size() * sizeof(float3), cudaMemcpyHostToDevice));
      }
   }

   printf("OptiX GAS built: %d custom primitives, %d hardware triangles, %d materials\n", num_geoms, num_tris,
          num_materials);

   // Build CUDA texture objects from SceneDescription textures
   if (g_state.d_textures)
   {
      std::vector<cudaTextureObject_t> old_texs(static_cast<size_t>(g_state.num_textures));
      CUDA_CHECK(cudaMemcpy(old_texs.data(), g_state.d_textures,
                             g_state.num_textures * sizeof(cudaTextureObject_t), cudaMemcpyDeviceToHost));
      for (int i = 0; i < g_state.num_textures; ++i)
      {
         if (old_texs[i])
         {
            cudaResourceDesc rd = {};
            cudaGetTextureObjectResourceDesc(&rd, old_texs[i]);
            cudaDestroyTextureObject(old_texs[i]);
            if (rd.resType == cudaResourceTypeArray && rd.res.array.array)
               cudaFreeArray(rd.res.array.array);
         }
      }
      CUDA_CHECK(cudaFree(g_state.d_textures));
      g_state.d_textures = nullptr;
      g_state.num_textures = 0;
   }

   g_state.num_textures = static_cast<int>(scene.textures.size());
   if (g_state.num_textures > 0)
   {
      std::vector<cudaTextureObject_t> host_tex_objs(static_cast<size_t>(g_state.num_textures), 0);
      for (int ti = 0; ti < g_state.num_textures; ++ti)
      {
         const Scene::TextureDesc &td = scene.textures[ti];
         if (td.data.empty() || td.width <= 0 || td.height <= 0)
            continue;
         cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float4>();
         cudaArray_t cu_array = nullptr;
         CUDA_CHECK(cudaMallocArray(&cu_array, &fmt, static_cast<size_t>(td.width), static_cast<size_t>(td.height)));
         std::vector<float4> float_pixels(static_cast<size_t>(td.width) * static_cast<size_t>(td.height));
         for (int p = 0; p < td.width * td.height; ++p)
         {
            const uint8_t *px = td.data.data() + p * 4;
            float_pixels[p] = make_float4(px[0] / 255.0f, px[1] / 255.0f, px[2] / 255.0f, px[3] / 255.0f);
         }
         CUDA_CHECK(cudaMemcpy2DToArray(cu_array, 0, 0, float_pixels.data(),
                                         static_cast<size_t>(td.width) * sizeof(float4),
                                         static_cast<size_t>(td.width) * sizeof(float4),
                                         static_cast<size_t>(td.height), cudaMemcpyHostToDevice));
         cudaResourceDesc res_desc = {};
         res_desc.resType = cudaResourceTypeArray;
         res_desc.res.array.array = cu_array;
         cudaTextureDesc tex_desc = {};
         tex_desc.addressMode[0] = cudaAddressModeClamp;
         tex_desc.addressMode[1] = cudaAddressModeClamp;
         tex_desc.filterMode = cudaFilterModeLinear;
         tex_desc.readMode = cudaReadModeElementType;
         tex_desc.normalizedCoords = 1;
         CUDA_CHECK(cudaCreateTextureObject(&host_tex_objs[ti], &res_desc, &tex_desc, nullptr));
         printf("OptiX Texture[%d]: '%s' (%dx%d)\n", ti, td.path.c_str(), td.width, td.height);
      }
      CUDA_CHECK(cudaMalloc(&g_state.d_textures, g_state.num_textures * sizeof(cudaTextureObject_t)));
      CUDA_CHECK(cudaMemcpy(g_state.d_textures, host_tex_objs.data(),
                             g_state.num_textures * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));
   }

   // ── Build light list for NEE / MIS ──────────────────────────────────────
   if (g_state.d_lights)        { CUDA_CHECK(cudaFree(g_state.d_lights));        g_state.d_lights = nullptr; }
   if (g_state.d_light_cdfs)    { CUDA_CHECK(cudaFree(g_state.d_light_cdfs));    g_state.d_light_cdfs = nullptr; }
   if (g_state.d_geom_to_light) { CUDA_CHECK(cudaFree(g_state.d_geom_to_light)); g_state.d_geom_to_light = nullptr; }
   g_state.num_lights = 0;
   g_state.num_geom_entries = num_geoms;

   {
      std::vector<OptixLightData> host_lights;
      std::vector<float> host_areas;
      std::vector<float> host_power;
      // Map: SBT index → light list index (-1 if not a light)
      std::vector<int> geom_to_light(num_geoms, -1);

      for (int sbt_idx = 0; sbt_idx < num_geoms; ++sbt_idx)
      {
         int orig_idx = supported_indices[sbt_idx];
         const auto &g = scene.geometries[orig_idx];
         int mat_id = g.material_id;
         if (mat_id < 0 || mat_id >= static_cast<int>(scene.materials.size())) continue;
         if (scene.materials[mat_id].type != Scene::MaterialType::LIGHT) continue;

         const auto &mat = scene.materials[mat_id];
         OptixLightData ld = {};
         ld.emission = make_float3(static_cast<float>(mat.emission.x()),
                                    static_cast<float>(mat.emission.y()),
                                    static_cast<float>(mat.emission.z()));

         bool valid = false;
         if (g.type == Scene::GeometryType::RECTANGLE)
         {
            ld.geom_type = OptixLightGeomType::RECTANGLE;
            ld.corner = make_float3(static_cast<float>(g.data.rectangle.corner.x()),
                                     static_cast<float>(g.data.rectangle.corner.y()),
                                     static_cast<float>(g.data.rectangle.corner.z()));
            float3 u = make_float3(static_cast<float>(g.data.rectangle.u.x()),
                                    static_cast<float>(g.data.rectangle.u.y()),
                                    static_cast<float>(g.data.rectangle.u.z()));
            float3 v = make_float3(static_cast<float>(g.data.rectangle.v.x()),
                                    static_cast<float>(g.data.rectangle.v.y()),
                                    static_cast<float>(g.data.rectangle.v.z()));
            ld.u_vec = u;
            ld.v_vec = v;
            float3 n = make_float3(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x);
            float area = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
            float inv = (area > 1e-8f) ? 1.0f / area : 0.0f;
            ld.normal = make_float3(n.x * inv, n.y * inv, n.z * inv);
            ld.area   = area;
            valid = true;
            host_areas.push_back(area);
         }
         else if (g.type == Scene::GeometryType::SPHERE)
         {
            ld.geom_type = OptixLightGeomType::SPHERE;
            ld.center = make_float3(static_cast<float>(g.data.sphere.center.x()),
                                     static_cast<float>(g.data.sphere.center.y()),
                                     static_cast<float>(g.data.sphere.center.z()));
            ld.radius = static_cast<float>(g.data.sphere.radius);
            float area = 4.0f * CUDART_PI_F * ld.radius * ld.radius;
            ld.area   = area;
            valid = true;
            host_areas.push_back(area);
         }

         if (valid)
         {
            // Select lights by emitted power (area x luminance) rather than area alone: a small
            // bright lamp next to a large dim panel gets the samples it deserves
            const float lum = 0.2126f * ld.emission.x + 0.7152f * ld.emission.y + 0.0722f * ld.emission.z;
            host_power.push_back(host_areas.back() * std::max(lum, 0.0f));
            geom_to_light[sbt_idx] = static_cast<int>(host_lights.size());
            host_lights.push_back(ld);
         }
      }

      g_state.num_lights = static_cast<int>(host_lights.size());

      // Always upload geom_to_light map (even if no lights — avoids null pointer)
      if (num_geoms > 0)
      {
         CUDA_CHECK(cudaMalloc(&g_state.d_geom_to_light, num_geoms * sizeof(int)));
         CUDA_CHECK(cudaMemcpy(g_state.d_geom_to_light, geom_to_light.data(),
                                num_geoms * sizeof(int), cudaMemcpyHostToDevice));
      }

      if (g_state.num_lights > 0)
      {
         // Build CDF for power-weighted light selection (area-weighted if every light is black)
         double total_power = 0.0;
         for (float p : host_power) total_power += p;
         const std::vector<float> &weights = total_power > 0.0 ? host_power : host_areas;
         std::vector<float> cdf(g_state.num_lights + 1, 0.0f);
         for (int i = 0; i < g_state.num_lights; ++i)
            cdf[i + 1] = cdf[i] + weights[i];
         float total_weight = cdf.back();
         if (total_weight > 0.0f)
            for (auto &v : cdf) v /= total_weight;
         cdf.back() = 1.0f; // ensure last entry is exactly 1

         CUDA_CHECK(cudaMalloc(&g_state.d_lights, g_state.num_lights * sizeof(OptixLightData)));
         CUDA_CHECK(cudaMemcpy(g_state.d_lights, host_lights.data(),
                                g_state.num_lights * sizeof(OptixLightData), cudaMemcpyHostToDevice));

         CUDA_CHECK(cudaMalloc(&g_state.d_light_cdfs, (g_state.num_lights + 1) * sizeof(float)));
         CUDA_CHECK(cudaMemcpy(g_state.d_light_cdfs, cdf.data(),
                                (g_state.num_lights + 1) * sizeof(float), cudaMemcpyHostToDevice));

         printf("OptiX NEE: %d lights uploaded (selected by %s)\n", g_state.num_lights, total_power > 0.0 ? "power" : "area");
      }
   }

   uploadCausticTargets();

   // All uploads above use the legacy default stream. Pageable cudaMemcpy returns before the DMA
   // has necessarily landed, and the render stream is non-blocking (it does not synchronise with
   // the legacy stream), so fence here before the first launch can read any of these buffers.
   CUDA_CHECK(cudaDeviceSynchronize());
} // end buildGAS()

//==============================================================================
// PUBLIC INTERFACE (extern "C" for host renderer)
//==============================================================================

/// Runtime sampler toggle for OptiX — mirrors the CUDA renderer's g_use_sobol.
static bool g_optix_use_sobol = true;

extern "C" void setOptiXSobolSampler(bool use_sobol) { g_optix_use_sobol = use_sobol; }

extern "C" void setOptiXMISEnabled(bool enabled)          { g_state.mis_enabled = enabled; }
/// > 0: fixed per-sample luminance ceiling (camera moving: each frame is displayed on its own);
/// 0: the ceiling grows with the sample count.
static float g_optix_firefly_clamp_ceiling = 0.0f;
static float3 g_optix_background_color = make_float3(0.5f, 0.7f, 1.0f);
extern "C" void setOptiXBackgroundColor(float r, float g, float b) { g_optix_background_color = make_float3(r, g, b); }
extern "C" void setOptiXFireflyClampCeiling(float ceiling) { g_optix_firefly_clamp_ceiling = ceiling; }
extern "C" void setOptiXNEEStride(int stride)              { g_state.nee_stride = (stride < 1) ? 1 : stride; }
extern "C" void setOptiXLightFalloff(float falloff)        { g_state.light_falloff = (falloff < 0.0f) ? 0.0f : falloff; }
extern "C" void setOptiXIndirectClamp(float ceiling)      { g_state.indirect_clamp = (ceiling < 0.0f) ? 0.0f : ceiling; }
extern "C" void setOptiXCaustics(bool enabled)             { g_state.caustics_enabled = enabled; }

/// Converged-pixel skipping. Buffers are (re)allocated by optixRendererResetAccum().
extern "C" void setOptiXAdaptiveSampling(bool enabled, int min_samples, float threshold)
{
   g_state.adaptive_enabled     = enabled;
   g_state.adaptive_min_samples = std::max(1, min_samples);
   g_state.adaptive_threshold   = threshold;
}

extern "C" void optixRendererInit() { initializeOptiX(); }

// Rewrite one material in place (GUI edits, live material swaps) without rebuilding the scene;
// the caller restarts the accumulation. The type may change, but the FIBONACCI_DOTS table and
// the volume SDF are fixed at build time: callers keep such materials out of live swaps.
extern "C" void optixRendererUpdateMaterial(int idx, const Scene::MaterialDesc &m)
{
   if (!g_state.d_materials || idx < 0 || idx >= (int)g_state.host_materials.size())
      return;
   OptixMaterialData &d = g_state.host_materials[idx];
   d.type     = static_cast<OptixMaterialType>(static_cast<uint8_t>(m.type));
   d.pattern  = static_cast<unsigned char>(m.pattern);
   d.pattern_color  = make_float3((float)m.pattern_color.x(), (float)m.pattern_color.y(), (float)m.pattern_color.z());
   d.pattern_param1 = m.pattern_param1;
   d.pattern_param2 = m.pattern_param2;
   d.texture_id     = m.texture_id;
   d.albedo   = make_float3((float)m.albedo.x(), (float)m.albedo.y(), (float)m.albedo.z());
   d.emission = make_float3((float)m.emission.x(), (float)m.emission.y(), (float)m.emission.z());
   d.roughness        = m.roughness;
   d.refractive_index = m.refractive_index;
   d.anisotropy       = m.anisotropy;
   d.eta = make_float3((float)m.eta.x(), (float)m.eta.y(), (float)m.eta.z());
   d.k   = make_float3((float)m.k.x(), (float)m.k.y(), (float)m.k.z());
   d.film_thickness = m.film_thickness;
   d.film_ior       = m.film_ior;
   d.volume.density        = m.density; // the SDF binding is kept as uploaded
   d.volume.noise_scale    = m.noise_scale;
   d.volume.noise_strength = m.noise_strength;
   d.volume.edge           = m.edge;
   d.volume.lumps          = m.lumps;
   d.volume.billow         = m.billow;
   if (d.type == OptixMaterialType::VOLUME && d.volume.sdf != 0)
      bakeVolumeField(d.volume, VOLUME_FIELD_CELLS); // the parameters changed: rebake the field
   CUDA_CHECK(cudaMemcpyAsync(g_state.d_materials + idx, &d, sizeof(OptixMaterialData), cudaMemcpyHostToDevice,
                              getOptiXStream()));
   CUDA_CHECK(cudaStreamSynchronize(getOptiXStream()));
   uploadCausticTargets(); // the material may have become (or stopped being) a mirror or glass
}

extern "C" void optixRendererBuildScene(const Scene::SceneDescription &scene)
{
   if (!g_state.initialized)
   {
      printf("OptiX error: optixRendererBuildScene called before successful initialization; "
             "scene build skipped.\n");
      return;
   }

   buildGAS(scene);
}

// Reset (zero) the device accumulation buffer. Called once before the render loop.
// Allocates or resizes if needed, then zeros via cudaMemset — no host↔device transfer.
extern "C" void optixRendererResetAccum(int width, int height)
{
   if (!g_state.initialized)
      return;

   // Allocate/resize device accum buffer if needed
   if (g_state.d_accum_buffer == nullptr || g_state.accum_width != width || g_state.accum_height != height)
   {
      if (g_state.d_accum_buffer)
         CUDA_CHECK(cudaFree(g_state.d_accum_buffer));

      size_t accum_size = (size_t)width * height * sizeof(float4);
      CUDA_CHECK(cudaMalloc(&g_state.d_accum_buffer, accum_size));
      g_state.accum_width = width;
      g_state.accum_height = height;

      // Adaptive sampling buffers follow the accumulation buffer size
      if (g_state.d_pixel_sample_counts) CUDA_CHECK(cudaFree(g_state.d_pixel_sample_counts));
      if (g_state.d_lum_sq)              CUDA_CHECK(cudaFree(g_state.d_lum_sq));
      if (g_state.d_conv_streak)         CUDA_CHECK(cudaFree(g_state.d_conv_streak));
      CUDA_CHECK(cudaMalloc(&g_state.d_pixel_sample_counts, (size_t)width * height * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&g_state.d_lum_sq, (size_t)width * height * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&g_state.d_conv_streak, (size_t)width * height));
      if (g_state.d_prior) CUDA_CHECK(cudaFree(g_state.d_prior));
      CUDA_CHECK(cudaMalloc(&g_state.d_prior, (size_t)width * height * sizeof(float4)));
      if (g_state.d_resolved) CUDA_CHECK(cudaFree(g_state.d_resolved));
      CUDA_CHECK(cudaMalloc(&g_state.d_resolved, (size_t)width * height * sizeof(float4)));
      // Denoiser guides and output follow the accumulation buffer size
      float4 **denoise_bufs[] = {&g_state.d_albedo_accum, &g_state.d_normal_accum, &g_state.d_guide_albedo,
                                 &g_state.d_guide_normal, &g_state.d_denoised};
      for (float4 **buf : denoise_bufs)
      {
         if (*buf) CUDA_CHECK(cudaFree(*buf));
         CUDA_CHECK(cudaMalloc(buf, (size_t)width * height * sizeof(float4)));
      }
   }
   g_state.prior_active = false;

   // Zero the buffers on the render stream so the memsets are ordered before the next optixLaunch.
   // cudaMemset on the default stream (0) races with optixLaunch on the non-blocking render stream.
   const size_t n_pixels = (size_t)g_state.accum_width * g_state.accum_height;
   CUDA_CHECK(cudaMemsetAsync(g_state.d_accum_buffer, 0, n_pixels * sizeof(float4), getOptiXStream()));
   CUDA_CHECK(cudaMemsetAsync(g_state.d_pixel_sample_counts, 0, n_pixels * sizeof(int), getOptiXStream()));
   CUDA_CHECK(cudaMemsetAsync(g_state.d_lum_sq, 0, n_pixels * sizeof(float), getOptiXStream()));
   CUDA_CHECK(cudaMemsetAsync(g_state.d_conv_streak, 0, n_pixels, getOptiXStream()));
   CUDA_CHECK(cudaMemsetAsync(g_state.d_albedo_accum, 0, n_pixels * sizeof(float4), getOptiXStream()));
   CUDA_CHECK(cudaMemsetAsync(g_state.d_normal_accum, 0, n_pixels * sizeof(float4), getOptiXStream()));

   // Allocate persistent launch params buffer (once)
   if (g_state.d_launch_params == 0)
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_state.d_launch_params), sizeof(OptixLaunchParams)));
}

//==============================================================================
// Per-pixel sample bookkeeping, run after every launch: advance the per-pixel sample counts
// and, when adaptive sampling is enabled, mark pixels whose mean is known precisely enough
// (standard error in display space below `threshold`) as converged (negative count). The counts are
// kept accurate even while adaptive sampling is off, so it can be toggled mid-accumulation
// without restarting (the display always normalises by the per-pixel count).
// Mirrors the test in the CUDA renderAccKernel as a separate pass so the OptiX raygen
// program only has to read the per-pixel flag.
//==============================================================================
// Convergence test details, chosen so that pixels lit mostly by rare bright paths (caustics
// through glass, light arriving through soap bubbles) are not frozen at a wrong value:
//   * the standard error is tested only at checkpoints every ADAPTIVE_CHECK_INTERVAL samples,
//     and the pixel must pass ADAPTIVE_STREAK consecutive checkpoints after min_samples. (An
//     earlier version compared the running mean between checkpoints instead; that change
//     shrinks like 1/n whatever the noise, so the 10^-4.5 threshold was reached only after
//     ~10^5 samples on indirectly lit pixels and the image never reported 100 % converged.);
//   * a pixel that has received no energy at all is not considered converged before
//     ADAPTIVE_ZERO_SAMPLES, since a stable zero may just mean its contribution has not been
//     found yet. Two kinds of pixel are legitimately black for ever and must still converge:
//     one whose primary rays never hit geometry (accum.w == 0, maintained by raygen) only sees
//     the background and converges as soon as it is stable; one that sees geometry but no light
//     (a mirror or glass sphere reflecting a black sky, 3 % of default_scene_no_ambient) waits
//     for ADAPTIVE_ZERO_SAMPLES. A bright path found during a later re-check reopens it;
//   * once per ADAPTIVE_REFRESH_INTERVAL samples (of the whole image), each converged pixel is
//     re-checked for one checkpoint, so a late bright path can still correct it. The re-checks
//     are spread over the interval by pixel index and do not clear the converged flag.
constexpr int ADAPTIVE_CHECK_INTERVAL   = 16;
constexpr int ADAPTIVE_STREAK           = 2;
constexpr int ADAPTIVE_REFRESH_INTERVAL = 256;
constexpr int ADAPTIVE_ZERO_SAMPLES     = 512; // samples a lit-geometry pixel may stay at zero before converging

__global__ void optixAdaptiveUpdateKernel(const float4 *__restrict__ accum, int *pixel_sample_counts,
                                          const float *__restrict__ lum_sq, unsigned char *conv_streak,
                                          int num_pixels, int samples_added, int total_samples_after,
                                          bool adaptive, int min_samples, float threshold)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= num_pixels)
      return;
   const int count   = pixel_sample_counts[idx];
   unsigned char streak = conv_streak[idx];
   const bool rechecking = adaptive && count < 0 && streak == ADAPTIVE_RECHECK_MARK;

   if (adaptive && count < 0 && !rechecking)
   {
      // Converged: raygen skipped it. Once per ADAPTIVE_REFRESH_INTERVAL samples the pixel is
      // marked for a re-check, in a slot derived from its index so that only ~1/16 of the
      // converged pixels are re-checked at any checkpoint. The count keeps its sign meanwhile, so
      // the converged percentage stays steady and the per-launch cost stays smooth (reopening
      // every converged pixel at once made the readout fall to 0 % and the frame time spike every
      // 256 samples).
      constexpr int slots = ADAPTIVE_REFRESH_INTERVAL / ADAPTIVE_CHECK_INTERVAL;
      const int before      = total_samples_after - samples_added;
      const int slot_before = before / ADAPTIVE_CHECK_INTERVAL;
      const int slot_after  = total_samples_after / ADAPTIVE_CHECK_INTERVAL;
      const int crossed     = slot_after - slot_before; // checkpoints crossed by this launch
      const int phase       = (int)(((unsigned)idx * 2654435761u) >> 16) % slots;
      const int wait        = ((phase - slot_before - 1) % slots + slots) % slots; // slots until this pixel's turn
      if (crossed > 0 && (crossed >= slots || wait < crossed))
         conv_streak[idx] = ADAPTIVE_RECHECK_MARK;
      return;
   }

   // Open pixel, or converged pixel under re-check: it received samples_added samples.
   // (A pixel flagged converged earlier also lands here while adaptive sampling is off.)
   const int old_count = abs(count);
   const int new_count = old_count + samples_added;

   int result = rechecking ? -new_count : new_count;
   if (adaptive && old_count / ADAPTIVE_CHECK_INTERVAL != new_count / ADAPTIVE_CHECK_INTERVAL)
   {
      // Checkpoint: standard error of the pixel mean, from the running sums
      //   mean = sum(lum) / n,  var = (sum(lum^2) / n - mean^2) * n / (n - 1),  se = sqrt(var / n)
      // expressed in display space: the image is shown with gamma 2, so an error `se` on the
      // linear mean is an error of se / (2 sqrt(mean)) on the displayed value in [0, 1]. The
      // threshold is therefore a fraction of full scale (0.01 ≈ 2.5 levels of 255), which lets
      // dark pixels, whose noise the display cannot show, converge without a relative test
      // that silhouette-edge pixels (mostly background, rarely geometry) would never pass.
      const float4 acc  = accum[idx];
      const float  inv  = 1.0f / (float)new_count;
      const float  mean = (0.299f * acc.x + 0.587f * acc.y + 0.114f * acc.z) * inv;
      float        var  = fmaxf(lum_sq[idx] * inv - mean * mean, 0.0f);
      if (new_count > 1)
         var *= (float)new_count / (float)(new_count - 1);
      const float se_display = sqrtf(var * inv) / (2.0f * sqrtf(fmaxf(mean, 1e-4f)));
      // Background-only pixel: no primary ray ever hit geometry, so its value cannot change.
      const bool background_only = (acc.w == 0.0f);
      const bool stable = se_display < threshold &&
                          (mean > 0.0f || background_only || new_count >= ADAPTIVE_ZERO_SAMPLES);

      if (rechecking)
      {
         // Re-check done: still converged, or reopened because the new samples moved the estimate
         if (stable)
            streak = ADAPTIVE_STREAK;
         else
         {
            result = new_count;
            streak = 0;
         }
      }
      else
      {
         streak = stable ? (unsigned char)min(streak + 1, (int)ADAPTIVE_RECHECK_MARK - 1) : (unsigned char)0;
         if (new_count >= min_samples && streak >= ADAPTIVE_STREAK)
            result = -new_count; // sign bit = converged
      }
      conv_streak[idx] = streak;
   }
   pixel_sample_counts[idx] = result;
}

//==============================================================================
// Coarse-to-fine hand-over: when the camera stops, the reduced-resolution motion image is
// bilinearly upscaled into a separate "prior" image. The display blends it with the fresh
// full-resolution accumulation using a weight that starts at `strength` samples and fades
// linearly to zero once that many real samples have arrived, so the picture refines smoothly
// instead of snapping to noise. The accumulation itself is never touched by the prior.
//==============================================================================
__global__ void optixHandoverPriorKernel(const float4 *__restrict__ src, const int *__restrict__ src_counts,
                                         int src_w, int src_h, float4 *prior, int dst_w, int dst_h)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   if (x >= dst_w || y >= dst_h)
      return;

   // Bilinear sample of the source at the destination pixel centre
   float fx = ((float)x + 0.5f) * (float)src_w / (float)dst_w - 0.5f;
   float fy = ((float)y + 0.5f) * (float)src_h / (float)dst_h - 0.5f;
   fx = fminf(fmaxf(fx, 0.0f), (float)(src_w - 1));
   fy = fminf(fmaxf(fy, 0.0f), (float)(src_h - 1));
   int x0 = (int)fx, y0 = (int)fy;
   int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
   float tx = fx - (float)x0, ty = fy - (float)y0;

   // Source texels are sums; divide by their own sample counts to get means
   int i00 = y0 * src_w + x0, i10 = y0 * src_w + x1, i01 = y1 * src_w + x0, i11 = y1 * src_w + x1;
   float n00 = (float)max(abs(src_counts[i00]), 1), n10 = (float)max(abs(src_counts[i10]), 1);
   float n01 = (float)max(abs(src_counts[i01]), 1), n11 = (float)max(abs(src_counts[i11]), 1);
   float4 a = src[i00], b = src[i10], c = src[i01], d = src[i11];
   float w00 = (1.0f - tx) * (1.0f - ty) / n00, w10 = tx * (1.0f - ty) / n10;
   float w01 = (1.0f - tx) * ty / n01, w11 = tx * ty / n11;
   prior[y * dst_w + x] = make_float4(a.x * w00 + b.x * w10 + c.x * w01 + d.x * w11,
                                      a.y * w00 + b.y * w10 + c.y * w01 + d.y * w11,
                                      a.z * w00 + b.z * w10 + c.z * w01 + d.z * w11, 0.0f);
}

// Take the current accumulation (render_width x render_height) as the display prior for a new
// new_width x new_height pass, then clear the accumulation for that pass. `strength` is the
// prior's initial weight in samples; it fades out linearly as real samples arrive.
// Returns false (and leaves the caller to do a plain reset) if there is nothing to hand over.
extern "C" bool optixRendererHandoverPrior(int new_width, int new_height, float strength)
{
   const int src_w = g_state.render_width, src_h = g_state.render_height;
   if (!g_state.d_accum_buffer || !g_state.d_prior || src_w <= 0 || src_h <= 0 || strength <= 0.0f ||
       new_width > g_state.accum_width || new_height > g_state.accum_height)
      return false;

   cudaStream_t stream = getOptiXStream();
   dim3 threads(32, 8);
   dim3 blocks((new_width + threads.x - 1) / threads.x, (new_height + threads.y - 1) / threads.y);
   optixHandoverPriorKernel<<<blocks, threads, 0, stream>>>(g_state.d_accum_buffer, g_state.d_pixel_sample_counts,
                                                            src_w, src_h, g_state.d_prior, new_width, new_height);

   // Fresh accumulation for the new pass (same stream, so ordered after the prior kernel)
   const size_t n_pixels = (size_t)g_state.accum_width * g_state.accum_height;
   CUDA_CHECK(cudaMemsetAsync(g_state.d_accum_buffer, 0, n_pixels * sizeof(float4), stream));
   CUDA_CHECK(cudaMemsetAsync(g_state.d_pixel_sample_counts, 0, n_pixels * sizeof(int), stream));
   CUDA_CHECK(cudaMemsetAsync(g_state.d_lum_sq, 0, n_pixels * sizeof(float), stream));
   CUDA_CHECK(cudaMemsetAsync(g_state.d_conv_streak, 0, n_pixels, stream));
   CUDA_CHECK(cudaMemsetAsync(g_state.d_albedo_accum, 0, n_pixels * sizeof(float4), stream));
   CUDA_CHECK(cudaMemsetAsync(g_state.d_normal_accum, 0, n_pixels * sizeof(float4), stream));

   g_state.render_width   = new_width;
   g_state.render_height  = new_height;
   g_state.prior_active   = true;
   g_state.prior_strength = strength;
   return true;
}

// Launch a batch of samples (one thread per pixel, looping over the batch).
// Accumulation buffer stays on device — no host transfer per batch.
// Uses persistent device launch params buffer to avoid cudaMalloc/cudaFree overhead per batch.
extern "C" unsigned long long optixRendererLaunch(int width, int height, int num_materials, int samples_to_add,
                                                   int total_samples_so_far, int max_depth, float cam_cx, float cam_cy,
                                                   float cam_cz, float p00x, float p00y, float p00z, float dux,
                                                   float duy, float duz, float dvx, float dvy, float dvz,
                                                   float cam_ux, float cam_uy, float cam_uz, float cam_vx,
                                                   float cam_vy, float cam_vz, float bg_intensity, bool dof_enabled,
                                                   float dof_aperture, float dof_focus_dist,
                                                   float light_intensity, float metal_fuzziness,
                                                   float glass_ior_multiplier)
{
   if (!g_state.initialized || samples_to_add <= 0)
      return 0;
   if (width > g_state.accum_width || height > g_state.accum_height)
   {
      printf("OptiX error: launch %dx%d exceeds accumulation buffer %dx%d (call optixRendererResetAccum first)\n",
             width, height, g_state.accum_width, g_state.accum_height);
      return 0;
   }
   g_state.render_width  = width;
   g_state.render_height = height;

   // Fill launch params on host stack, then single cudaMemcpy to persistent device buffer
   OptixLaunchParams launch_params = {};
   launch_params.accum_buffer = g_state.d_accum_buffer;
   launch_params.width = width;
   launch_params.height = height;
   launch_params.pixel_sample_counts = g_state.adaptive_enabled ? g_state.d_pixel_sample_counts : nullptr;
   launch_params.lum_sq_buffer = g_state.d_lum_sq;
   launch_params.albedo_buffer = g_state.d_albedo_accum;
   launch_params.normal_buffer = g_state.d_normal_accum;
   launch_params.conv_streak = g_state.adaptive_enabled ? g_state.d_conv_streak : nullptr;
   launch_params.camera_center = make_float3(cam_cx, cam_cy, cam_cz);
   launch_params.pixel00_loc = make_float3(p00x, p00y, p00z);
   launch_params.pixel_delta_u = make_float3(dux, duy, duz);
   launch_params.pixel_delta_v = make_float3(dvx, dvy, dvz);
   launch_params.cam_u = make_float3(cam_ux, cam_uy, cam_uz);
   launch_params.cam_v = make_float3(cam_vx, cam_vy, cam_vz);
   {
      // w = u x v (the camera basis is right-handed, w points towards the viewer)
      float wx = cam_uy * cam_vz - cam_uz * cam_vy, wy = cam_uz * cam_vx - cam_ux * cam_vz,
            wz = cam_ux * cam_vy - cam_uy * cam_vx;
      float len = sqrtf(wx * wx + wy * wy + wz * wz);
      len = len > 0.0f ? 1.0f / len : 0.0f;
      launch_params.cam_w = make_float3(wx * len, wy * len, wz * len);
   }
   launch_params.samples_per_launch = samples_to_add;
   launch_params.total_samples_so_far = total_samples_so_far;
   launch_params.max_depth = max_depth;
   // Fresh Sobol scramble per accumulation (constant across its batches to keep stratification)
   static unsigned int sobol_sequence_seed = 0u;
   if (total_samples_so_far == 0)
      ++sobol_sequence_seed;
   launch_params.sobol_sequence_seed = sobol_sequence_seed;
   launch_params.use_sobol  = g_optix_use_sobol;
   launch_params.firefly_clamp_ceiling = g_optix_firefly_clamp_ceiling;
   launch_params.materials = g_state.d_materials;
   launch_params.num_materials = num_materials;
   launch_params.d_textures = g_state.d_textures;
   launch_params.num_textures = g_state.num_textures;
   launch_params.traversable = g_state.gas_handle;
   launch_params.background_intensity = bg_intensity;
   launch_params.background_color     = g_optix_background_color;
   launch_params.dof_enabled = dof_enabled;
   launch_params.dof_aperture = dof_aperture;
   launch_params.dof_focus_distance = dof_focus_dist;
   launch_params.light_intensity = light_intensity;
   launch_params.metal_fuzziness = metal_fuzziness;
   launch_params.glass_ior_multiplier = glass_ior_multiplier;
   launch_params.golf_dimple_count  = g_state.golf_dimple_count;
   launch_params.golf_dimple_radius = g_state.golf_dimple_radius;
   launch_params.golf_dimple_depth  = g_state.golf_dimple_depth;
   uploadGolfPointTable();
   launch_params.golf_points        = g_state.d_golf_points;
   launch_params.pattern_points     = g_state.d_pattern_points;
   launch_params.hdr_env_tex        = g_state.hdr_tex_obj;
   launch_params.use_hdr_env        = (g_state.hdr_tex_obj != 0);
   launch_params.env_marginal_cdf    = g_state.d_env_marginal_cdf;
   launch_params.env_conditional_cdf = g_state.d_env_conditional_cdf;
   launch_params.env_w               = g_state.env_w;
   launch_params.env_h               = g_state.env_h;

   // NEE / MIS
   launch_params.lights             = g_state.d_lights;
   launch_params.num_lights         = g_state.num_lights;
   launch_params.light_cdfs         = g_state.d_light_cdfs;
   launch_params.mis_enabled        = g_state.mis_enabled;
   launch_params.nee_stride         = g_state.nee_stride;
   launch_params.light_falloff      = g_state.light_falloff;
   launch_params.indirect_clamp     = g_state.indirect_clamp;
   launch_params.has_volume         = g_state.has_volume;
   launch_params.has_thin_film      = g_state.has_thin_film;
   launch_params.geom_to_light_map  = g_state.d_geom_to_light;
   launch_params.num_geom_entries   = g_state.num_geom_entries;

   // Caustic pass: off with a thin lens (the splat assumes a pinhole camera), without lights or
   // delta targets. Light paths: a quarter of the camera samples, in one block per sample.
   const bool lens = dof_enabled && dof_aperture > 0.0f;
   launch_params.caustics_active     = g_state.caustics_enabled && !lens && g_state.num_lights > 0 &&
                                   g_state.num_caustic_targets > 0;
   launch_params.caustic_paths       = std::max(1u, (unsigned int)width * height / 4) * (unsigned int)samples_to_add;
   launch_params.caustic_targets     = g_state.d_caustic_targets;
   launch_params.num_caustic_targets = g_state.num_caustic_targets;

   // Hardware triangles
   launch_params.tri_vertices  = g_state.d_tri_vertices;
   launch_params.tri_data      = g_state.d_tri_data;
   launch_params.num_triangles = g_state.num_triangles;

   // Single memcpy to persistent device buffer — no malloc/free per batch.
   // Use the dedicated stream for async param upload + launch.
   cudaStream_t stream = getOptiXStream();
   CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(g_state.d_launch_params), &launch_params,
                               sizeof(OptixLaunchParams), cudaMemcpyHostToDevice, stream));

   OPTIX_CHECK(optixLaunch(g_state.pipeline, stream, g_state.d_launch_params, sizeof(OptixLaunchParams), &g_state.sbt,
                            width, height, 1));
   if (launch_params.caustics_active)
   {
      // Light paths of the caustic pass, splatted into the same accumulation buffer before the
      // per-pixel counts advance (same stream: ordered after the camera pass)
      OptixShaderBindingTable sbt_caustics = g_state.sbt;
      sbt_caustics.raygenRecord            = g_state.d_sbt_raygen_caustics;
      OPTIX_CHECK(optixLaunch(g_state.pipeline, stream, g_state.d_launch_params, sizeof(OptixLaunchParams),
                              &sbt_caustics, launch_params.caustic_paths, 1, 1));
   }

   {
      const int num_pixels = width * height;
      const int threads = 256;
      optixAdaptiveUpdateKernel<<<(num_pixels + threads - 1) / threads, threads, 0, stream>>>(
          g_state.d_accum_buffer, g_state.d_pixel_sample_counts, g_state.d_lum_sq, g_state.d_conv_streak,
          num_pixels, samples_to_add, total_samples_so_far + samples_to_add, g_state.adaptive_enabled,
          g_state.adaptive_min_samples, g_state.adaptive_threshold);
   }

   // Stream-specific sync instead of cudaDeviceSynchronize() — only waits for
   // this stream, allowing other work (display pipeline) to proceed.
   CUDA_CHECK(cudaStreamSynchronize(stream));

   return (unsigned long long)width * height * samples_to_add;
}

// Download accumulated results from device to host. Called once after the entire render loop.
// Performs float4 → float3 conversion only at this final step.
extern "C" void optixRendererDownloadAccum(float *host_accum_buffer, int width, int height)
{
   if (!g_state.d_accum_buffer)
      return;

   const int num_pixels = width * height;
   const size_t accum_size = static_cast<size_t>(num_pixels) * sizeof(float4);

   // Reuse a persistent host buffer to avoid per-call malloc/free overhead.
   static std::vector<float4> host_f4;
   if (static_cast<int>(host_f4.size()) < num_pixels)
   {
      host_f4.resize(num_pixels);
   }

   CUDA_CHECK(cudaMemcpy(host_f4.data(), g_state.d_accum_buffer, accum_size, cudaMemcpyDeviceToHost));

   for (int i = 0; i < num_pixels; ++i)
   {
      host_accum_buffer[i * 3] = host_f4[i].x;
      host_accum_buffer[i * 3 + 1] = host_f4[i].y;
      host_accum_buffer[i * 3 + 2] = host_f4[i].z;
   }
}

extern "C" void optixRendererSetGolfDimples(int count, float radius, float depth)
{
   g_state.golf_dimple_count  = count;
   g_state.golf_dimple_radius = radius;
   g_state.golf_dimple_depth  = depth;
   // The direction table is refreshed lazily at the next launch (uploadGolfPointTable)
}

//==============================================================================
// GPU-side gamma correction for OptiX — mirrors the CUDA renderer's pipeline.
// Converts float4 accum buffer directly to uint8 display image on the GPU,
// then async-copies via pinned memory. Avoids the expensive float4 D2H transfer
// + host-side conversion that the original optixRendererDownloadAccum() used.
//==============================================================================
// The accumulation buffer may be smaller than the display (reduced resolution while the
// camera moves): each display pixel reads its nearest render pixel. With adaptive sampling
// each pixel is normalised by its own sample count.
// Mean radiance of one accumulated render pixel, with the hand-over prior blended in when active.
__device__ __forceinline__ float3 optixAccumPixelMean(const float4 *__restrict__ accum_buffer,
                                                      const int *__restrict__ pixel_sample_counts,
                                                      const float4 *__restrict__ prior, float prior_strength,
                                                      int idx, int num_samples)
{
   float4 acc = accum_buffer[idx];
   // Converged pixels store a negative count (sign bit = converged flag), so use abs().
   int samples = pixel_sample_counts ? abs(pixel_sample_counts[idx]) : num_samples;
   float n = (float)max(samples, 0);
   if (prior)
   {
      // Hand-over blend: the prior counts as max(0, strength - n) samples, so it dominates the
      // first frames after a resolution switch and is gone once `strength` samples have arrived.
      float w = fmaxf(prior_strength - n, 0.0f);
      float4 p = prior[idx];
      float inv = (n + w > 0.0f) ? 1.0f / (n + w) : 0.0f;
      return make_float3((acc.x + p.x * w) * inv, (acc.y + p.y * w) * inv, (acc.z + p.z * w) * inv);
   }
   float inv_samples = (n > 0.0f) ? 1.0f / n : 0.0f;
   return make_float3(acc.x * inv_samples, acc.y * inv_samples, acc.z * inv_samples);
}

__device__ __forceinline__ float optixLuminance601(float3 c) { return 0.299f * c.x + 0.587f * c.y + 0.114f * c.z; }

// Pass 1 (render resolution): mean radiance per pixel with the hand-over prior blended in and,
// for motion frames, the despeckle applied. Writes linear float4 so the display pass can upscale.
__global__ void optixResolveKernel(const float4 *__restrict__ accum_buffer,
                                   const int *__restrict__ pixel_sample_counts,
                                   const float4 *__restrict__ prior, float prior_strength, float4 *resolved,
                                   int render_width, int render_height, int num_samples, bool despeckle,
                                   float despeckle_ratio, float despeckle_floor,
                                   const float4 *__restrict__ albedo_accum, const float4 *__restrict__ normal_accum,
                                   float4 *guide_albedo, float4 *guide_normal)
{
   int rx = blockIdx.x * blockDim.x + threadIdx.x;
   int ry = blockIdx.y * blockDim.y + threadIdx.y;
   if (rx >= render_width || ry >= render_height)
      return;

   const int idx = ry * render_width + rx;
   float3 mean = optixAccumPixelMean(accum_buffer, pixel_sample_counts, prior, prior_strength, idx, num_samples);

   // Denoiser guides: per-pixel mean albedo and unit mean normal (zero where only the background
   // was seen). The hand-over prior is display-only colour, so it does not enter the guides.
   if (guide_albedo)
   {
      const int   samples = pixel_sample_counts ? abs(pixel_sample_counts[idx]) : num_samples;
      const float inv_n   = samples > 0 ? 1.0f / (float)samples : 0.0f;
      float4 a = albedo_accum[idx];
      guide_albedo[idx] = make_float4(a.x * inv_n, a.y * inv_n, a.z * inv_n, 0.0f);
      float4 n = normal_accum[idx];
      float  len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
      float  inv = len > 1e-6f ? 1.0f / len : 0.0f;
      guide_normal[idx] = make_float4(n.x * inv, n.y * inv, n.z * inv, 0.0f);
   }

   // Motion-frame despeckle: a frame shown on its own holds only a few samples per pixel, so one
   // sample of a few units is a bright dot. Scale a pixel down when it is far brighter than its
   // neighbourhood; genuinely bright regions have bright neighbours and are left alone.
   if (despeckle)
   {
      float lum = optixLuminance601(mean);
      float sum_nb = 0.0f;
      for (int dy = -1; dy <= 1; dy++)
         for (int dx = -1; dx <= 1; dx++)
         {
            if (dx == 0 && dy == 0)
               continue;
            int nx = min(max(rx + dx, 0), render_width - 1), ny = min(max(ry + dy, 0), render_height - 1);
            sum_nb += optixLuminance601(optixAccumPixelMean(accum_buffer, pixel_sample_counts, prior, prior_strength,
                                                            ny * render_width + nx, num_samples));
         }
      float ceiling = despeckle_ratio * (sum_nb * 0.125f) + despeckle_floor;
      if (lum > ceiling)
      {
         float s = ceiling / lum;
         mean = make_float3(mean.x * s, mean.y * s, mean.z * s);
      }
   }
   resolved[ry * render_width + rx] = make_float4(mean.x, mean.y, mean.z, 0.0f);
}

// Pass 2 (display resolution): gamma-encode to 8 bit. When the render resolution is lower than the
// display, sample the resolved image bilinearly so reduced-resolution frames look soft rather than
// blocky and the step to the next resolution stage is less visible.
__global__ void optixGammaCorrectKernel(const float4 *__restrict__ resolved, unsigned char *display_image,
                                         int render_width, int render_height, int width, int height,
                                         int channels, float gamma, int tone_map_op, float exposure)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   if (x >= width || y >= height)
      return;

   float3 mean;
   if (render_width == width && render_height == height)
   {
      float4 v = resolved[y * width + x];
      mean = make_float3(v.x, v.y, v.z);
   }
   else
   {
      float fx = ((float)x + 0.5f) * (float)render_width / (float)width - 0.5f;
      float fy = ((float)y + 0.5f) * (float)render_height / (float)height - 0.5f;
      fx = fminf(fmaxf(fx, 0.0f), (float)(render_width - 1));
      fy = fminf(fmaxf(fy, 0.0f), (float)(render_height - 1));
      int x0 = (int)fx, y0 = (int)fy;
      int x1 = min(x0 + 1, render_width - 1), y1 = min(y0 + 1, render_height - 1);
      float tx = fx - (float)x0, ty = fy - (float)y0;
      float4 a = resolved[y0 * render_width + x0], b = resolved[y0 * render_width + x1];
      float4 c = resolved[y1 * render_width + x0], d = resolved[y1 * render_width + x1];
      float w00 = (1.0f - tx) * (1.0f - ty), w10 = tx * (1.0f - ty), w01 = (1.0f - tx) * ty, w11 = tx * ty;
      mean = make_float3(a.x * w00 + b.x * w10 + c.x * w01 + d.x * w11, a.y * w00 + b.y * w10 + c.y * w01 + d.y * w11,
                         a.z * w00 + b.z * w10 + c.z * w01 + d.z * w11);
   }
   float inv_gamma = 1.0f / gamma;

   applyToneMap(mean.x, mean.y, mean.z, tone_map_op, exposure);
   float r = fminf(powf(fmaxf(mean.x, 0.0f), inv_gamma), 0.999f);
   float g = fminf(powf(fmaxf(mean.y, 0.0f), inv_gamma), 0.999f);
   float b = fminf(powf(fmaxf(mean.z, 0.0f), inv_gamma), 0.999f);

   int image_idx = (y * width + x) * channels;
   display_image[image_idx + 0] = (unsigned char)(256.0f * r);
   display_image[image_idx + 1] = (unsigned char)(256.0f * g);
   display_image[image_idx + 2] = (unsigned char)(256.0f * b);
   if (channels == 4)
      display_image[image_idx + 3] = 255;
}

// Allocate (or resize) the device display buffer and the pinned host staging buffer
static void ensureDisplayBuffers(size_t display_size)
{
   if (g_state.d_display == nullptr || g_state.d_display_size != display_size)
   {
      if (g_state.d_display != nullptr)
         cudaFree(g_state.d_display);
      cudaMalloc(&g_state.d_display, display_size);
      g_state.d_display_size = display_size;
   }
   if (g_state.pinned_display == nullptr || g_state.pinned_display_size != display_size)
   {
      if (g_state.pinned_display != nullptr)
         cudaFreeHost(g_state.pinned_display);
      cudaMallocHost(&g_state.pinned_display, display_size);
      g_state.pinned_display_size = display_size;
   }
}

// Convert the accumulation buffer (at the resolution of the last launch) into a width x height
// 8-bit display image. num_samples is the normalisation when adaptive sampling is off.
// despeckle: scale down pixels brighter than ratio x (mean of the 8 neighbours) + floor (motion frames).
//------------------------------------------------------------------------------
// OptiX AI denoiser. HDR model with albedo + normal guides; the normal guide is in camera space
// as that model expects. State and scratch are sized once for the accumulation buffer (the
// largest input), optixDenoiserSetup is redone whenever the render resolution changes.
//------------------------------------------------------------------------------
static bool ensureDenoiser(int w, int h)
{
   if (g_state.denoiser_failed || !g_state.context)
      return false;
   if (!g_state.denoiser)
   {
      OptixDenoiserOptions options = {};
      options.guideAlbedo  = 1;
      options.guideNormal  = 1;
      options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
      OptixResult res = optixDenoiserCreate(g_state.context, OPTIX_DENOISER_MODEL_KIND_HDR, &options, &g_state.denoiser);
      if (res != OPTIX_SUCCESS)
      {
         printf("OptiX denoiser unavailable: %s\n", optixGetErrorString(res));
         g_state.denoiser_failed = true;
         return false;
      }
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_state.d_denoiser_intensity), sizeof(float)));
      CUDA_CHECK(cudaEventCreate(&g_state.denoise_start));
      CUDA_CHECK(cudaEventCreate(&g_state.denoise_end));
   }
   // Memory for the largest input this accumulation can produce
   const int max_w = std::max(w, g_state.accum_width), max_h = std::max(h, g_state.accum_height);
   if (max_w > g_state.denoiser_max_w || max_h > g_state.denoiser_max_h)
   {
      OptixDenoiserSizes sizes = {};
      OptixResult res = optixDenoiserComputeMemoryResources(g_state.denoiser, max_w, max_h, &sizes);
      if (res != OPTIX_SUCCESS)
      {
         printf("OptiX denoiser: memory query failed: %s\n", optixGetErrorString(res));
         g_state.denoiser_failed = true;
         return false;
      }
      if (g_state.d_denoiser_state)   CUDA_CHECK(cudaFree(reinterpret_cast<void *>(g_state.d_denoiser_state)));
      if (g_state.d_denoiser_scratch) CUDA_CHECK(cudaFree(reinterpret_cast<void *>(g_state.d_denoiser_scratch)));
      g_state.denoiser_state_size   = sizes.stateSizeInBytes;
      g_state.denoiser_scratch_size = sizes.withoutOverlapScratchSizeInBytes;
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_state.d_denoiser_state), g_state.denoiser_state_size));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_state.d_denoiser_scratch), g_state.denoiser_scratch_size));
      g_state.denoiser_max_w   = max_w;
      g_state.denoiser_max_h   = max_h;
      g_state.denoiser_setup_w = g_state.denoiser_setup_h = 0;
   }
   if (w != g_state.denoiser_setup_w || h != g_state.denoiser_setup_h)
   {
      OptixResult res = optixDenoiserSetup(g_state.denoiser, getOptiXStream(), w, h, g_state.d_denoiser_state,
                                           g_state.denoiser_state_size, g_state.d_denoiser_scratch,
                                           g_state.denoiser_scratch_size);
      if (res != OPTIX_SUCCESS)
      {
         printf("OptiX denoiser: setup failed: %s\n", optixGetErrorString(res));
         g_state.denoiser_failed = true;
         return false;
      }
      g_state.denoiser_setup_w = w;
      g_state.denoiser_setup_h = h;
   }
   return true;
}

static OptixImage2D denoiserImage(float4 *data, int w, int h)
{
   OptixImage2D img = {};
   img.data               = reinterpret_cast<CUdeviceptr>(data);
   img.width              = (unsigned int)w;
   img.height             = (unsigned int)h;
   img.rowStrideInBytes   = (unsigned int)(w * sizeof(float4));
   img.pixelStrideInBytes = sizeof(float4);
   img.format             = OPTIX_PIXEL_FORMAT_FLOAT4;
   return img;
}

// Denoise d_resolved (w x h) into d_denoised. blend: 0 = fully denoised, 1 = untouched input.
static bool denoiseResolved(int w, int h, float blend, cudaStream_t stream)
{
   if (!ensureDenoiser(w, h))
      return false;
   cudaEventRecord(g_state.denoise_start, stream);
   OptixImage2D input = denoiserImage(g_state.d_resolved, w, h);
   OPTIX_CHECK(optixDenoiserComputeIntensity(g_state.denoiser, stream, &input, g_state.d_denoiser_intensity,
                                             g_state.d_denoiser_scratch, g_state.denoiser_scratch_size));
   OptixDenoiserParams params = {};
   params.hdrIntensity = g_state.d_denoiser_intensity;
   params.blendFactor  = blend;
   OptixDenoiserGuideLayer guide = {};
   guide.albedo = denoiserImage(g_state.d_guide_albedo, w, h);
   guide.normal = denoiserImage(g_state.d_guide_normal, w, h);
   OptixDenoiserLayer layer = {};
   layer.input  = input;
   layer.output = denoiserImage(g_state.d_denoised, w, h);
   OPTIX_CHECK(optixDenoiserInvoke(g_state.denoiser, stream, &params, g_state.d_denoiser_state,
                                   g_state.denoiser_state_size, &guide, &layer, 1, 0, 0, g_state.d_denoiser_scratch,
                                   g_state.denoiser_scratch_size));
   cudaEventRecord(g_state.denoise_end, stream);
   return true;
}

// Denoiser time of the last display conversion (0 when it did not denoise)
extern "C" float optixRendererLastDenoiseMs() { return g_state.last_denoise_ms; }

extern "C" void optixRendererConvertAccumToDisplay(unsigned char *display_image, int width, int height,
                                                    int channels, int num_samples, float gamma, bool despeckle,
                                                    float despeckle_ratio, float despeckle_floor, bool denoise,
                                                    float denoise_blend)
{
   g_state.last_denoise_ms = 0.0f;
   if (!g_state.d_accum_buffer || !display_image || num_samples <= 0)
      return;

   const int render_width  = g_state.render_width > 0 ? g_state.render_width : width;
   const int render_height = g_state.render_height > 0 ? g_state.render_height : height;
   size_t display_size = (size_t)width * height * channels * sizeof(unsigned char);
   ensureDisplayBuffers(display_size);

   dim3 threads(32, 8);
   dim3 rblocks((render_width + threads.x - 1) / threads.x, (render_height + threads.y - 1) / threads.y);
   dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

   cudaStream_t stream = getOptiXStream();

   // Per-pixel counts are maintained after every launch, so they are always the right normalisation
   const int *counts = g_state.d_pixel_sample_counts;
   const float4 *prior = g_state.prior_active ? g_state.d_prior : nullptr;
   optixResolveKernel<<<rblocks, threads, 0, stream>>>(g_state.d_accum_buffer, counts, prior, g_state.prior_strength,
                                                       g_state.d_resolved, render_width, render_height, num_samples,
                                                       despeckle, despeckle_ratio, despeckle_floor,
                                                       g_state.d_albedo_accum, g_state.d_normal_accum,
                                                       denoise ? g_state.d_guide_albedo : nullptr,
                                                       denoise ? g_state.d_guide_normal : nullptr);
   const float4 *display_src = g_state.d_resolved;
   const bool denoised = denoise && denoiseResolved(render_width, render_height, denoise_blend, stream);
   if (denoised)
      display_src = g_state.d_denoised;
   const ToneMapSettings tone_map = displayToneMap();
   optixGammaCorrectKernel<<<blocks, threads, 0, stream>>>(display_src, g_state.d_display, render_width,
                                                            render_height, width, height, channels, gamma,
                                                            tone_map.op, tone_map.exposure);

   // Async D2H copy via pinned memory, then single stream sync
   cudaMemcpyAsync(g_state.pinned_display, g_state.d_display, display_size, cudaMemcpyDeviceToHost, stream);
   cudaStreamSynchronize(stream);
   if (denoised)
      cudaEventElapsedTime(&g_state.last_denoise_ms, g_state.denoise_start, g_state.denoise_end);
   memcpy(display_image, g_state.pinned_display, display_size);
}

//==============================================================================
// Adaptive sampling diagnostics: converged-pixel count and sample-count heatmap
//==============================================================================
__global__ void optixCountConvergedKernel(const int *__restrict__ pixel_sample_counts, int num_pixels,
                                          int *d_converged_count)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int converged = (idx < num_pixels && pixel_sample_counts[idx] < 0) ? 1 : 0;
   for (int offset = 16; offset > 0; offset >>= 1)
      converged += __shfl_down_sync(0xFFFFFFFF, converged, offset);
   if ((threadIdx.x & 31) == 0 && converged > 0)
      atomicAdd(d_converged_count, converged);
}

// Returns the number of converged pixels at the resolution of the last launch (-1 if unavailable)
extern "C" int optixRendererCountConvergedPixels()
{
   if (!g_state.adaptive_enabled || !g_state.d_pixel_sample_counts || g_state.render_width <= 0)
      return -1;
   if (!g_state.d_converged_count)
      CUDA_CHECK(cudaMalloc(&g_state.d_converged_count, sizeof(int)));

   cudaStream_t stream = getOptiXStream();
   const int num_pixels = g_state.render_width * g_state.render_height;
   CUDA_CHECK(cudaMemsetAsync(g_state.d_converged_count, 0, sizeof(int), stream));
   const int threads = 256;
   optixCountConvergedKernel<<<(num_pixels + threads - 1) / threads, threads, 0, stream>>>(
       g_state.d_pixel_sample_counts, num_pixels, g_state.d_converged_count);
   int converged = 0;
   CUDA_CHECK(cudaMemcpyAsync(&converged, g_state.d_converged_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
   CUDA_CHECK(cudaStreamSynchronize(stream));
   return converged;
}

__device__ void plasmaColormap(float t, float &r, float &g, float &b); // shaders/render_acc_kernel.cu

__global__ void optixSampleHeatmapKernel(const int *__restrict__ pixel_sample_counts, unsigned char *display_image,
                                         int render_width, int render_height, int width, int height, int channels,
                                         int max_samples_for_scale)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   if (x >= width || y >= height)
      return;
   int rx = (render_width == width) ? x : min(x * render_width / width, render_width - 1);
   int ry = (render_height == height) ? y : min(y * render_height / height, render_height - 1);

   int sample_count = abs(pixel_sample_counts[ry * render_width + rx]);
   float t = (max_samples_for_scale > 0) ? fminf((float)sample_count / (float)max_samples_for_scale, 1.0f) : 0.0f;
   float r, g, b;
   plasmaColormap(t, r, g, b);

   int image_idx = (y * width + x) * channels;
   display_image[image_idx + 0] = (unsigned char)(255.0f * r);
   display_image[image_idx + 1] = (unsigned char)(255.0f * g);
   display_image[image_idx + 2] = (unsigned char)(255.0f * b);
   if (channels == 4)
      display_image[image_idx + 3] = 255;
}

// Replace the display image with the per-pixel sample-count heatmap (dark = converged early)
extern "C" void optixRendererRenderSampleHeatmap(unsigned char *display_image, int width, int height, int channels,
                                                 int max_samples_for_scale)
{
   if (!g_state.d_pixel_sample_counts || !display_image || g_state.render_width <= 0)
      return;
   size_t display_size = (size_t)width * height * channels * sizeof(unsigned char);
   ensureDisplayBuffers(display_size);

   dim3 threads(32, 8);
   dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
   cudaStream_t stream = getOptiXStream();
   optixSampleHeatmapKernel<<<blocks, threads, 0, stream>>>(g_state.d_pixel_sample_counts, g_state.d_display,
                                                             g_state.render_width, g_state.render_height, width,
                                                             height, channels, max_samples_for_scale);
   cudaMemcpyAsync(g_state.pinned_display, g_state.d_display, display_size, cudaMemcpyDeviceToHost, stream);
   cudaStreamSynchronize(stream);
   memcpy(display_image, g_state.pinned_display, display_size);
}

static void freeEnvSamplingTables()
{
   if (g_state.d_env_marginal_cdf)    { cudaFree(g_state.d_env_marginal_cdf);    g_state.d_env_marginal_cdf = nullptr; }
   if (g_state.d_env_conditional_cdf) { cudaFree(g_state.d_env_conditional_cdf); g_state.d_env_conditional_cdf = nullptr; }
   g_state.env_w = 0;
   g_state.env_h = 0;
}

/// Sky importance sampling: a 2D distribution proportional to luminance x sin(theta) over a
/// downsampled lat-long grid of the map (at most 2048 x 1024 cells, a 4 x 4 texel box on an 8k
/// map, widened by the bilinear filter's reach). The device picks a row from the marginal CDF and a cell from that row's conditional
/// CDF, so the sun of a clear-sky HDRI (a few hundred texels holding half the sky's energy) is
/// hit by next-event estimation instead of by chance. The CDFs are accumulated in double and
/// stored as float: the rounding is monotone, so a cell's probability is exactly the difference
/// of two consecutive entries, on the sampling side as on the pdf side.
/// Returns false (no tables, sky left to BSDF sampling) on a black map or a failed upload.
static bool buildEnvSamplingTables(const uint16_t *rgba16, int w, int h)
{
   freeEnvSamplingTables();
   const int W = std::min(w, 2048);
   const int H = std::min(h, 1024);

   // A texel's weight goes to every cell a lookup can read it from. The texture is sampled with
   // bilinear filtering, so a lookup at u reaches texels u*w - 0.5 +- 1: texel x is read for
   // u in ((x - 0.5) / w, (x + 1.5) / w), half a texel beyond its own cell on the low side and
   // a full one on the high side (wrap in u, clamp in v). Without this margin, the half-texel
   // band along the sun's edge returns a blend of sun and sky while the table says sky: the pdf
   // is tiny there, the MIS weight of a BSDF ray goes to 1, and the band is a firefly source.
   std::vector<double> cell(static_cast<size_t>(W) * H, 0.0);
   auto half_to_float = [](uint16_t bits) {
      __half hv;
      std::memcpy(&hv, &bits, sizeof(uint16_t));
      return __half2float(hv);
   };
   auto span = [](int i, int n_tex, int n_cell, int &lo, int &hi) {
      lo = static_cast<int>(std::floor((i - 0.5) * n_cell / n_tex));
      hi = static_cast<int>(std::floor((i + 1.5) * n_cell / n_tex - 1e-9));
   };
   for (int y = 0; y < h; ++y)
   {
      int cy0, cy1;
      span(y, h, H, cy0, cy1);
      cy0 = std::max(cy0, 0);
      cy1 = std::min(cy1, H - 1);
      const double sin_theta = std::sin((y + 0.5) / h * CUDART_PI);
      const uint16_t *row    = rgba16 + static_cast<size_t>(y) * w * 4u;
      for (int x = 0; x < w; ++x)
      {
         const double lum = 0.2126 * half_to_float(row[x * 4 + 0]) + 0.7152 * half_to_float(row[x * 4 + 1]) +
                            0.0722 * half_to_float(row[x * 4 + 2]);
         if (!(lum > 0.0))
            continue;
         int cx0, cx1;
         span(x, w, W, cx0, cx1);
         const double wgt = lum * sin_theta;
         for (int cy = cy0; cy <= cy1; ++cy)
            for (int cx = cx0; cx <= cx1; ++cx)
               cell[static_cast<size_t>(cy) * W + ((cx + W) % W)] += wgt;
      }
   }

   // Conditional CDF of every row (uniform ramp on an all-black row, which the marginal never picks)
   std::vector<float>  cond(static_cast<size_t>(H) * (W + 1));
   std::vector<double> row_sum(static_cast<size_t>(H), 0.0);
   for (int y = 0; y < H; ++y)
   {
      double sum = 0.0;
      for (int x = 0; x < W; ++x) sum += cell[static_cast<size_t>(y) * W + x];
      row_sum[y]  = sum;
      float *cdf  = cond.data() + static_cast<size_t>(y) * (W + 1);
      double acc  = 0.0;
      cdf[0]      = 0.0f;
      for (int x = 0; x < W; ++x)
      {
         acc += cell[static_cast<size_t>(y) * W + x];
         cdf[x + 1] = (sum > 0.0) ? static_cast<float>(acc / sum) : static_cast<float>(x + 1) / W;
      }
      cdf[W] = 1.0f;
   }
   double total = 0.0;
   for (double r : row_sum) total += r;
   if (!(total > 0.0))
      return false;
   std::vector<float> marg(static_cast<size_t>(H) + 1);
   marg[0] = 0.0f;
   double acc = 0.0;
   for (int y = 0; y < H; ++y)
   {
      acc += row_sum[y];
      marg[y + 1] = static_cast<float>(acc / total);
   }
   marg[H] = 1.0f;

   if (cudaMalloc(&g_state.d_env_marginal_cdf, marg.size() * sizeof(float)) != cudaSuccess ||
       cudaMalloc(&g_state.d_env_conditional_cdf, cond.size() * sizeof(float)) != cudaSuccess ||
       cudaMemcpy(g_state.d_env_marginal_cdf, marg.data(), marg.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
       cudaMemcpy(g_state.d_env_conditional_cdf, cond.data(), cond.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
   {
      fprintf(stderr, "OptiX HDR: sky sampling tables upload failed (sky left to BSDF sampling)\n");
      freeEnvSamplingTables();
      return false;
   }
   g_state.env_w = W;
   g_state.env_h = H;
   return true;
}

extern "C" void optixRendererClearHdrEnv()
{
   freeEnvSamplingTables();
   if (g_state.hdr_tex_obj != 0)
   {
      cudaDestroyTextureObject(g_state.hdr_tex_obj);
      g_state.hdr_tex_obj = 0;
   }
   if (g_state.hdr_cuda_array != nullptr)
   {
      cudaFreeArray(g_state.hdr_cuda_array);
      g_state.hdr_cuda_array = nullptr;
   }
}

extern "C" bool optixRendererUploadHdrEnvHalf(const uint16_t *rgba16, int w, int h)
{
   optixRendererClearHdrEnv();
   if (!rgba16 || w <= 0 || h <= 0)
      return false;

   // float16×4 channel format — half VRAM vs float4 for the same resolution
   cudaChannelFormatDesc channelDesc = {16, 16, 16, 16, cudaChannelFormatKindFloat};
   if (cudaMallocArray(&g_state.hdr_cuda_array, &channelDesc, (size_t)w, (size_t)h) != cudaSuccess)
   {
      fprintf(stderr, "OptiX HDR: cudaMallocArray failed (%dx%d)\n", w, h);
      g_state.hdr_cuda_array = nullptr;
      return false;
   }

   if (cudaMemcpy2DToArray(g_state.hdr_cuda_array, 0, 0, rgba16,
                           (size_t)w * 4 * sizeof(uint16_t), (size_t)w * 4 * sizeof(uint16_t), (size_t)h,
                           cudaMemcpyHostToDevice) != cudaSuccess)
   {
      fprintf(stderr, "OptiX HDR: cudaMemcpy2DToArray failed\n");
      cudaFreeArray(g_state.hdr_cuda_array);
      g_state.hdr_cuda_array = nullptr;
      return false;
   }

   cudaResourceDesc resDesc = {};
   resDesc.resType               = cudaResourceTypeArray;
   resDesc.res.array.array       = g_state.hdr_cuda_array;

   cudaTextureDesc texDesc = {};
   texDesc.addressMode[0]   = cudaAddressModeWrap;
   texDesc.addressMode[1]   = cudaAddressModeClamp;
   texDesc.filterMode       = cudaFilterModeLinear;
   texDesc.readMode         = cudaReadModeElementType;
   texDesc.normalizedCoords = 1;

   if (cudaCreateTextureObject(&g_state.hdr_tex_obj, &resDesc, &texDesc, nullptr) != cudaSuccess)
   {
      fprintf(stderr, "OptiX HDR: cudaCreateTextureObject failed\n");
      cudaFreeArray(g_state.hdr_cuda_array);
      g_state.hdr_cuda_array = nullptr;
      return false;
   }

   buildEnvSamplingTables(rgba16, w, h);

   CUDA_CHECK(cudaDeviceSynchronize()); // fence the array upload before the render stream reads it
   return true;
}

extern "C" bool optixRendererUploadHdrEnv(const float *rgba_data, int w, int h)
{
   if (!rgba_data || w <= 0 || h <= 0)
      return false;

   // Convert float4 → half4 using CUDA intrinsic (round-to-nearest)
   const size_t          npixels = static_cast<size_t>(w) * static_cast<size_t>(h);
   std::vector<uint16_t> rgba16(npixels * 4u);
   for (size_t i = 0; i < npixels * 4u; ++i)
   {
      __half h16 = __float2half_rn(rgba_data[i]);
      std::memcpy(&rgba16[i], &h16, sizeof(uint16_t));
   }

   return optixRendererUploadHdrEnvHalf(rgba16.data(), w, h);
}

extern "C" void optixRendererCleanup()
{
   optixRendererClearHdrEnv();
   if (g_state.d_accum_buffer)
      CUDA_CHECK(cudaFree(g_state.d_accum_buffer));
   if (g_state.d_pixel_sample_counts)
      CUDA_CHECK(cudaFree(g_state.d_pixel_sample_counts));
   if (g_state.d_lum_sq)
      CUDA_CHECK(cudaFree(g_state.d_lum_sq));
   if (g_state.d_conv_streak)
      CUDA_CHECK(cudaFree(g_state.d_conv_streak));
   if (g_state.d_converged_count)
      CUDA_CHECK(cudaFree(g_state.d_converged_count));
   if (g_state.d_prior)
      CUDA_CHECK(cudaFree(g_state.d_prior));
   if (g_state.d_resolved)
      CUDA_CHECK(cudaFree(g_state.d_resolved));
   for (float4 *buf : {g_state.d_albedo_accum, g_state.d_normal_accum, g_state.d_guide_albedo,
                       g_state.d_guide_normal, g_state.d_denoised})
      if (buf)
         CUDA_CHECK(cudaFree(buf));
   if (g_state.denoiser)
      OPTIX_CHECK(optixDenoiserDestroy(g_state.denoiser));
   if (g_state.d_denoiser_state)
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(g_state.d_denoiser_state)));
   if (g_state.d_denoiser_scratch)
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(g_state.d_denoiser_scratch)));
   if (g_state.d_denoiser_intensity)
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(g_state.d_denoiser_intensity)));
   if (g_state.denoise_start)
      CUDA_CHECK(cudaEventDestroy(g_state.denoise_start));
   if (g_state.denoise_end)
      CUDA_CHECK(cudaEventDestroy(g_state.denoise_end));
   if (g_state.d_golf_points)
      CUDA_CHECK(cudaFree(g_state.d_golf_points));
   if (g_state.d_pattern_points)
      CUDA_CHECK(cudaFree(g_state.d_pattern_points));
   if (g_state.d_launch_params)
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(g_state.d_launch_params)));
   for (auto &hm : g_state.host_materials)
      freeVolumeSDF(hm.volume);
   g_state.host_materials.clear();
   if (g_state.d_materials)
      CUDA_CHECK(cudaFree(g_state.d_materials));
   if (g_state.d_lights)
      CUDA_CHECK(cudaFree(g_state.d_lights));
   if (g_state.d_light_cdfs)
      CUDA_CHECK(cudaFree(g_state.d_light_cdfs));
   if (g_state.d_geom_to_light)
      CUDA_CHECK(cudaFree(g_state.d_geom_to_light));
   if (g_state.d_textures)
   {
      std::vector<cudaTextureObject_t> texs(static_cast<size_t>(g_state.num_textures));
      cudaMemcpy(texs.data(), g_state.d_textures, g_state.num_textures * sizeof(cudaTextureObject_t), cudaMemcpyDeviceToHost);
      for (int i = 0; i < g_state.num_textures; ++i)
      {
         if (texs[i])
         {
            cudaResourceDesc rd = {};
            cudaGetTextureObjectResourceDesc(&rd, texs[i]);
            cudaDestroyTextureObject(texs[i]);
            if (rd.resType == cudaResourceTypeArray && rd.res.array.array)
               cudaFreeArray(rd.res.array.array);
         }
      }
      CUDA_CHECK(cudaFree(g_state.d_textures));
   }
   if (g_state.d_sbt_raygen)
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(g_state.d_sbt_raygen)));
   if (g_state.d_sbt_raygen_caustics)
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(g_state.d_sbt_raygen_caustics)));
   if (g_state.d_caustic_targets)
      CUDA_CHECK(cudaFree(g_state.d_caustic_targets));
   if (g_state.d_sbt_miss)
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(g_state.d_sbt_miss)));
   if (g_state.d_sbt_hitgroup)
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(g_state.d_sbt_hitgroup)));
   freeDevicePtr(g_state.d_ias_output);
   freeDevicePtr(g_state.d_custom_gas_output);
   freeDevicePtr(g_state.d_triangle_gas_output);
   if (g_state.d_tri_vertices)
      CUDA_CHECK(cudaFree(g_state.d_tri_vertices));
   if (g_state.d_tri_data)
      CUDA_CHECK(cudaFree(g_state.d_tri_data));

   // Clean up GPU display pipeline resources
   if (g_state.render_stream)
      CUDA_CHECK(cudaStreamDestroy(g_state.render_stream));
   if (g_state.d_display)
      CUDA_CHECK(cudaFree(g_state.d_display));
   if (g_state.pinned_display)
      CUDA_CHECK(cudaFreeHost(g_state.pinned_display));

   if (g_state.pipeline)
      OPTIX_CHECK(optixPipelineDestroy(g_state.pipeline));
   if (g_state.raygen_pg)
      OPTIX_CHECK(optixProgramGroupDestroy(g_state.raygen_pg));
   if (g_state.raygen_caustics_pg)
      OPTIX_CHECK(optixProgramGroupDestroy(g_state.raygen_caustics_pg));
   if (g_state.miss_pg)
      OPTIX_CHECK(optixProgramGroupDestroy(g_state.miss_pg));
   if (g_state.miss_shadow_pg)
      OPTIX_CHECK(optixProgramGroupDestroy(g_state.miss_shadow_pg));
   if (g_state.hitgroup_sphere_pg)
      OPTIX_CHECK(optixProgramGroupDestroy(g_state.hitgroup_sphere_pg));
   if (g_state.hitgroup_rect_pg)
      OPTIX_CHECK(optixProgramGroupDestroy(g_state.hitgroup_rect_pg));
   if (g_state.hitgroup_sdf_pg)
      OPTIX_CHECK(optixProgramGroupDestroy(g_state.hitgroup_sdf_pg));
   if (g_state.hitgroup_triangle_pg)
      OPTIX_CHECK(optixProgramGroupDestroy(g_state.hitgroup_triangle_pg));
   if (g_state.module)
      OPTIX_CHECK(optixModuleDestroy(g_state.module));
   if (g_state.context)
      OPTIX_CHECK(optixDeviceContextDestroy(g_state.context));

   g_state = OptixState{};
}
