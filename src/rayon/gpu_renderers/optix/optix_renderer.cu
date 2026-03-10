// Host-side OptiX renderer implementation
// Handles: context, module, pipeline, GAS, SBT, launch

#include "optix_params.h"
#include "scene_description.hpp"

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <cuda_runtime.h>
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
   OptixProgramGroup miss_pg = nullptr;
   OptixProgramGroup hitgroup_sphere_pg = nullptr;
   OptixProgramGroup hitgroup_rect_pg = nullptr;

   OptixTraversableHandle gas_handle = 0;
   CUdeviceptr d_gas_output = 0;

   // Device buffers
   CUdeviceptr d_sbt_raygen = 0;
   CUdeviceptr d_sbt_miss = 0;
   CUdeviceptr d_sbt_hitgroup = 0;
   OptixMaterialData *d_materials = nullptr;

   // Persistent device accumulation buffer — stays on GPU across batches
   // to avoid costly per-batch host↔device float3↔float4 round-trips
   float4 *d_accum_buffer = nullptr;
   int accum_width = 0;
   int accum_height = 0;

   // Persistent device launch params — avoids cudaMalloc/cudaFree per batch
   CUdeviceptr d_launch_params = 0;

   bool initialized = false;
};

static OptixState g_state;

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
   pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
   pipeline_options.numPayloadValues = 2; // PRD pointer (2 x uint32)
   pipeline_options.numAttributeValues = 3; // Normal (3 x float)
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

   // Miss
   OptixProgramGroupDesc miss_desc = {};
   miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
   miss_desc.miss.module = g_state.module;
   miss_desc.miss.entryFunctionName = "__miss__ms";
   optix_log_size = sizeof(optix_log);
   OPTIX_CHECK(
       optixProgramGroupCreate(g_state.context, &miss_desc, 1, &pg_options, optix_log, &optix_log_size, &g_state.miss_pg));

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

   // Create pipeline
   const uint32_t max_trace_depth = 1; // We loop in raygen, single trace per bounce
   OptixProgramGroup program_groups[] = {g_state.raygen_pg, g_state.miss_pg, g_state.hitgroup_sphere_pg,
                                          g_state.hitgroup_rect_pg};

   OptixPipelineLinkOptions link_options = {};
   link_options.maxTraceDepth = max_trace_depth;
   optix_log_size = sizeof(optix_log);
   OPTIX_CHECK(optixPipelineCreate(g_state.context, &pipeline_options, &link_options, program_groups, 4, optix_log,
                                    &optix_log_size, &g_state.pipeline));

   // Compute stack sizes
   OptixStackSizes stack_sizes = {};
   for (auto &pg : program_groups)
      OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes, g_state.pipeline));

   uint32_t dc_from_traversal, dc_from_state, continuation;
   OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth, 0, 0, &dc_from_traversal, &dc_from_state,
                                           &continuation));
   OPTIX_CHECK(optixPipelineSetStackSize(g_state.pipeline, dc_from_traversal, dc_from_state, continuation, 1));

   // Create raygen SBT record
   RayGenRecord rg_record;
   OPTIX_CHECK(optixSbtRecordPackHeader(g_state.raygen_pg, &rg_record));
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_state.d_sbt_raygen), sizeof(RayGenRecord)));
   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(g_state.d_sbt_raygen), &rg_record, sizeof(RayGenRecord),
                          cudaMemcpyHostToDevice));

   // Create miss SBT record
   MissRecord ms_record;
   OPTIX_CHECK(optixSbtRecordPackHeader(g_state.miss_pg, &ms_record));
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_state.d_sbt_miss), sizeof(MissRecord)));
   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(g_state.d_sbt_miss), &ms_record, sizeof(MissRecord),
                          cudaMemcpyHostToDevice));

   g_state.sbt.raygenRecord = g_state.d_sbt_raygen;
   g_state.sbt.missRecordBase = g_state.d_sbt_miss;
   g_state.sbt.missRecordStrideInBytes = sizeof(MissRecord);
   g_state.sbt.missRecordCount = 1;

   g_state.initialized = true;
   printf("OptiX renderer initialized successfully\n");
}

// Build GAS from scene description
static void buildGAS(const Scene::SceneDescription &scene)
{
   // Free previous resources
   if (g_state.d_gas_output)
   {
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(g_state.d_gas_output)));
      g_state.d_gas_output = 0;
   }
   if (g_state.d_sbt_hitgroup)
   {
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(g_state.d_sbt_hitgroup)));
      g_state.d_sbt_hitgroup = 0;
   }

   // Filter out SDF primitives — OptiX has no intersection program for ray-marched SDFs,
   // matching CUDA behavior which also skips them (intersect_geometry returns false).
   std::vector<int> supported_indices;
   for (int i = 0; i < static_cast<int>(scene.geometries.size()); ++i)
   {
      if (scene.geometries[i].type != Scene::GeometryType::SDF_PRIMITIVE)
         supported_indices.push_back(i);
   }

   int num_geoms = static_cast<int>(supported_indices.size());
   if (num_geoms == 0)
      return;

   // Build AABBs (only for supported geometry types)
   std::vector<OptixAabb> aabbs(num_geoms);
   for (int i = 0; i < num_geoms; ++i)
   {
      const auto &g = scene.geometries[supported_indices[i]];
      aabbs[i].minX = static_cast<float>(g.bounds_min.x());
      aabbs[i].minY = static_cast<float>(g.bounds_min.y());
      aabbs[i].minZ = static_cast<float>(g.bounds_min.z());
      aabbs[i].maxX = static_cast<float>(g.bounds_max.x());
      aabbs[i].maxY = static_cast<float>(g.bounds_max.y());
      aabbs[i].maxZ = static_cast<float>(g.bounds_max.z());
   }

   CUdeviceptr d_aabbs;
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_aabbs), num_geoms * sizeof(OptixAabb)));
   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_aabbs), aabbs.data(), num_geoms * sizeof(OptixAabb),
                          cudaMemcpyHostToDevice));

   // Per-geometry SBT index (each geometry gets its own record for per-geom data)
   std::vector<uint32_t> sbt_indices(num_geoms);
   for (int i = 0; i < num_geoms; ++i)
      sbt_indices[i] = i;

   CUdeviceptr d_sbt_indices;
   CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_sbt_indices), num_geoms * sizeof(uint32_t)));
   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_sbt_indices), sbt_indices.data(), num_geoms * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

   // One flag per SBT record (= per geometry)
   std::vector<uint32_t> flags(num_geoms, OPTIX_GEOMETRY_FLAG_NONE);

   OptixBuildInput build_input = {};
   build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
   build_input.customPrimitiveArray.aabbBuffers = &d_aabbs;
   build_input.customPrimitiveArray.numPrimitives = num_geoms;
   build_input.customPrimitiveArray.flags = flags.data();
   build_input.customPrimitiveArray.numSbtRecords = num_geoms;
   build_input.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_indices;
   build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
   build_input.customPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

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

   OPTIX_CHECK(optixAccelBuild(g_state.context, 0, &accel_options, &build_input, 1, d_temp, gas_sizes.tempSizeInBytes,
                                d_output, gas_sizes.outputSizeInBytes, &g_state.gas_handle, &emit_desc, 1));
   CUDA_CHECK(cudaDeviceSynchronize());

   // Compact
   size_t compacted_size;
   CUDA_CHECK(cudaMemcpy(&compacted_size, reinterpret_cast<void *>(d_output + compacted_offset), sizeof(size_t),
                          cudaMemcpyDeviceToHost));
   if (compacted_size < gas_sizes.outputSizeInBytes)
   {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_state.d_gas_output), compacted_size));
      OPTIX_CHECK(
          optixAccelCompact(g_state.context, 0, g_state.gas_handle, g_state.d_gas_output, compacted_size, &g_state.gas_handle));
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_output)));
   }
   else
   {
      g_state.d_gas_output = d_output;
   }

   CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp)));
   CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_aabbs)));
   CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_sbt_indices)));

   // Build per-geometry SBT hit group records (using filtered indices)
   std::vector<HitGroupRecord> hitgroup_records(num_geoms);
   for (int i = 0; i < num_geoms; ++i)
   {
      const auto &g = scene.geometries[supported_indices[i]];
      HitGroupRecord &rec = hitgroup_records[i];

      // Select program group based on geometry type
      OptixProgramGroup pg =
          (g.type == Scene::GeometryType::RECTANGLE) ? g_state.hitgroup_rect_pg : g_state.hitgroup_sphere_pg;
      OPTIX_CHECK(optixSbtRecordPackHeader(pg, &rec));

      rec.data.material_idx = g.material_id;

      switch (g.type)
      {
      case Scene::GeometryType::SPHERE:
         rec.data.geom_type = OptixGeomType::SPHERE;
         rec.data.center = make_float3(static_cast<float>(g.data.sphere.center.x()),
                                        static_cast<float>(g.data.sphere.center.y()),
                                        static_cast<float>(g.data.sphere.center.z()));
         rec.data.radius = static_cast<float>(g.data.sphere.radius);
         break;

      case Scene::GeometryType::DISPLACED_SPHERE:
         rec.data.geom_type = OptixGeomType::DISPLACED_SPHERE;
         rec.data.center = make_float3(static_cast<float>(g.data.displaced_sphere.center.x()),
                                        static_cast<float>(g.data.displaced_sphere.center.y()),
                                        static_cast<float>(g.data.displaced_sphere.center.z()));
         rec.data.radius = static_cast<float>(g.data.displaced_sphere.radius);
         break;

      case Scene::GeometryType::RECTANGLE:
      {
         rec.data.geom_type = OptixGeomType::RECTANGLE;
         rec.data.center = make_float3(static_cast<float>(g.data.rectangle.corner.x()),
                                        static_cast<float>(g.data.rectangle.corner.y()),
                                        static_cast<float>(g.data.rectangle.corner.z()));
         float3 u = make_float3(static_cast<float>(g.data.rectangle.u.x()), static_cast<float>(g.data.rectangle.u.y()),
                                 static_cast<float>(g.data.rectangle.u.z()));
         float3 v = make_float3(static_cast<float>(g.data.rectangle.v.x()), static_cast<float>(g.data.rectangle.v.y()),
                                 static_cast<float>(g.data.rectangle.v.z()));
         rec.data.u_vec = u;
         rec.data.v_vec = v;
         // Precompute normal = normalize(u x v)
         float3 n = make_float3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
         float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
         rec.data.normal = (len > 1e-8f) ? make_float3(n.x / len, n.y / len, n.z / len) : make_float3(0, 1, 0);
         break;
      }

      default:
         // Treat unknown as sphere at origin
         rec.data.geom_type = OptixGeomType::SPHERE;
         rec.data.center = make_float3(0, 0, 0);
         rec.data.radius = 0.5f;
         break;
      }
   }

   CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_state.d_sbt_hitgroup), num_geoms * sizeof(HitGroupRecord)));
   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(g_state.d_sbt_hitgroup), hitgroup_records.data(),
                          num_geoms * sizeof(HitGroupRecord), cudaMemcpyHostToDevice));

   g_state.sbt.hitgroupRecordBase = g_state.d_sbt_hitgroup;
   g_state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
   g_state.sbt.hitgroupRecordCount = num_geoms;

   // Upload materials
   if (g_state.d_materials)
   {
      CUDA_CHECK(cudaFree(g_state.d_materials));
      g_state.d_materials = nullptr;
   }

   int num_materials = static_cast<int>(scene.materials.size());
   if (num_materials > 0)
   {
      std::vector<OptixMaterialData> materials(num_materials);
      for (int i = 0; i < num_materials; ++i)
      {
         const auto &m = scene.materials[i];
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
      }
      CUDA_CHECK(cudaMalloc(&g_state.d_materials, num_materials * sizeof(OptixMaterialData)));
      CUDA_CHECK(cudaMemcpy(g_state.d_materials, materials.data(), num_materials * sizeof(OptixMaterialData),
                             cudaMemcpyHostToDevice));
   }

   printf("OptiX GAS built: %d geometries, %d materials\n", num_geoms, num_materials);
}

//==============================================================================
// PUBLIC INTERFACE (extern "C" for host renderer)
//==============================================================================

extern "C" void optixRendererInit() { initializeOptiX(); }

extern "C" void optixRendererBuildScene(const Scene::SceneDescription &scene) { buildGAS(scene); }

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
   }

   // Zero the buffer on device — no host round-trip needed
   CUDA_CHECK(cudaMemset(g_state.d_accum_buffer, 0, (size_t)width * height * sizeof(float4)));

   // Allocate persistent launch params buffer (once)
   if (g_state.d_launch_params == 0)
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_state.d_launch_params), sizeof(OptixLaunchParams)));
}

// Launch a batch of samples. Accumulation buffer stays on device — no host transfer per batch.
// Uses persistent device launch params buffer to avoid cudaMalloc/cudaFree overhead per batch.
extern "C" unsigned long long optixRendererLaunch(int width, int height, int num_materials, int samples_to_add,
                                                   int total_samples_so_far, int max_depth, float cam_cx, float cam_cy,
                                                   float cam_cz, float p00x, float p00y, float p00z, float dux,
                                                   float duy, float duz, float dvx, float dvy, float dvz,
                                                   float cam_ux, float cam_uy, float cam_uz, float cam_vx,
                                                   float cam_vy, float cam_vz, float bg_intensity, bool dof_enabled,
                                                   float dof_aperture, float dof_focus_dist)
{
   if (!g_state.initialized)
      return 0;

   // Fill launch params on host stack, then single cudaMemcpy to persistent device buffer
   OptixLaunchParams launch_params = {};
   launch_params.accum_buffer = g_state.d_accum_buffer;
   launch_params.width = width;
   launch_params.height = height;
   launch_params.camera_center = make_float3(cam_cx, cam_cy, cam_cz);
   launch_params.pixel00_loc = make_float3(p00x, p00y, p00z);
   launch_params.pixel_delta_u = make_float3(dux, duy, duz);
   launch_params.pixel_delta_v = make_float3(dvx, dvy, dvz);
   launch_params.cam_u = make_float3(cam_ux, cam_uy, cam_uz);
   launch_params.cam_v = make_float3(cam_vx, cam_vy, cam_vz);
   launch_params.samples_per_launch = samples_to_add;
   launch_params.total_samples_so_far = total_samples_so_far;
   launch_params.max_depth = max_depth;
   launch_params.frame_seed = total_samples_so_far + 42;
   launch_params.materials = g_state.d_materials;
   launch_params.num_materials = num_materials;
   launch_params.traversable = g_state.gas_handle;
   launch_params.background_intensity = bg_intensity;
   launch_params.dof_enabled = dof_enabled;
   launch_params.dof_aperture = dof_aperture;
   launch_params.dof_focus_distance = dof_focus_dist;

   // Single memcpy to persistent device buffer — no malloc/free per batch
   CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(g_state.d_launch_params), &launch_params, sizeof(OptixLaunchParams),
                          cudaMemcpyHostToDevice));

   OPTIX_CHECK(optixLaunch(g_state.pipeline, 0, g_state.d_launch_params, sizeof(OptixLaunchParams), &g_state.sbt,
                            width, height, 1));
   CUDA_CHECK(cudaDeviceSynchronize());

   return (unsigned long long)width * height * samples_to_add;
}

// Download accumulated results from device to host. Called once after the entire render loop.
// Performs float4 → float3 conversion only at this final step.
extern "C" void optixRendererDownloadAccum(float *host_accum_buffer, int width, int height)
{
   if (!g_state.d_accum_buffer)
      return;

   int num_pixels = width * height;
   size_t accum_size = (size_t)num_pixels * sizeof(float4);
   float4 *host_f4 = (float4 *)malloc(accum_size);
   CUDA_CHECK(cudaMemcpy(host_f4, g_state.d_accum_buffer, accum_size, cudaMemcpyDeviceToHost));

   for (int i = 0; i < num_pixels; ++i)
   {
      host_accum_buffer[i * 3] = host_f4[i].x;
      host_accum_buffer[i * 3 + 1] = host_f4[i].y;
      host_accum_buffer[i * 3 + 2] = host_f4[i].z;
   }
   free(host_f4);
}

extern "C" void optixRendererCleanup()
{
   if (g_state.d_accum_buffer)
      CUDA_CHECK(cudaFree(g_state.d_accum_buffer));
   if (g_state.d_launch_params)
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(g_state.d_launch_params)));
   if (g_state.d_materials)
      CUDA_CHECK(cudaFree(g_state.d_materials));
   if (g_state.d_sbt_raygen)
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(g_state.d_sbt_raygen)));
   if (g_state.d_sbt_miss)
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(g_state.d_sbt_miss)));
   if (g_state.d_sbt_hitgroup)
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(g_state.d_sbt_hitgroup)));
   if (g_state.d_gas_output)
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(g_state.d_gas_output)));

   if (g_state.pipeline)
      OPTIX_CHECK(optixPipelineDestroy(g_state.pipeline));
   if (g_state.raygen_pg)
      OPTIX_CHECK(optixProgramGroupDestroy(g_state.raygen_pg));
   if (g_state.miss_pg)
      OPTIX_CHECK(optixProgramGroupDestroy(g_state.miss_pg));
   if (g_state.hitgroup_sphere_pg)
      OPTIX_CHECK(optixProgramGroupDestroy(g_state.hitgroup_sphere_pg));
   if (g_state.hitgroup_rect_pg)
      OPTIX_CHECK(optixProgramGroupDestroy(g_state.hitgroup_rect_pg));
   if (g_state.module)
      OPTIX_CHECK(optixModuleDestroy(g_state.module));
   if (g_state.context)
      OPTIX_CHECK(optixDeviceContextDestroy(g_state.context));

   g_state = OptixState{};
}
