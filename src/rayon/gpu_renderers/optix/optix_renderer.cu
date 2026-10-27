// Host-side OptiX renderer implementation
// Handles: context, module, pipeline, GAS, SBT, launch

#include "optix_params.h"
#include "scene_description.hpp"

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
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
   OptixProgramGroup miss_pg = nullptr;
   OptixProgramGroup miss_shadow_pg = nullptr;
   OptixProgramGroup hitgroup_sphere_pg = nullptr;
   OptixProgramGroup hitgroup_rect_pg = nullptr;
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
   CUdeviceptr d_sbt_miss = 0;
   CUdeviceptr d_sbt_hitgroup = 0;
   OptixMaterialData *d_materials = nullptr;
   cudaTextureObject_t *d_textures = nullptr;
   int num_textures = 0;

   // Light list for NEE / MIS
   OptixLightData *d_lights = nullptr;
   float          *d_light_cdfs = nullptr;
   int            *d_geom_to_light = nullptr;
   int             num_lights = 0;
   int             num_geom_entries = 0;

   // MIS / NEE runtime settings (cached between launches)
   bool  mis_enabled          = true;
   bool  nee_first_bounce_only = false;
   int   nee_stride           = 1;

   // Persistent device accumulation buffer — stays on GPU across batches
   // to avoid costly per-batch host↔device float3↔float4 round-trips
   float4 *d_accum_buffer = nullptr;
   int accum_width = 0;
   int accum_height = 0;

   // Persistent device launch params — avoids cudaMalloc/cudaFree per batch
   CUdeviceptr d_launch_params = 0;

   // Golf ball dimple parameters (cached between launches, set via optixRendererSetGolfDimples)
   int   golf_dimple_count  = 150;
   float golf_dimple_radius = 0.24f;
   float golf_dimple_depth  = 0.35f;

   // HDR environment map managed outside the scene textures to survive scene rebuilds
   cudaArray_t         hdr_cuda_array = nullptr;
   cudaTextureObject_t hdr_tex_obj    = 0;

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
   OptixProgramGroup program_groups[] = {g_state.raygen_pg,          g_state.miss_pg,
                                          g_state.miss_shadow_pg,     g_state.hitgroup_sphere_pg,
                                          g_state.hitgroup_rect_pg,   g_state.hitgroup_triangle_pg};
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

   // Partition geometry: custom primitives (own SBT record each) vs. hardware triangles.
   // Unsupported types (SDF, CUBE, ...) are skipped with a one-time warning.
   std::vector<int> supported_indices; // custom primitives, in SBT order
   std::vector<int> triangle_indices;
   for (int i = 0; i < static_cast<int>(scene.geometries.size()); ++i)
   {
      switch (scene.geometries[i].type)
      {
      case Scene::GeometryType::SPHERE:
      case Scene::GeometryType::RECTANGLE:
      case Scene::GeometryType::DISPLACED_SPHERE:
         supported_indices.push_back(i);
         break;
      case Scene::GeometryType::TRIANGLE:
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
   const int num_tris  = static_cast<int>(triangle_indices.size());
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
         OptixProgramGroup pg =
             (g.type == Scene::GeometryType::RECTANGLE) ? g_state.hitgroup_rect_pg : g_state.hitgroup_sphere_pg;
         OPTIX_CHECK(optixSbtRecordPackHeader(pg, &rec));

         rec.data.material_idx = g.material_id;

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
      for (int i = 0; i < num_tris; ++i)
      {
         const auto &t = scene.geometries[triangle_indices[i]].data.triangle;
         vertices[3 * i + 0] = toFloat3(t.v0);
         vertices[3 * i + 1] = toFloat3(t.v1);
         vertices[3 * i + 2] = toFloat3(t.v2);

         OptixTriangleData &td = tri_data[i];
         td.material_idx = scene.geometries[triangle_indices[i]].material_id;
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
         materials[i].anisotropy = m.anisotropy;
         materials[i].eta = make_float3(static_cast<float>(m.eta.x()), static_cast<float>(m.eta.y()),
                                         static_cast<float>(m.eta.z()));
         materials[i].k = make_float3(static_cast<float>(m.k.x()), static_cast<float>(m.k.y()),
                                       static_cast<float>(m.k.z()));
         materials[i].film_thickness = m.film_thickness;
         materials[i].film_ior = m.film_ior;
         materials[i].texture_id = m.texture_id;
      }
      CUDA_CHECK(cudaMalloc(&g_state.d_materials, num_materials * sizeof(OptixMaterialData)));
      CUDA_CHECK(cudaMemcpy(g_state.d_materials, materials.data(), num_materials * sizeof(OptixMaterialData),
                             cudaMemcpyHostToDevice));
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
         // Build CDF for area-weighted light selection
         std::vector<float> cdf(g_state.num_lights + 1, 0.0f);
         for (int i = 0; i < g_state.num_lights; ++i)
            cdf[i + 1] = cdf[i] + host_areas[i];
         float total_area = cdf.back();
         if (total_area > 0.0f)
            for (auto &v : cdf) v /= total_area;
         cdf.back() = 1.0f; // ensure last entry is exactly 1

         CUDA_CHECK(cudaMalloc(&g_state.d_lights, g_state.num_lights * sizeof(OptixLightData)));
         CUDA_CHECK(cudaMemcpy(g_state.d_lights, host_lights.data(),
                                g_state.num_lights * sizeof(OptixLightData), cudaMemcpyHostToDevice));

         CUDA_CHECK(cudaMalloc(&g_state.d_light_cdfs, (g_state.num_lights + 1) * sizeof(float)));
         CUDA_CHECK(cudaMemcpy(g_state.d_light_cdfs, cdf.data(),
                                (g_state.num_lights + 1) * sizeof(float), cudaMemcpyHostToDevice));

         printf("OptiX NEE: %d lights uploaded (total area %.2f)\n", g_state.num_lights, total_area);
      }
   }
} // end buildGAS()

//==============================================================================
// PUBLIC INTERFACE (extern "C" for host renderer)
//==============================================================================

/// Runtime sampler toggle for OptiX — mirrors the CUDA renderer's g_use_sobol.
static bool g_optix_use_sobol = true;

extern "C" void setOptiXSobolSampler(bool use_sobol) { g_optix_use_sobol = use_sobol; }

extern "C" void setOptiXMISEnabled(bool enabled)          { g_state.mis_enabled = enabled; }
extern "C" void setOptiXNEEFirstBounceOnly(bool enabled)   { g_state.nee_first_bounce_only = enabled; }
extern "C" void setOptiXNEEStride(int stride)              { g_state.nee_stride = (stride < 1) ? 1 : stride; }

extern "C" void optixRendererInit() { initializeOptiX(); }

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
   }

   // Zero the buffer on the render stream so the memset is ordered before the next optixLaunch.
   // cudaMemset on the default stream (0) races with optixLaunch on the non-blocking render stream.
   CUDA_CHECK(cudaMemsetAsync(g_state.d_accum_buffer, 0, (size_t)width * height * sizeof(float4), getOptiXStream()));

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
                                                   float dof_aperture, float dof_focus_dist,
                                                   float light_intensity, float metal_fuzziness,
                                                   float glass_ior_multiplier)
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
   // Fresh Sobol scramble per accumulation (constant across its batches to keep stratification)
   static unsigned int sobol_sequence_seed = 0u;
   if (total_samples_so_far == 0)
      ++sobol_sequence_seed;
   launch_params.sobol_sequence_seed = sobol_sequence_seed;
   launch_params.use_sobol  = g_optix_use_sobol;
   launch_params.materials = g_state.d_materials;
   launch_params.num_materials = num_materials;
   launch_params.d_textures = g_state.d_textures;
   launch_params.num_textures = g_state.num_textures;
   launch_params.traversable = g_state.gas_handle;
   launch_params.background_intensity = bg_intensity;
   launch_params.dof_enabled = dof_enabled;
   launch_params.dof_aperture = dof_aperture;
   launch_params.dof_focus_distance = dof_focus_dist;
   launch_params.light_intensity = light_intensity;
   launch_params.metal_fuzziness = metal_fuzziness;
   launch_params.glass_ior_multiplier = glass_ior_multiplier;
   launch_params.golf_dimple_count  = g_state.golf_dimple_count;
   launch_params.golf_dimple_radius = g_state.golf_dimple_radius;
   launch_params.golf_dimple_depth  = g_state.golf_dimple_depth;
   launch_params.hdr_env_tex        = g_state.hdr_tex_obj;
   launch_params.use_hdr_env        = (g_state.hdr_tex_obj != 0);

   // NEE / MIS
   launch_params.lights             = g_state.d_lights;
   launch_params.num_lights         = g_state.num_lights;
   launch_params.light_cdfs         = g_state.d_light_cdfs;
   launch_params.mis_enabled        = g_state.mis_enabled;
   launch_params.nee_first_bounce_only = g_state.nee_first_bounce_only;
   launch_params.nee_stride         = g_state.nee_stride;
   launch_params.geom_to_light_map  = g_state.d_geom_to_light;
   launch_params.num_geom_entries   = g_state.num_geom_entries;

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
}

//==============================================================================
// GPU-side gamma correction for OptiX — mirrors the CUDA renderer's pipeline.
// Converts float4 accum buffer directly to uint8 display image on the GPU,
// then async-copies via pinned memory. Avoids the expensive float4 D2H transfer
// + host-side conversion that the original optixRendererDownloadAccum() used.
//==============================================================================
__global__ void optixGammaCorrectKernel(const float4 *__restrict__ accum_buffer, unsigned char *display_image,
                                         int width, int height, int num_samples, int channels, float gamma)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   if (x >= width || y >= height)
      return;

   int pixel_idx = y * width + x;
   float4 acc = accum_buffer[pixel_idx];

   float inv_samples = 1.0f / (float)num_samples;
   float inv_gamma = 1.0f / gamma;

   float r = fminf(powf(fmaxf(acc.x * inv_samples, 0.0f), inv_gamma), 0.999f);
   float g = fminf(powf(fmaxf(acc.y * inv_samples, 0.0f), inv_gamma), 0.999f);
   float b = fminf(powf(fmaxf(acc.z * inv_samples, 0.0f), inv_gamma), 0.999f);

   int image_idx = pixel_idx * channels;
   display_image[image_idx + 0] = (unsigned char)(256.0f * r);
   display_image[image_idx + 1] = (unsigned char)(256.0f * g);
   display_image[image_idx + 2] = (unsigned char)(256.0f * b);
   if (channels == 4)
      display_image[image_idx + 3] = 255;
}

extern "C" void optixRendererConvertAccumToDisplay(unsigned char *display_image, int width, int height,
                                                    int channels, int num_samples, float gamma)
{
   if (!g_state.d_accum_buffer || !display_image || num_samples <= 0)
      return;

   size_t display_size = (size_t)width * height * channels * sizeof(unsigned char);

   // Allocate/resize device display buffer (persistent across calls)
   if (g_state.d_display == nullptr || g_state.d_display_size != display_size)
   {
      if (g_state.d_display != nullptr)
         cudaFree(g_state.d_display);
      cudaMalloc(&g_state.d_display, display_size);
      g_state.d_display_size = display_size;
   }

   // Allocate/resize pinned host staging buffer for async D2H copy
   if (g_state.pinned_display == nullptr || g_state.pinned_display_size != display_size)
   {
      if (g_state.pinned_display != nullptr)
         cudaFreeHost(g_state.pinned_display);
      cudaMallocHost(&g_state.pinned_display, display_size);
      g_state.pinned_display_size = display_size;
   }

   dim3 threads(32, 8);
   dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

   cudaStream_t stream = getOptiXStream();

   optixGammaCorrectKernel<<<blocks, threads, 0, stream>>>(
       g_state.d_accum_buffer, g_state.d_display, width, height, num_samples, channels, gamma);

   // Async D2H copy via pinned memory, then single stream sync
   cudaMemcpyAsync(g_state.pinned_display, g_state.d_display, display_size, cudaMemcpyDeviceToHost, stream);
   cudaStreamSynchronize(stream);
   memcpy(display_image, g_state.pinned_display, display_size);
}

extern "C" void optixRendererClearHdrEnv()
{
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
   if (g_state.d_launch_params)
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(g_state.d_launch_params)));
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
   if (g_state.miss_pg)
      OPTIX_CHECK(optixProgramGroupDestroy(g_state.miss_pg));
   if (g_state.miss_shadow_pg)
      OPTIX_CHECK(optixProgramGroupDestroy(g_state.miss_shadow_pg));
   if (g_state.hitgroup_sphere_pg)
      OPTIX_CHECK(optixProgramGroupDestroy(g_state.hitgroup_sphere_pg));
   if (g_state.hitgroup_rect_pg)
      OPTIX_CHECK(optixProgramGroupDestroy(g_state.hitgroup_rect_pg));
   if (g_state.hitgroup_triangle_pg)
      OPTIX_CHECK(optixProgramGroupDestroy(g_state.hitgroup_triangle_pg));
   if (g_state.module)
      OPTIX_CHECK(optixModuleDestroy(g_state.module));
   if (g_state.context)
      OPTIX_CHECK(optixDeviceContextDestroy(g_state.context));

   g_state = OptixState{};
}
