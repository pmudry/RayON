/**
 * @file scene_builder_cuda.cu
 * @brief CUDA implementation of scene builder
 *
 * This file contains the CUDA-specific scene building code that converts
 * host SceneDescription to GPU-friendly format and manages device memory.
 */

#include "cuda_scene.cuh"
#include "scene_description.hpp"
#include <cmath>
#include <cuda_runtime.h>

namespace Scene
{

// Declare CudaSceneBuilder class here to avoid pulling in scene_builder.h
class CudaSceneBuilder
{
 public:
   static CudaScene::Scene *buildGPUScene(const SceneDescription &desc);
   static void freeGPUScene(CudaScene::Scene *scene);
};

/**
 * @brief Convert MaterialDesc to GPU Material
 */
static CudaScene::Material convertMaterial(const MaterialDesc &desc)
{
   CudaScene::Material mat;

   // Convert material type
   mat.type = static_cast<CudaScene::MaterialType>(static_cast<uint8_t>(desc.type));

   // Convert vectors
   mat.albedo = f3(static_cast<float>(desc.albedo.x()), static_cast<float>(desc.albedo.y()),
                   static_cast<float>(desc.albedo.z()));
   mat.emission = f3(static_cast<float>(desc.emission.x()), static_cast<float>(desc.emission.y()),
                     static_cast<float>(desc.emission.z()));

   // Copy scalar properties
   mat.roughness = desc.roughness;
   mat.metallic = desc.metallic;
   mat.refractive_index = desc.refractive_index;
   mat.transmission = desc.transmission;
   mat.anisotropy = desc.anisotropy;
   mat.eta = f3(static_cast<float>(desc.eta.x()), static_cast<float>(desc.eta.y()),
                static_cast<float>(desc.eta.z()));
   mat.k = f3(static_cast<float>(desc.k.x()), static_cast<float>(desc.k.y()),
              static_cast<float>(desc.k.z()));
   mat.texture_id = desc.texture_id;

   // Copy pattern information
   mat.pattern = static_cast<CudaScene::ProceduralPattern>(static_cast<uint8_t>(desc.pattern));
   mat.pattern_color = f3(static_cast<float>(desc.pattern_color.x()), static_cast<float>(desc.pattern_color.y()),
                          static_cast<float>(desc.pattern_color.z()));
   mat.pattern_param1 = desc.pattern_param1;
   mat.pattern_param2 = desc.pattern_param2;

   // Copy thin-film interference parameters
   mat.film_thickness = desc.film_thickness;
   mat.film_ior = desc.film_ior;

   return mat;
}

/**
 * @brief Convert GeometryDesc to GPU Geometry
 */
static CudaScene::Geometry convertGeometry(const GeometryDesc &desc)
{
   CudaScene::Geometry geom;

   // Convert geometry type
   geom.type = static_cast<CudaScene::GeometryType>(static_cast<uint8_t>(desc.type));
   geom.material_id = desc.material_id;
   geom.visible = desc.visible;

   // Convert bounding box
   geom.bounds_min = f3(static_cast<float>(desc.bounds_min.x()), static_cast<float>(desc.bounds_min.y()),
                        static_cast<float>(desc.bounds_min.z()));
   geom.bounds_max = f3(static_cast<float>(desc.bounds_max.x()), static_cast<float>(desc.bounds_max.y()),
                        static_cast<float>(desc.bounds_max.z()));

   // Convert geometry-specific data
   switch (desc.type)
   {
   case GeometryType::SPHERE:
      geom.data.sphere.center =
          f3(static_cast<float>(desc.data.sphere.center.x()), static_cast<float>(desc.data.sphere.center.y()),
             static_cast<float>(desc.data.sphere.center.z()));
      geom.data.sphere.radius = static_cast<float>(desc.data.sphere.radius);
      break;

   case GeometryType::RECTANGLE:
      geom.data.rectangle.corner =
          f3(static_cast<float>(desc.data.rectangle.corner.x()), static_cast<float>(desc.data.rectangle.corner.y()),
             static_cast<float>(desc.data.rectangle.corner.z()));
      geom.data.rectangle.u =
          f3(static_cast<float>(desc.data.rectangle.u.x()), static_cast<float>(desc.data.rectangle.u.y()),
             static_cast<float>(desc.data.rectangle.u.z()));
      geom.data.rectangle.v =
          f3(static_cast<float>(desc.data.rectangle.v.x()), static_cast<float>(desc.data.rectangle.v.y()),
             static_cast<float>(desc.data.rectangle.v.z()));
      break;

   case GeometryType::DISPLACED_SPHERE:
      geom.data.displaced_sphere.center = f3(static_cast<float>(desc.data.displaced_sphere.center.x()),
                                             static_cast<float>(desc.data.displaced_sphere.center.y()),
                                             static_cast<float>(desc.data.displaced_sphere.center.z()));
      geom.data.displaced_sphere.radius = static_cast<float>(desc.data.displaced_sphere.radius);
      geom.data.displaced_sphere.displacement_scale = desc.data.displaced_sphere.displacement_scale;
      geom.data.displaced_sphere.pattern_type = desc.data.displaced_sphere.pattern_type;
      break;

   case GeometryType::TRIANGLE:
      geom.data.triangle.v0 = f3(static_cast<float>(desc.data.triangle.v0.x()),
                                 static_cast<float>(desc.data.triangle.v0.y()),
                                 static_cast<float>(desc.data.triangle.v0.z()));
      geom.data.triangle.v1 = f3(static_cast<float>(desc.data.triangle.v1.x()),
                                 static_cast<float>(desc.data.triangle.v1.y()),
                                 static_cast<float>(desc.data.triangle.v1.z()));
      geom.data.triangle.v2 = f3(static_cast<float>(desc.data.triangle.v2.x()),
                                 static_cast<float>(desc.data.triangle.v2.y()),
                                 static_cast<float>(desc.data.triangle.v2.z()));
      geom.data.triangle.n0 = f3(static_cast<float>(desc.data.triangle.n0.x()),
                                 static_cast<float>(desc.data.triangle.n0.y()),
                                 static_cast<float>(desc.data.triangle.n0.z()));
      geom.data.triangle.n1 = f3(static_cast<float>(desc.data.triangle.n1.x()),
                                 static_cast<float>(desc.data.triangle.n1.y()),
                                 static_cast<float>(desc.data.triangle.n1.z()));
      geom.data.triangle.n2 = f3(static_cast<float>(desc.data.triangle.n2.x()),
                                 static_cast<float>(desc.data.triangle.n2.y()),
                                 static_cast<float>(desc.data.triangle.n2.z()));
      geom.data.triangle.has_normals = desc.data.triangle.has_normals;
      break;

   default:
      break;
   }

   return geom;
}

/**
 * @brief Convert SceneDescription to GPU-compatible CudaScene
 */
CudaScene::Scene *CudaSceneBuilder::buildGPUScene(const SceneDescription &desc)
{
   // Allocate scene struct on HOST first (we'll copy to device at the end)
   CudaScene::Scene host_scene;

   // Allocate host arrays
   int num_materials = static_cast<int>(desc.materials.size());
   int num_geometries = static_cast<int>(desc.geometries.size());

   CudaScene::Material *host_materials = new CudaScene::Material[num_materials];
   CudaScene::Geometry *host_geometries = new CudaScene::Geometry[num_geometries];

   // Convert materials
   for (int i = 0; i < num_materials; ++i)
   {
      host_materials[i] = convertMaterial(desc.materials[i]);
   }

   // Convert geometries
   for (int i = 0; i < num_geometries; ++i)
   {
      host_geometries[i] = convertGeometry(desc.geometries[i]);
   }

   // Set scene properties
   host_scene.num_materials = num_materials;
   host_scene.num_geometries = num_geometries;
   host_scene.max_ray_march_steps = 100;
   host_scene.ray_march_epsilon = 0.001f;

   // Handle BVH if available
   host_scene.use_bvh = desc.use_bvh && !desc.top_level_bvh.nodes.empty();
   if (host_scene.use_bvh)
   {
      host_scene.num_bvh_nodes = static_cast<int>(desc.top_level_bvh.nodes.size());
      host_scene.bvh_root_idx = desc.top_level_bvh.root_index;

      // Convert BVH nodes
      CudaScene::BVHNode *host_bvh_nodes = new CudaScene::BVHNode[host_scene.num_bvh_nodes];
      for (int i = 0; i < host_scene.num_bvh_nodes; ++i)
      {
         const BVHNode &host_node = desc.top_level_bvh.nodes[i];
         CudaScene::BVHNode &device_node = host_bvh_nodes[i];

         // Copy bounding box
         device_node.bounds_min =
             f3(static_cast<float>(host_node.bounds_min.x()), static_cast<float>(host_node.bounds_min.y()),
                static_cast<float>(host_node.bounds_min.z()));
         device_node.bounds_max =
             f3(static_cast<float>(host_node.bounds_max.x()), static_cast<float>(host_node.bounds_max.y()),
                static_cast<float>(host_node.bounds_max.z()));

         device_node.is_leaf = host_node.is_leaf;
         device_node.split_axis = host_node.split_axis;

         if (host_node.is_leaf)
         {
            device_node.data.leaf.first_geom_idx = host_node.data.leaf.first_geom_idx;
            device_node.data.leaf.geom_count = host_node.data.leaf.geom_count;
         }
         else
         {
            device_node.data.interior.left_child = host_node.data.interior.left_child;
            device_node.data.interior.right_child = host_node.data.interior.right_child;
         }
      }

      // Allocate and copy BVH to device
      cudaMalloc(&host_scene.bvh_nodes, host_scene.num_bvh_nodes * sizeof(CudaScene::BVHNode));
      cudaMemcpy(host_scene.bvh_nodes, host_bvh_nodes, host_scene.num_bvh_nodes * sizeof(CudaScene::BVHNode),
                 cudaMemcpyHostToDevice);

      // Validate BVH root bounds
      if (host_scene.bvh_root_idx >= 0 && host_scene.bvh_root_idx < host_scene.num_bvh_nodes)
      {
         const CudaScene::BVHNode &root = host_bvh_nodes[host_scene.bvh_root_idx];
         if (std::isnan(root.bounds_min.x) || std::isinf(root.bounds_max.x) || std::isnan(root.bounds_min.y) ||
             std::isinf(root.bounds_max.y) || std::isnan(root.bounds_min.z) || std::isinf(root.bounds_max.z))
         {
            printf("❌ ERROR: BVH root has invalid bounds (NaN or Inf)!\n");
         }
         else
         {
            // printf("- Computed BVH root bounds: [%.2f, %.2f, %.2f] to [%.2f, %.2f, %.2f]\n", root.bounds_min.x,
            //        root.bounds_min.y, root.bounds_min.z, root.bounds_max.x, root.bounds_max.y, root.bounds_max.z);
         }
      }

      delete[] host_bvh_nodes;
   }
   else
   {
      host_scene.num_bvh_nodes = 0;
      host_scene.bvh_root_idx = -1;
      host_scene.bvh_nodes = nullptr;
   }

   // Allocate device memory and copy
   if (host_scene.num_materials > 0)
   {
      cudaMalloc(&host_scene.materials, host_scene.num_materials * sizeof(CudaScene::Material));
      cudaMemcpy(host_scene.materials, host_materials, host_scene.num_materials * sizeof(CudaScene::Material),
                 cudaMemcpyHostToDevice);
   }
   else
   {
      host_scene.materials = nullptr;
   }

   if (host_scene.num_geometries > 0)
   {
      cudaMalloc(&host_scene.geometries, host_scene.num_geometries * sizeof(CudaScene::Geometry));
      cudaMemcpy(host_scene.geometries, host_geometries, host_scene.num_geometries * sizeof(CudaScene::Geometry),
                 cudaMemcpyHostToDevice);
   }
   else
   {
      host_scene.geometries = nullptr;
   }

   // Free host arrays
   delete[] host_materials;
   delete[] host_geometries;

// Diagnostic: Print scene transfer summary
#ifdef DIAGS
   printf("\n- GPU Scene Transfer Summary:\n");
   printf("   Materials: %d\n", host_scene.num_materials);
   printf("   Geometries: %d\n", host_scene.num_geometries);
   printf("   BVH enabled: %s\n", host_scene.use_bvh ? "YES" : "NO");

   if (host_scene.use_bvh)
   {
      printf("   BVH nodes: %d (root idx: %d)\n", host_scene.num_bvh_nodes, host_scene.bvh_root_idx);
   }

#endif

   if (host_scene.num_geometries == 0)
   {
      printf("❌ ERROR: No geometries transferred to GPU! Scene will be empty.\n");
   }

   // Allocate the scene struct itself on device memory
   // This is required for older GPU architectures (Turing/RTX 2080) which
   // cannot access host-allocated structs containing device pointers
   CudaScene::Scene *d_scene;
   cudaMalloc(&d_scene, sizeof(CudaScene::Scene));
   cudaMemcpy(d_scene, &host_scene, sizeof(CudaScene::Scene), cudaMemcpyHostToDevice);

#ifdef DIAGS
   printf("   Scene struct allocated on device at: %p\n", d_scene);
#endif

   return d_scene;
}

/**
 * @brief Free GPU scene memory
 */
void CudaSceneBuilder::freeGPUScene(CudaScene::Scene *d_scene)
{
   if (!d_scene)
      return;

   // Copy scene struct from device to host to read the pointers
   CudaScene::Scene host_scene;
   cudaMemcpy(&host_scene, d_scene, sizeof(CudaScene::Scene), cudaMemcpyDeviceToHost);

   // Free device arrays
   if (host_scene.materials)
   {
      cudaFree(host_scene.materials);
   }
   if (host_scene.geometries)
   {
      cudaFree(host_scene.geometries);
   }
   if (host_scene.bvh_nodes)
   {
      cudaFree(host_scene.bvh_nodes);
   }

   // Free the scene struct itself (now on device)
   cudaFree(d_scene);
}

} // namespace Scene
