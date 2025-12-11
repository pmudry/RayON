/**
 * @file scene_builder_cuda.cu
 * @brief CUDA implementation of scene builder
 *
 * This file contains the CUDA-specific scene building code that converts
 * host SceneDescription to GPU-friendly format and manages device memory.
 */

#include "cuda_scene.cuh"
#include "cuda_utils.cuh"
#include "scene_description.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

namespace Scene
{

// Declare CudaSceneBuilder class here to avoid pulling in scene_builder.h
class CudaSceneBuilder
{
 public:
   static CudaScene::Scene *buildGPUScene(const SceneDescription &desc);
   static void freeGPUScene(CudaScene::Scene *scene);
};

//==============================================================================
// MESH BVH BUILDER HELPERS
//==============================================================================

struct BVHBuildNode
{
   f3 bounds_min, bounds_max;
   bool is_leaf;
   int left_child;  // or first_geom_idx
   int right_child; // or geom_count
   int split_axis;
};

// Helper to compute bounds of a triangle
static void getTriangleBounds(const CudaScene::MeshTriangle &tri, f3 &min_b, f3 &max_b)
{
   min_b.x = fminf(tri.v0.x, fminf(tri.v1.x, tri.v2.x));
   min_b.y = fminf(tri.v0.y, fminf(tri.v1.y, tri.v2.y));
   min_b.z = fminf(tri.v0.z, fminf(tri.v1.z, tri.v2.z));

   max_b.x = fmaxf(tri.v0.x, fmaxf(tri.v1.x, tri.v2.x));
   max_b.y = fmaxf(tri.v0.y, fmaxf(tri.v1.y, tri.v2.y));
   max_b.z = fmaxf(tri.v0.z, fmaxf(tri.v1.z, tri.v2.z));
}

// Recursive BVH builder
static int buildMeshBVHRecursive(std::vector<CudaScene::BVHNode> &nodes, std::vector<int> &indices,
                                 const std::vector<CudaScene::MeshTriangle> &triangles, int start, int end)
{
   CudaScene::BVHNode node;
   
   // Compute bounds for all triangles in this range
   f3 bounds_min(1e30f, 1e30f, 1e30f);
   f3 bounds_max(-1e30f, -1e30f, -1e30f);

   for (int i = start; i < end; ++i)
   {
      f3 t_min, t_max;
      getTriangleBounds(triangles[indices[i]], t_min, t_max);

      bounds_min.x = fminf(bounds_min.x, t_min.x);
      bounds_min.y = fminf(bounds_min.y, t_min.y);
      bounds_min.z = fminf(bounds_min.z, t_min.z);

      bounds_max.x = fmaxf(bounds_max.x, t_max.x);
      bounds_max.y = fmaxf(bounds_max.y, t_max.y);
      bounds_max.z = fmaxf(bounds_max.z, t_max.z);
   }

   node.bounds_min = bounds_min;
   node.bounds_max = bounds_max;

   int count = end - start;

   // Leaf node condition
   if (count <= 4)
   {
      node.is_leaf = true;
      node.split_axis = 0;
      node.data.leaf.first_geom_idx = start;
      node.data.leaf.geom_count = count;

      int node_idx = static_cast<int>(nodes.size());
      nodes.push_back(node);
      return node_idx;
   }

   // Interior node: Median Split
   f3 centroid_min(1e30f, 1e30f, 1e30f);
   f3 centroid_max(-1e30f, -1e30f, -1e30f);

   for (int i = start; i < end; ++i)
   {
      f3 t_min, t_max;
      getTriangleBounds(triangles[indices[i]], t_min, t_max);
      f3 center = (t_min + t_max) * 0.5f;

      centroid_min.x = fminf(centroid_min.x, center.x);
      centroid_min.y = fminf(centroid_min.y, center.y);
      centroid_min.z = fminf(centroid_min.z, center.z);

      centroid_max.x = fmaxf(centroid_max.x, center.x);
      centroid_max.y = fmaxf(centroid_max.y, center.y);
      centroid_max.z = fmaxf(centroid_max.z, center.z);
   }

   f3 extent = centroid_max - centroid_min;
   int best_axis = 0;
   if (extent.y > extent.x) best_axis = 1;
   if (extent.z > extent.y && extent.z > extent.x) best_axis = 2;

   int mid = start + count / 2;

   // Partition
   std::nth_element(indices.begin() + start, indices.begin() + mid, indices.begin() + end,
                    [&](int a, int b) {
                       f3 min_a, max_a, min_b, max_b;
                       getTriangleBounds(triangles[a], min_a, max_a);
                       getTriangleBounds(triangles[b], min_b, max_b);
                       f3 ca = (min_a + max_a) * 0.5f;
                       f3 cb = (min_b + max_b) * 0.5f;
                       
                       if (best_axis == 0) return ca.x < cb.x;
                       if (best_axis == 1) return ca.y < cb.y;
                       return ca.z < cb.z;
                    });

   node.is_leaf = false;
   node.split_axis = static_cast<uint8_t>(best_axis);

   int node_idx = static_cast<int>(nodes.size());
   nodes.push_back(node);

   int left = buildMeshBVHRecursive(nodes, indices, triangles, start, mid);
   int right = buildMeshBVHRecursive(nodes, indices, triangles, mid, end);

   // Update node (careful with vector reallocation, use index)
   nodes[node_idx].data.interior.left_child = left;
   nodes[node_idx].data.interior.right_child = right;

   return node_idx;
}

/**
 * @brief Convert a TriangleMesh to a CudaScene::Mesh (allocating device memory)
 */
static CudaScene::Mesh processMesh(const TriangleMesh &mesh_desc)
{
   CudaScene::Mesh gpu_mesh;
   
   if (mesh_desc.triangles.empty())
      return gpu_mesh;

   // 1. Convert triangles to GPU format
   std::vector<CudaScene::MeshTriangle> host_triangles;
   host_triangles.reserve(mesh_desc.triangles.size());

   for (const auto &tri : mesh_desc.triangles)
   {
      CudaScene::MeshTriangle t;
      t.v0 = f3(tri.v0.x(), tri.v0.y(), tri.v0.z());
      t.v1 = f3(tri.v1.x(), tri.v1.y(), tri.v1.z());
      t.v2 = f3(tri.v2.x(), tri.v2.y(), tri.v2.z());
      
      if (tri.smooth_shadings) {
          t.n0 = f3(tri.n0.x(), tri.n0.y(), tri.n0.z());
          t.n1 = f3(tri.n1.x(), tri.n1.y(), tri.n1.z());
          t.n2 = f3(tri.n2.x(), tri.n2.y(), tri.n2.z());
      }
      t.smooth_shadings = tri.smooth_shadings;
      host_triangles.push_back(t);
   }

   // 2. Build BVH
   std::vector<CudaScene::BVHNode> host_nodes;
   std::vector<int> indices(host_triangles.size());
   for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;

   int root_idx = buildMeshBVHRecursive(host_nodes, indices, host_triangles, 0, static_cast<int>(host_triangles.size()));

   // 3. Reorder triangles to match leaf order
   std::vector<CudaScene::MeshTriangle> reordered_triangles;
   reordered_triangles.reserve(host_triangles.size());
   
   // We need to traverse the built BVH to collect triangles in order, 
   // but since we updated indices in-place during build, we can just use the indices array!
   // WAIT: The indices array is partitioned, so indices[start...end] corresponds to the leaf's triangles.
   // But we need to make sure the leaf.first_geom_idx points to the correct offset in the *reordered* array.
   // The current recursive builder sets first_geom_idx = start (which is index into 'indices' array).
   // So if we just reorder host_triangles according to 'indices', then 'start' will be correct index into reordered array.

   for (int idx : indices) {
      reordered_triangles.push_back(host_triangles[idx]);
   }

   // 4. Upload to GPU
   gpu_mesh.num_triangles = static_cast<int>(reordered_triangles.size());
   gpu_mesh.num_bvh_nodes = static_cast<int>(host_nodes.size());
   gpu_mesh.bvh_root_idx = root_idx;

   cudaMalloc((void**)&gpu_mesh.triangles, gpu_mesh.num_triangles * sizeof(CudaScene::MeshTriangle));
   cudaMemcpy(gpu_mesh.triangles, reordered_triangles.data(), gpu_mesh.num_triangles * sizeof(CudaScene::MeshTriangle), cudaMemcpyHostToDevice);

   cudaMalloc((void**)&gpu_mesh.bvh_nodes, gpu_mesh.num_bvh_nodes * sizeof(CudaScene::BVHNode));
   cudaMemcpy(gpu_mesh.bvh_nodes, host_nodes.data(), gpu_mesh.num_bvh_nodes * sizeof(CudaScene::BVHNode), cudaMemcpyHostToDevice);

   return gpu_mesh;
}

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
   mat.texture_id = desc.texture_id;

   // Copy pattern information
   mat.pattern = static_cast<CudaScene::ProceduralPattern>(static_cast<uint8_t>(desc.pattern));
   mat.pattern_color = f3(static_cast<float>(desc.pattern_color.x()), static_cast<float>(desc.pattern_color.y()),
                          static_cast<float>(desc.pattern_color.z()));
   mat.pattern_param1 = desc.pattern_param1;
   mat.pattern_param2 = desc.pattern_param2;

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
      
      if (desc.data.triangle.smooth_shadings) {
          geom.data.triangle.n0 = f3(static_cast<float>(desc.data.triangle.n0.x()),
                                     static_cast<float>(desc.data.triangle.n0.y()),
                                     static_cast<float>(desc.data.triangle.n0.z()));
          geom.data.triangle.n1 = f3(static_cast<float>(desc.data.triangle.n1.x()),
                                     static_cast<float>(desc.data.triangle.n1.y()),
                                     static_cast<float>(desc.data.triangle.n1.z()));
          geom.data.triangle.n2 = f3(static_cast<float>(desc.data.triangle.n2.x()),
                                     static_cast<float>(desc.data.triangle.n2.y()),
                                     static_cast<float>(desc.data.triangle.n2.z()));
      }
      geom.data.triangle.smooth_shadings = desc.data.triangle.smooth_shadings;
      break;

   case GeometryType::TRIANGLE_MESH:
      geom.data.mesh_instance.mesh_id = desc.data.mesh_instance.mesh_id;
      geom.data.mesh_instance.bvh_root_idx = -1; // Not used for mesh instance? Or maybe store root of mesh BVH here?
                                                 // No, mesh BVH root is in Mesh struct.
      geom.data.mesh_instance.translation = f3(static_cast<float>(desc.data.mesh_instance.translation.x()),
                                               static_cast<float>(desc.data.mesh_instance.translation.y()),
                                               static_cast<float>(desc.data.mesh_instance.translation.z()));
      geom.data.mesh_instance.rotation = f3(static_cast<float>(desc.data.mesh_instance.rotation.x()),
                                            static_cast<float>(desc.data.mesh_instance.rotation.y()),
                                            static_cast<float>(desc.data.mesh_instance.rotation.z()));
      geom.data.mesh_instance.scale = f3(static_cast<float>(desc.data.mesh_instance.scale.x()),
                                         static_cast<float>(desc.data.mesh_instance.scale.y()),
                                         static_cast<float>(desc.data.mesh_instance.scale.z()));
      break;

   // Other geometry types to be implemented
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
   int num_meshes = static_cast<int>(desc.meshes.size());

   CudaScene::Material *host_materials = new CudaScene::Material[num_materials];
   CudaScene::Geometry *host_geometries = new CudaScene::Geometry[num_geometries];
   CudaScene::Mesh *host_meshes = num_meshes > 0 ? new CudaScene::Mesh[num_meshes] : nullptr;

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

   // Convert meshes (and build their BVHs)
   for (int i = 0; i < num_meshes; ++i)
   {
      host_meshes[i] = processMesh(desc.meshes[i]);
   }

   // Set scene properties
   host_scene.num_materials = num_materials;
   host_scene.num_geometries = num_geometries;
   host_scene.num_meshes = num_meshes;
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
      cudaMalloc((void**)&host_scene.bvh_nodes, host_scene.num_bvh_nodes * sizeof(CudaScene::BVHNode));
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
      cudaMalloc((void**)&host_scene.materials, host_scene.num_materials * sizeof(CudaScene::Material));
      cudaMemcpy(host_scene.materials, host_materials, host_scene.num_materials * sizeof(CudaScene::Material),
                 cudaMemcpyHostToDevice);
   }
   else
   {
      host_scene.materials = nullptr;
   }

   if (host_scene.num_geometries > 0)
   {
      cudaMalloc((void**)&host_scene.geometries, host_scene.num_geometries * sizeof(CudaScene::Geometry));
      cudaMemcpy(host_scene.geometries, host_geometries, host_scene.num_geometries * sizeof(CudaScene::Geometry),
                 cudaMemcpyHostToDevice);
   }
   else
   {
      host_scene.geometries = nullptr;
   }

   if (host_scene.num_meshes > 0)
   {
      cudaMalloc((void**)&host_scene.meshes, host_scene.num_meshes * sizeof(CudaScene::Mesh));
      cudaMemcpy(host_scene.meshes, host_meshes, host_scene.num_meshes * sizeof(CudaScene::Mesh), cudaMemcpyHostToDevice);
   }
   else
   {
      host_scene.meshes = nullptr;
   }

   // Free host arrays
   delete[] host_materials;
   delete[] host_geometries;
   if (host_meshes) delete[] host_meshes;

// Diagnostic: Print scene transfer summary
#ifdef DIAGS
   printf("\n- GPU Scene Transfer Summary:\n");
   printf("   Materials: %d\n", host_scene.num_materials);
   printf("   Geometries: %d\n", host_scene.num_geometries);
   printf("   Meshes: %d\n", host_scene.num_meshes);
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
   cudaMalloc((void**)&d_scene, sizeof(CudaScene::Scene));
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
   
   // Free meshes
   if (host_scene.meshes)
   {
      // We need to retrieve the meshes array from device to free their internal pointers
      CudaScene::Mesh *host_meshes = new CudaScene::Mesh[host_scene.num_meshes];
      cudaMemcpy(host_meshes, host_scene.meshes, host_scene.num_meshes * sizeof(CudaScene::Mesh), cudaMemcpyDeviceToHost);
      
      for (int i = 0; i < host_scene.num_meshes; ++i)
      {
         if (host_meshes[i].triangles) cudaFree(host_meshes[i].triangles);
         if (host_meshes[i].bvh_nodes) cudaFree(host_meshes[i].bvh_nodes);
      }
      
      delete[] host_meshes;
      cudaFree(host_scene.meshes);
   }

   // Free the scene struct itself (now on device)
   cudaFree(d_scene);
}

} // namespace Scene
