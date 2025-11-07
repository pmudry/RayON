/**
 * @file scene_builder_cuda.cu
 * @brief CUDA implementation of scene builder
 * 
 * This file contains the CUDA-specific scene building code that converts
 * host SceneDescription to GPU-friendly format and manages device memory.
 */

#include "../scene_description.h"
#include "cuda_scene.cuh"
#include <cuda_runtime.h>

namespace Scene {

// Declare CudaSceneBuilder class here to avoid pulling in scene_builder.h
class CudaSceneBuilder {
public:
    static CudaScene::Scene* buildGPUScene(const SceneDescription& desc);
    static void freeGPUScene(CudaScene::Scene* scene);
};



/**
 * @brief Convert MaterialDesc to GPU Material
 */
static CudaScene::Material convertMaterial(const MaterialDesc& desc) {
    CudaScene::Material mat;
    
    // Convert material type
    mat.type = static_cast<CudaScene::MaterialType>(static_cast<uint8_t>(desc.type));
    
    // Convert vectors
    mat.albedo = float3_simple(
        static_cast<float>(desc.albedo.x()),
        static_cast<float>(desc.albedo.y()),
        static_cast<float>(desc.albedo.z())
    );
    mat.emission = float3_simple(
        static_cast<float>(desc.emission.x()),
        static_cast<float>(desc.emission.y()),
        static_cast<float>(desc.emission.z())
    );
    
    // Copy scalar properties
    mat.roughness = desc.roughness;
    mat.metallic = desc.metallic;
    mat.refractive_index = desc.refractive_index;
    mat.transmission = desc.transmission;
    mat.texture_id = desc.texture_id;
    
    // Copy pattern information
    mat.pattern = static_cast<CudaScene::ProceduralPattern>(static_cast<uint8_t>(desc.pattern));
    mat.pattern_color = float3_simple(
        static_cast<float>(desc.pattern_color.x()),
        static_cast<float>(desc.pattern_color.y()),
        static_cast<float>(desc.pattern_color.z())
    );
    mat.pattern_param1 = desc.pattern_param1;
    mat.pattern_param2 = desc.pattern_param2;
    
    return mat;
}

/**
 * @brief Convert GeometryDesc to GPU Geometry
 */
static CudaScene::Geometry convertGeometry(const GeometryDesc& desc) {
    CudaScene::Geometry geom;
    
    // Convert geometry type
    geom.type = static_cast<CudaScene::GeometryType>(static_cast<uint8_t>(desc.type));
    geom.material_id = desc.material_id;
    
    // Convert bounding box
    geom.bounds_min = float3_simple(
        static_cast<float>(desc.bounds_min.x()),
        static_cast<float>(desc.bounds_min.y()),
        static_cast<float>(desc.bounds_min.z())
    );
    geom.bounds_max = float3_simple(
        static_cast<float>(desc.bounds_max.x()),
        static_cast<float>(desc.bounds_max.y()),
        static_cast<float>(desc.bounds_max.z())
    );
    
    // Convert geometry-specific data
    switch(desc.type) {
        case GeometryType::SPHERE:
            geom.data.sphere.center = float3_simple(
                static_cast<float>(desc.data.sphere.center.x()),
                static_cast<float>(desc.data.sphere.center.y()),
                static_cast<float>(desc.data.sphere.center.z())
            );
            geom.data.sphere.radius = static_cast<float>(desc.data.sphere.radius);
            break;
            
        case GeometryType::RECTANGLE:
            geom.data.rectangle.corner = float3_simple(
                static_cast<float>(desc.data.rectangle.corner.x()),
                static_cast<float>(desc.data.rectangle.corner.y()),
                static_cast<float>(desc.data.rectangle.corner.z())
            );
            geom.data.rectangle.u = float3_simple(
                static_cast<float>(desc.data.rectangle.u.x()),
                static_cast<float>(desc.data.rectangle.u.y()),
                static_cast<float>(desc.data.rectangle.u.z())
            );
            geom.data.rectangle.v = float3_simple(
                static_cast<float>(desc.data.rectangle.v.x()),
                static_cast<float>(desc.data.rectangle.v.y()),
                static_cast<float>(desc.data.rectangle.v.z())
            );
            break;
            
        case GeometryType::DISPLACED_SPHERE:
            geom.data.displaced_sphere.center = float3_simple(
                static_cast<float>(desc.data.displaced_sphere.center.x()),
                static_cast<float>(desc.data.displaced_sphere.center.y()),
                static_cast<float>(desc.data.displaced_sphere.center.z())
            );
            geom.data.displaced_sphere.radius = static_cast<float>(desc.data.displaced_sphere.radius);
            geom.data.displaced_sphere.displacement_scale = desc.data.displaced_sphere.displacement_scale;
            geom.data.displaced_sphere.pattern_type = desc.data.displaced_sphere.pattern_type;
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
CudaScene::Scene* CudaSceneBuilder::buildGPUScene(const SceneDescription& desc) {
    CudaScene::Scene* scene = new CudaScene::Scene();
    
    // Allocate host arrays
    int num_materials = static_cast<int>(desc.materials.size());
    int num_geometries = static_cast<int>(desc.geometries.size());
    
    CudaScene::Material* host_materials = new CudaScene::Material[num_materials];
    CudaScene::Geometry* host_geometries = new CudaScene::Geometry[num_geometries];
    
    // Convert materials
    for (int i = 0; i < num_materials; ++i) {
        host_materials[i] = convertMaterial(desc.materials[i]);
    }
    
    // Convert geometries
    for (int i = 0; i < num_geometries; ++i) {
        host_geometries[i] = convertGeometry(desc.geometries[i]);
    }
    
    // Set scene properties
    scene->num_materials = num_materials;
    scene->num_geometries = num_geometries;
    scene->num_bvh_nodes = 0;  // BVH not yet implemented
    scene->bvh_root_idx = -1;
    scene->use_bvh = false;
    scene->max_ray_march_steps = 100;
    scene->ray_march_epsilon = 0.001f;
    
    // Allocate device memory and copy
    if (scene->num_materials > 0) {
        cudaMalloc(&scene->materials, scene->num_materials * sizeof(CudaScene::Material));
        cudaMemcpy(scene->materials, host_materials,
                  scene->num_materials * sizeof(CudaScene::Material),
                  cudaMemcpyHostToDevice);
    } else {
        scene->materials = nullptr;
    }
    
    if (scene->num_geometries > 0) {
        cudaMalloc(&scene->geometries, scene->num_geometries * sizeof(CudaScene::Geometry));
        cudaMemcpy(scene->geometries, host_geometries,
                  scene->num_geometries * sizeof(CudaScene::Geometry),
                  cudaMemcpyHostToDevice);
    } else {
        scene->geometries = nullptr;
    }
    
    // BVH not yet implemented
    scene->bvh_nodes = nullptr;
    
    // Free host arrays
    delete[] host_materials;
    delete[] host_geometries;
    
    return scene;
}

/**
 * @brief Free GPU scene memory
 */
void CudaSceneBuilder::freeGPUScene(CudaScene::Scene* scene) {
    if (!scene) return;
    
    if (scene->materials) {
        cudaFree(scene->materials);
        scene->materials = nullptr;
    }
    if (scene->geometries) {
        cudaFree(scene->geometries);
        scene->geometries = nullptr;
    }
    if (scene->bvh_nodes) {
        cudaFree(scene->bvh_nodes);
        scene->bvh_nodes = nullptr;
    }
    
    delete scene;
}

} // namespace Scene
