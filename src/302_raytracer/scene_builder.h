/**
 * @file scene_builder.h
 * @brief Converts SceneDescription to renderer-specific formats
 * 
 * This file provides builders that convert the unified SceneDescription format to:
 * 1. CPU format: Hittable_list with polymorphic objects
 * 2. GPU format: Flat arrays in device memory
 */

#pragma once

#include "scene_description.h"
#include "hittable_list.h"
#include "sphere.h"
#include "rectangle.h"
#include "material.h"
#include <memory>

using std::shared_ptr;
using std::make_shared;

// Forward declaration for CUDA scene
namespace CudaScene {
    struct Scene;
}

namespace Scene {

//==============================================================================
// CPU SCENE BUILDER
//==============================================================================

class CPUSceneBuilder {
public:
    /**
     * @brief Convert SceneDescription to CPU-compatible Hittable_list
     * @param desc Scene description to convert
     * @return Hittable_list containing polymorphic geometry objects
     */
    static Hittable_list buildCPUScene(const SceneDescription& desc) {
        Hittable_list scene;
        
        // First, create all materials
        std::vector<shared_ptr<Material>> cpu_materials;
        cpu_materials.reserve(desc.materials.size());
        
        for (const auto& mat_desc : desc.materials) {
            cpu_materials.push_back(createMaterial(mat_desc));
        }
        
        // Then create all geometries with material references
        for (const auto& geom_desc : desc.geometries) {
            if (geom_desc.material_id < 0 || geom_desc.material_id >= static_cast<int>(cpu_materials.size())) {
                continue; // Skip invalid material IDs
            }
            
            shared_ptr<Material> mat = cpu_materials[geom_desc.material_id];
            shared_ptr<Hittable> geom = createGeometry(geom_desc, mat);
            
            if (geom) {
                scene.add(geom);
            }
        }
        
        return scene;
    }
    
private:
    /**
     * @brief Create a CPU Material from MaterialDesc
     */
    static shared_ptr<Material> createMaterial(const MaterialDesc& desc) {
        switch(desc.type) {
            case MaterialType::LAMBERTIAN:
                return make_shared<Lambertian>(desc.albedo);
                
            case MaterialType::CONSTANT:
                return make_shared<Constant>(desc.albedo);
                
            case MaterialType::SHOW_NORMALS:
                return make_shared<ShowNormals>(desc.albedo);
                
            // For materials not yet implemented in CPU renderer, use Lambertian as fallback
            case MaterialType::METAL:
            case MaterialType::MIRROR:
            case MaterialType::ROUGH_MIRROR:
            case MaterialType::GLASS:
            case MaterialType::DIELECTRIC:
            case MaterialType::LIGHT:
            case MaterialType::SDF_MATERIAL:
            default:
                // Use Lambertian as fallback
                return make_shared<Lambertian>(desc.albedo);
        }
    }
    
    /**
     * @brief Create CPU Hittable geometry from GeometryDesc
     */
    static shared_ptr<Hittable> createGeometry(const GeometryDesc& desc, shared_ptr<Material> mat) {
        switch(desc.type) {
            case GeometryType::SPHERE:
                return make_shared<Sphere>(
                    desc.data.sphere.center,
                    desc.data.sphere.radius,
                    mat
                );
                
            case GeometryType::RECTANGLE:
                // Note: Rectangle class needs modification to accept Material pointer
                // For now, create a basic rectangle
                return make_shared<Rectangle>(
                    desc.data.rectangle.corner,
                    desc.data.rectangle.u,
                    desc.data.rectangle.v
                );
                
            // Other geometry types not yet supported in CPU renderer
            case GeometryType::CUBE:
            case GeometryType::DISPLACED_SPHERE:
            case GeometryType::TRIANGLE:
            case GeometryType::TRIANGLE_MESH:
            case GeometryType::SDF_PRIMITIVE:
            default:
                // Return null for unsupported types
                return nullptr;
        }
    }
};

//==============================================================================
// CUDA SCENE BUILDER - Declared here, implemented in scene_builder_cuda.cu
//==============================================================================

class CudaSceneBuilder {
public:
    static CudaScene::Scene* buildGPUScene(const SceneDescription& desc);
    static void freeGPUScene(CudaScene::Scene* scene);
};

} // namespace Scene
