/**
 * @file scene_builder.hpp
 * @brief Converts SceneDescription to renderer-specific formats
 *
 * This file provides builders that convert the unified SceneDescription format to:
 * 1. CPU format: Hittable_list with polymorphic objects
 * 2. GPU format: Flat arrays in device memory
 */

#pragma once

#include "scene_description.hpp"

#include "hittable_list.hpp"
#include "material.hpp"

#include "cpu_shapes/rectangle.hpp"
#include "cpu_shapes/sdf_shape.hpp"
#include "cpu_shapes/sphere.hpp"
#include <memory>

using std::make_shared;
using std::shared_ptr;

// Forward declaration for CUDA scene
namespace CudaScene
{
struct Scene;
}

namespace Scene
{

//==============================================================================
// CPU SCENE BUILDER
//==============================================================================
class CPUSceneBuilder
{
 public:
   /**
    * @brief Convert SceneDescription to CPU-compatible Hittable_list
    * @param desc Scene description to convert
    * @return Hittable_list containing polymorphic geometry objects
    */
   static Hittable_list buildCPUScene(const SceneDescription &desc)
   {
      Hittable_list scene;

      // First, create all materials
      std::vector<shared_ptr<Material>> cpu_materials;
      cpu_materials.reserve(desc.materials.size());

      for (const auto &mat_desc : desc.materials)
      {
         cpu_materials.push_back(createMaterial(mat_desc));
      }

      // Then create all geometries with material references
      for (const auto &geom_desc : desc.geometries)
      {
         if (geom_desc.material_id < 0 || geom_desc.material_id >= static_cast<int>(cpu_materials.size()))
         {
            continue; // Skip invalid material IDs
         }

         shared_ptr<Material> mat = cpu_materials[geom_desc.material_id];
         shared_ptr<Hittable> geom = createGeometry(geom_desc, mat);

         if (geom)
         {
            scene.add(geom);
         }
      }

      return scene;
   }

 private:
   /**
    * @brief Create a CPU Material from MaterialDesc
    */
   static shared_ptr<Material> createMaterial(const MaterialDesc &desc)
   {
      switch (desc.type)
      {
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
   static shared_ptr<Hittable> createGeometry(const GeometryDesc &desc, shared_ptr<Material> mat)
   {
      switch (desc.type)
      {
      case GeometryType::SPHERE:
         return make_shared<Sphere>(desc.data.sphere.center, desc.data.sphere.radius, mat);

      case GeometryType::RECTANGLE:
         return make_shared<Rectangle>(desc.data.rectangle.corner, desc.data.rectangle.u, desc.data.rectangle.v, mat);

      case GeometryType::SDF_PRIMITIVE:
         return createSDFShape(desc, mat);

      // Other geometry types not yet supported in CPU renderer
      case GeometryType::CUBE:
      case GeometryType::DISPLACED_SPHERE:
      case GeometryType::TRIANGLE:
      case GeometryType::TRIANGLE_MESH:
      default:
         // Return null for unsupported types
         return nullptr;
      }
   }

   /**
    * @brief Create SDF shape from GeometryDesc
    */
   static shared_ptr<Hittable> createSDFShape(const GeometryDesc &desc, shared_ptr<Material> mat)
   {
      const auto &sdf_data = desc.data.sdf;
      const Vec3 &rotation = sdf_data.rotation;

      switch (sdf_data.sdf_type)
      {
      case SDFType::SPHERE:
         return SDFShape::createSphere(sdf_data.position,
                                       sdf_data.parameters.x(), // radius
                                       mat, rotation);

      case SDFType::BOX:
         return SDFShape::createBox(sdf_data.position,
                                    sdf_data.parameters, // half-extents (size)
                                    mat, rotation);

      case SDFType::TORUS:
         return SDFShape::createTorus(sdf_data.position,
                                      sdf_data.parameters.x(), // major radius
                                      sdf_data.parameters.y(), // minor radius
                                      mat, rotation);

      case SDFType::CAPSULE:
      {
         // For capsule, we need start and end points
         // parameters.x = radius, parameters.y = height
         Vec3 a = sdf_data.position - Vec3(0, sdf_data.parameters.y() * 0.5, 0);
         Vec3 b = sdf_data.position + Vec3(0, sdf_data.parameters.y() * 0.5, 0);
         return SDFShape::createCapsule(a, b, sdf_data.parameters.x(), mat, rotation);
      }

      case SDFType::CYLINDER:
         return SDFShape::createCylinder(sdf_data.position,
                                         sdf_data.parameters.y(), // height
                                         sdf_data.parameters.x(), // radius
                                         mat, rotation);

      case SDFType::PLANE:
         return SDFShape::createPlane(Vec3(0, 1, 0),           // normal (default: up)
                                      sdf_data.parameters.x(), // distance from origin
                                      mat, rotation);

      case SDFType::MANDELBULB:
         return SDFShape::createMandelbulb(sdf_data.position,
                                           sdf_data.parameters.x(),                   // power (typically 8)
                                           static_cast<int>(sdf_data.parameters.y()), // iterations
                                           mat, rotation);

      case SDFType::DEATH_STAR:
         return SDFShape::createDeathStar(sdf_data.position,
                                          sdf_data.parameters.x(), // main radius
                                          sdf_data.parameters.y(), // cutout radius
                                          sdf_data.parameters.z(), // cutout distance
                                          mat, rotation);

      case SDFType::CUT_HOLLOW_SPHERE:
         return SDFShape::createCutHollowSphere(sdf_data.position,
                                                sdf_data.parameters.x(), // radius
                                                sdf_data.parameters.y(), // cut height
                                                sdf_data.parameters.z(), // thickness
                                                mat, rotation);

      case SDFType::OCTAHEDRON:
         return SDFShape::createOctahedron(sdf_data.position,
                                           sdf_data.parameters.x(), // size
                                           mat, rotation);

      case SDFType::PYRAMID:
         return SDFShape::createPyramid(sdf_data.position,
                                        sdf_data.parameters.x(), // height
                                        mat, rotation);

      case SDFType::CUSTOM:
      default:
         // Custom SDFs not yet supported
         return nullptr;
      }
   }
};

//==============================================================================
// CUDA SCENE BUILDER - Declared here, implemented in scene_builder_cuda.cu
//==============================================================================

class CudaSceneBuilder
{
 public:
   static CudaScene::Scene *buildGPUScene(const SceneDescription &desc);
   static void freeGPUScene(CudaScene::Scene *scene);
};

} // namespace Scene
