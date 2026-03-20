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
#include "cpu_shapes/triangle.hpp"
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
// CPU SCENE RESULT
//==============================================================================

/**
 * @brief Combined CPU scene: the full scene plus a separate list of emissive objects.
 *
 * The lights list contains the same shared_ptr<Hittable> objects that are also
 * in the main scene.  It is used exclusively by the MIS path tracer for Next
 * Event Estimation (NEE) — shadow rays are tested against the full scene.
 */
struct CPUScene
{
   Hittable_list scene;  ///< All geometry
   Hittable_list lights; ///< Only emissive geometry (for NEE light sampling)
};

//==============================================================================
// CPU SCENE BUILDER
//==============================================================================
class CPUSceneBuilder
{
 public:
   /**
    * @brief Convert SceneDescription to CPU-compatible CPUScene
    * @param desc Scene description to convert
    * @return CPUScene containing all geometry and a separate emissive-only list
    */
   static CPUScene buildCPUScene(const SceneDescription &desc)
   {
      CPUScene result;

      // Build all materials
      std::vector<shared_ptr<Material>> cpu_materials;
      cpu_materials.reserve(desc.materials.size());
      for (const auto &mat_desc : desc.materials)
         cpu_materials.push_back(createMaterial(mat_desc));

      // Build geometry, tracking which objects are emissive
      for (const auto &geom_desc : desc.geometries)
      {
         if (geom_desc.material_id < 0 || geom_desc.material_id >= static_cast<int>(cpu_materials.size()))
            continue;

         shared_ptr<Material> mat  = cpu_materials[geom_desc.material_id];
         shared_ptr<Hittable> geom = createGeometry(geom_desc, mat);

         if (!geom)
            continue;

         result.scene.add(geom);

         // Add to light list if material emits
         if (desc.materials[geom_desc.material_id].type == MaterialType::LIGHT)
            result.lights.add(geom);
      }

      return result;
   }

 private:
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

      case MaterialType::LIGHT:
         // Use emission if set, otherwise fall back to albedo as a warm white
         {
            Vec3 em = desc.emission;
            if (em.length_squared() < 1e-8)
               em = desc.albedo * 5.0; // sensible default: bright albedo
            return make_shared<Light>(em);
         }

      case MaterialType::THIN_FILM:
         return make_shared<ThinFilm>(desc.film_thickness, desc.film_ior);

      // Materials not yet CPU-implemented — fall back to Lambertian
      case MaterialType::METAL:
      case MaterialType::MIRROR:
      case MaterialType::ROUGH_MIRROR:
      case MaterialType::GLASS:
      case MaterialType::DIELECTRIC:
      case MaterialType::SDF_MATERIAL:
      case MaterialType::ANISOTROPIC_METAL:
      case MaterialType::CLEAR_COAT:
      default:
         return make_shared<Lambertian>(desc.albedo);
      }
   }

   static shared_ptr<Hittable> createGeometry(const GeometryDesc &desc, shared_ptr<Material> mat)
   {
      switch (desc.type)
      {
      case GeometryType::SPHERE:
         return make_shared<Sphere>(desc.data.sphere.center, desc.data.sphere.radius, mat);

      case GeometryType::RECTANGLE:
         return make_shared<Rectangle>(desc.data.rectangle.corner, desc.data.rectangle.u,
                                       desc.data.rectangle.v, mat);

      case GeometryType::SDF_PRIMITIVE:
         return createSDFShape(desc, mat);

      case GeometryType::TRIANGLE:
         if (desc.data.triangle.has_normals)
            return make_shared<TriangleShape>(desc.data.triangle.v0, desc.data.triangle.v1,
                                             desc.data.triangle.v2, desc.data.triangle.n0,
                                             desc.data.triangle.n1, desc.data.triangle.n2, mat);
         else
            return make_shared<TriangleShape>(desc.data.triangle.v0, desc.data.triangle.v1,
                                             desc.data.triangle.v2, mat);

      case GeometryType::CUBE:
      case GeometryType::DISPLACED_SPHERE:
      case GeometryType::TRIANGLE_MESH:
      default:
         return nullptr;
      }
   }

   static shared_ptr<Hittable> createSDFShape(const GeometryDesc &desc, shared_ptr<Material> mat)
   {
      const auto &sdf_data = desc.data.sdf;
      const Vec3 &rotation = sdf_data.rotation;

      switch (sdf_data.sdf_type)
      {
      case SDFType::SPHERE:
         return SDFShape::createSphere(sdf_data.position, sdf_data.parameters.x(), mat, rotation);
      case SDFType::BOX:
         return SDFShape::createBox(sdf_data.position, sdf_data.parameters, mat, rotation);
      case SDFType::TORUS:
         return SDFShape::createTorus(sdf_data.position, sdf_data.parameters.x(), sdf_data.parameters.y(), mat,
                                      rotation);
      case SDFType::CAPSULE:
      {
         Vec3 a = sdf_data.position - Vec3(0, sdf_data.parameters.y() * 0.5, 0);
         Vec3 b = sdf_data.position + Vec3(0, sdf_data.parameters.y() * 0.5, 0);
         return SDFShape::createCapsule(a, b, sdf_data.parameters.x(), mat, rotation);
      }
      case SDFType::CYLINDER:
         return SDFShape::createCylinder(sdf_data.position, sdf_data.parameters.y(), sdf_data.parameters.x(),
                                         mat, rotation);
      case SDFType::PLANE:
         return SDFShape::createPlane(Vec3(0, 1, 0), sdf_data.parameters.x(), mat, rotation);
      case SDFType::MANDELBULB:
         return SDFShape::createMandelbulb(sdf_data.position, sdf_data.parameters.x(),
                                           static_cast<int>(sdf_data.parameters.y()), mat, rotation);
      case SDFType::DEATH_STAR:
         return SDFShape::createDeathStar(sdf_data.position, sdf_data.parameters.x(), sdf_data.parameters.y(),
                                          sdf_data.parameters.z(), mat, rotation);
      case SDFType::CUT_HOLLOW_SPHERE:
         return SDFShape::createCutHollowSphere(sdf_data.position, sdf_data.parameters.x(),
                                                sdf_data.parameters.y(), sdf_data.parameters.z(), mat, rotation);
      case SDFType::OCTAHEDRON:
         return SDFShape::createOctahedron(sdf_data.position, sdf_data.parameters.x(), mat, rotation);
      case SDFType::PYRAMID:
         return SDFShape::createPyramid(sdf_data.position, sdf_data.parameters.x(), mat, rotation);
      case SDFType::CUSTOM:
      default:
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
