#pragma once

#include "scene_description.hpp"
#include "yaml_scene_loader.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;

namespace Scene
{

/**
 * @brief Factory class for creating SceneDescription from various sources
 */
class SceneFactory
{
 public:
   /**
    * @brief Load scene from file with fallback to default
    *
    * @param filename Path to YAML scene file
    * @return SceneDescription Loaded scene, or default scene if loading fails
    */
   static SceneDescription fromYAML(const std::string &filename)
   {      
      std::cout << "Loading scene from: " << filename << std::endl;

      try
      {
         std::cout << "Loading scene from: " << filename << std::endl;
         SceneDescription scene_desc;
         if (loadSceneFromYAML(filename.c_str(), scene_desc))
         {
            // Build BVH if enabled in scene
            if (scene_desc.use_bvh)
            {
               cout << "Building BVH acceleration structure..." "\n";
               scene_desc.buildBVH();
               cout << "BVH built with " << scene_desc.top_level_bvh.nodes.size() << " nodes" "\n";
            }
            return scene_desc; // Successfully loaded
         }
         else
         {
            throw std::runtime_error("YAML scene loading failed");
         }
      }
      catch (const std::exception &e)
      {
         std::cerr << "Failed to load scene from '" << filename << "': " << e.what() << std::endl;
         std::cerr << "Warning: Falling back to default scene" << std::endl;
         return createDefaultScene();
      }
   }

   /**
    * @brief Create a simple scene with a single red sphere
    *
    * @return SceneDescription Simple scene with one sphere and ground
    */
   static SceneDescription singleObjectScene()
   {
      using namespace Scene;
      SceneDescription scene_desc;

      scene_desc.camera_position = Vec3(0, 0.6, 2);
      scene_desc.camera_look_at = Vec3(0, 0.3, -1);
      scene_desc.camera_up = Vec3(0, 1, 0);
      scene_desc.camera_fov = 40.0f;

      int mat_red = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(0.9, 0.1, 0.1)));
      int mat_grey = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(0.3, 0.3, 0.3)));
      scene_desc.addSphere(Vec3(0, 0.6, -1), 0.5, mat_red);
      scene_desc.addSphere(Vec3(0, -10000, -1), 10000, mat_grey); // Ground
      return scene_desc;
   }

   /**
    * @brief Create the default programmatic scene
    *
    * @return SceneDescription Default scene with spheres, area lights, and various materials
    */
   static SceneDescription createDefaultScene()
   {
      cout << "Creating default scene..." "\n";
      using namespace Scene;
      SceneDescription scene_desc;

      // Camera
      scene_desc.camera_position = Vec3(-2, 2, 5);
      scene_desc.camera_look_at = Vec3(-2, -0.5, -1);
      scene_desc.camera_up = Vec3(0, 1, 0);
      scene_desc.camera_fov = 35.0f;

      // === Default scene - Materials ===
      int mat_ground = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(0.44, 0.7, 0.95)));
      int mat_golden = scene_desc.addMaterial(MaterialDesc::roughMirror(Vec3(1.0, 0.85, 0.47), 0.03));
      int mat_blue_rough = scene_desc.addMaterial(MaterialDesc::roughMirror(Vec3(0.3, 0.3, 0.91), 0.3));
      int mat_red_dots = scene_desc.addMaterial(MaterialDesc::fibonacciDots(Vec3(0.9, 0.1, 0.1), Vec3(0.02, 0.02, 0.02), 12, 0.33f));
      int mat_glass = scene_desc.addMaterial(MaterialDesc::glass(1.5));
      int mat_yellow = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(247 / 255.0, 241 / 255.0, 159 / 255.0)));
      int mat_blue = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(140 / 255.0, 198 / 255.0, 230 / 255.0)));
      int mat_violet = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(168 / 255.0, 144 / 255.0, 192 / 255.0)));
      int mat_rose = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(226 / 255.0, 171 / 255.0, 186 / 255.0)));
      int mat_green = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(152 / 255.0, 199 / 255.0, 191 / 255.0)));
      int mat_light = scene_desc.addMaterial(MaterialDesc::light(Vec3(4.8, 4.1, 3.7)));
      int mat_torus_orange = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(1.0, 0.6, 0.2)));
      int mat_normal = scene_desc.addMaterial(MaterialDesc::normal());

      // === Default scene - Geometry ===
      scene_desc.addSphere(Vec3(0, -950.5, -1), 950.0, mat_ground); // Ground "plane"
      scene_desc.addSphere(Vec3(-3.5, 0.45, -1.8), 0.8, mat_golden);
      scene_desc.addDisplacedSphere(Vec3(1.2, 0, -2), 0.5, mat_blue_rough, 0.2f, 0);
      scene_desc.addSphere(Vec3(-1.3, 0.18, -5), 0.7, mat_red_dots);
      scene_desc.addSphere(Vec3(-0.7, 0.2, -0.3), 0.6, mat_glass);

      // ISC spheres
      scene_desc.addSphere(Vec3(-3.5, -0.3, 1.2), 0.2, mat_yellow);
      scene_desc.addSphere(Vec3(-3.0, -0.3, 1.2), 0.2, mat_blue);
      scene_desc.addSphere(Vec3(-2.5, -0.3, 1.2), 0.2, mat_violet);
      scene_desc.addSphere(Vec3(-2.0, -0.3, 1.2), 0.2, mat_rose);
      scene_desc.addSphere(Vec3(-1.5, -0.3, 1.2), 0.2, mat_green);

      scene_desc.addRectangle(Vec3(-1.0, 3.0, -2.0), Vec3(2.5, 0, 0), Vec3(0, 0, 1.5), mat_light);

      // === NEW SDF SHAPES from Íñigo Quilez's distance functions ===
      // All SDF shapes now support rotation! Last parameter is Vec3(rotX, rotY, rotZ) in radians
      // Rotation is applied in X, Y, Z order using Euler angles
      // Example: Vec3(M_PI * 0.25, 0, 0) rotates 45 degrees around X axis

      // Materials for new SDF shapes
      int mat_death_star = scene_desc.addMaterial(MaterialDesc::metal(Vec3(0.6, 0.6, 0.65), 0.1));
      int mat_hollow_sphere = scene_desc.addMaterial(MaterialDesc::roughMirror(Vec3(0.9, 0.7, 0.3), 0.2));
      int mat_octahedron = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(0.2, 0.8, 0.4)));
      int mat_pyramid = scene_desc.addMaterial(MaterialDesc::roughMirror(Vec3(0.8, 0.3, 0.3), 0.15));

      // Death Star - positioned at back left, rotated to show cutout
      scene_desc.addSDFDeathStar(Vec3(-3.5, 1.2, -4.5), 0.8, 0.5, 1.0, mat_death_star, Vec3(0, M_PI * 0.3, 0));

      // Cut Hollow Sphere - positioned center back, tilted for better view
      scene_desc.addSDFCutHollowSphere(Vec3(0.0, 0.8, -5.0), 0.7, 0.3, 0.1, mat_hollow_sphere, Vec3(M_PI * 0.15, 0, 0));

      // Octahedron - positioned at front right, rotated 45 degrees
      scene_desc.addSDFOctahedron(Vec3(2.5, 0.5, -2.0), 0.6, mat_octahedron, Vec3(0, M_PI * 0.25, M_PI * 0.25));

      // Pyramid - positioned at right back, rotated to face camera
      scene_desc.addSDFPyramid(Vec3(3.0, 0.0, -4.0), 0.8, mat_pyramid, Vec3(0, M_PI * 0.4, 0));

      // Original SDF Torus - rotated to show hole better
      scene_desc.addSDFTorus(Vec3(1.5, 0.7, -3.5), 0.6, 0.2, mat_torus_orange, Vec3(M_PI * 0.3, M_PI * 0.2, 0));

      // // Add many more spheres to test BVH performance
      // for (int i = 0; i < 10; i++)
      // {
      //    for (int j = 0; j < 10; j++)
      //    {
      //       double x = -4.5 + i * 1.0;
      //       double z = -8.0 + j * 1.0;
      //       int mat = (i + j) % 4;
      //       int material = mat == 0 ? mat_yellow : (mat == 1 ? mat_blue : (mat == 2 ? mat_violet : mat_rose));
      //       scene_desc.addSphere(Vec3(x, -0.4, z), 0.15, material);
      //    }
      // }

      // Build BVH for default scene (always enabled for default)
      scene_desc.use_bvh = true;
      scene_desc.buildBVH();
      cout << "Built BVH with " << scene_desc.top_level_bvh.nodes.size() << " nodes for "
           << scene_desc.geometries.size() << " geometries" "\n";

      return scene_desc;
   }

   /**
    * @brief Create a simple Cornell Box scene
    *
    * @return SceneDescription Classic Cornell Box for validation
    */
   static SceneDescription createCornellBox();

   /**
    * @brief Create a minimal test scene (3 spheres)
    *
    * @return SceneDescription Simple scene for quick testing
    */
   static SceneDescription createSimpleScene();

 private:
   SceneFactory() = delete; // Static class - no instantiation
};

} // namespace Scene