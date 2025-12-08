#pragma once

#include "scene_description.hpp"
#include "yaml_scene_loader.hpp"
#include "external/tiny_obj_loader.h" // Include tiny_obj_loader
#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cctype>
#include <limits> // For numeric_limits

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
    * @brief Load scene from any supported file format (.yaml, .obj)
    *
    * @param filename Path to the scene file
    * @return SceneDescription Loaded scene
    */
   static SceneDescription load(const std::string &filename)
   {
      std::string ext = filename.substr(filename.find_last_of(".") + 1);
      
      // Convert to lowercase for case-insensitive comparison
      std::transform(ext.begin(), ext.end(), ext.begin(),
                     [](unsigned char c){ return std::tolower(c); });

      if (ext == "yaml" || ext == "yml")
      {
         return fromYAML(filename);
      }
      else if (ext == "obj")
      {
         return fromOBJ(filename);
      }
      else
      {
         std::cerr << "Unknown file format: " << ext << ". Loading default scene." << std::endl;
         return createDefaultScene();
      }
   }

   /**
    * @brief Load scene from OBJ file
    *
    * @param filename Path to OBJ file
    * @return SceneDescription Loaded scene
    */
   static SceneDescription fromOBJ(const std::string &filename)
   {
      std::cout << "Loading OBJ scene from: " << filename << std::endl;

      tinyobj::attrib_t attrib;
      std::vector<tinyobj::shape_t> shapes;
      std::vector<tinyobj::material_t> materials;
      std::string warn, err;

      std::string mtl_dir;
      size_t pos = filename.find_last_of("/\\\\");
      if (pos != std::string::npos) {
          mtl_dir = filename.substr(0, pos) + "/";
      }

      tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str(), mtl_dir.c_str());

      if (!warn.empty()) {
          std::cerr << "TinyObjLoader Warning: " << warn << std::endl;
      }
      if (!err.empty()) {
          std::cerr << "TinyObjLoader Error: " << err << std::endl;
          return createDefaultScene(); // Fallback to default on error
      }
      if (shapes.empty()) {
          std::cerr << "No shapes found in OBJ file. Loading default scene." << std::endl;
          return createDefaultScene();
      }

      SceneDescription scene_desc;

      // 1. Convert materials
      std::map<int, int> material_id_map; // tinyobj_mat_idx -> scene_mat_idx
      if (materials.empty()) {
          // Add a default material if none specified in OBJ/MTL
          int default_mat_id = scene_desc.addMaterial(MaterialDesc::lambertian(Vec3(0.7, 0.7, 0.7)));
          material_id_map[-1] = default_mat_id; // Map default OBJ material to this
      } else {
          for (size_t i = 0; i < materials.size(); ++i) {
              const auto& tinyobj_mat = materials[i];
              MaterialDesc scene_mat;

              // Basic Phong to PBR mapping (simplistic)
              // Diffuse -> Albedo (Lambertian)
              scene_mat.albedo = Vec3(tinyobj_mat.diffuse[0], tinyobj_mat.diffuse[1], tinyobj_mat.diffuse[2]);
              scene_mat.type = MaterialType::LAMBERTIAN; // Default to Lambertian

              // Specular/Shininess -> Metal/Roughness
              if (tinyobj_mat.specular[0] > 0 || tinyobj_mat.shininess > 0) {
                  scene_mat.type = MaterialType::METAL;
                  scene_mat.metallic = 1.0f;
                  // Roughness from shininess (Ns)
                  // Ns is typically 0-1000, roughness 0-1
                  scene_mat.roughness = 1.0f - std::min(1.0f, tinyobj_mat.shininess / 1000.0f);
              }
              
              // Transmittance/IOR -> Glass
              if (tinyobj_mat.transmittance[0] > 0 || tinyobj_mat.ior > 1.0f) {
                  scene_mat.type = MaterialType::GLASS;
                  scene_mat.refractive_index = tinyobj_mat.ior > 1.0f ? tinyobj_mat.ior : 1.5f;
              }

              // Emission
              scene_mat.emission = Vec3(tinyobj_mat.emission[0], tinyobj_mat.emission[1], tinyobj_mat.emission[2]);
              if (scene_mat.emission.x() > 0 || scene_mat.emission.y() > 0 || scene_mat.emission.z() > 0) {
                  scene_mat.type = MaterialType::LIGHT; // Override to light if emissive
              }
              
              // DEBUG: Print material info
              std::cout << "DEBUG: Material [" << tinyobj_mat.name << "] Em: " << scene_mat.emission 
                        << " Type: " << (int)scene_mat.type << " (7=LIGHT)" << std::endl;

              int scene_mat_id = scene_desc.addMaterial(scene_mat);
              material_id_map[static_cast<int>(i)] = scene_mat_id;
          }
      }

      // 2. Convert shapes (meshes)
      // Calculate overall bounding box for camera positioning
      Vec3 overall_min_bounds(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
      Vec3 overall_max_bounds(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
      bool first_vertex = true;
      bool has_emissive_material = false;

      // Check for emissive materials first
      for (const auto& mat : scene_desc.materials) {
          if (mat.type == MaterialType::LIGHT) {
              has_emissive_material = true;
              break;
          }
      }

      for (const auto& shape_tinyobj : shapes) {
          int mat_id = material_id_map.count(shape_tinyobj.mesh.material_ids[0]) ? material_id_map[shape_tinyobj.mesh.material_ids[0]] : material_id_map[-1];

          for (size_t f = 0; f < shape_tinyobj.mesh.num_face_vertices.size(); ++f) {
              int fv = shape_tinyobj.mesh.num_face_vertices[f];
              if (fv != 3) continue; // Skip non-triangles

              GeometryDesc geom;
              geom.type = GeometryType::TRIANGLE;
              geom.material_id = mat_id;
              // Initialize bounds to inverted infinity for correct min/max
              geom.bounds_min = Vec3(1e30, 1e30, 1e30);
              geom.bounds_max = Vec3(-1e30, -1e30, -1e30);

              for (int i = 0; i < 3; ++i) {
                  tinyobj::index_t idx = shape_tinyobj.mesh.indices[f * 3 + i];
                  
                  Vec3 v_pos(
                      attrib.vertices[3 * idx.vertex_index + 0],
                      attrib.vertices[3 * idx.vertex_index + 1],
                      attrib.vertices[3 * idx.vertex_index + 2]
                  );

                  // Update overall bounding box
                  if (first_vertex) {
                      overall_min_bounds = v_pos;
                      overall_max_bounds = v_pos;
                      first_vertex = false;
                  } else {
                      overall_min_bounds.e[0] = std::min(overall_min_bounds.e[0], v_pos.e[0]);
                      overall_min_bounds.e[1] = std::min(overall_min_bounds.e[1], v_pos.e[1]);
                      overall_min_bounds.e[2] = std::min(overall_min_bounds.e[2], v_pos.e[2]);
                      
                      overall_max_bounds.e[0] = std::max(overall_max_bounds.e[0], v_pos.e[0]);
                      overall_max_bounds.e[1] = std::max(overall_max_bounds.e[1], v_pos.e[1]);
                      overall_max_bounds.e[2] = std::max(overall_max_bounds.e[2], v_pos.e[2]);
                  }

                  // Update triangle bounds
                  geom.bounds_min.e[0] = std::min(geom.bounds_min.e[0], v_pos.e[0]);
                  geom.bounds_min.e[1] = std::min(geom.bounds_min.e[1], v_pos.e[1]);
                  geom.bounds_min.e[2] = std::min(geom.bounds_min.e[2], v_pos.e[2]);
                  
                  geom.bounds_max.e[0] = std::max(geom.bounds_max.e[0], v_pos.e[0]);
                  geom.bounds_max.e[1] = std::max(geom.bounds_max.e[1], v_pos.e[1]);
                  geom.bounds_max.e[2] = std::max(geom.bounds_max.e[2], v_pos.e[2]);

                  if (i == 0) geom.data.triangle.v0 = v_pos;
                  else if (i == 1) geom.data.triangle.v1 = v_pos;
                  else if (i == 2) geom.data.triangle.v2 = v_pos;

                  if (idx.normal_index >= 0) {
                      geom.data.triangle.smooth_shadings = true;
                      Vec3 v_norm(
                          attrib.normals[3 * idx.normal_index + 0],
                          attrib.normals[3 * idx.normal_index + 1],
                          attrib.normals[3 * idx.normal_index + 2]
                      );
                      if (i == 0) geom.data.triangle.n0 = v_norm;
                      else if (i == 1) geom.data.triangle.n1 = v_norm;
                      else if (i == 2) geom.data.triangle.n2 = v_norm;
                  }
              }
              scene_desc.addGeometry(geom);
          }
      }
      
      // 3. Auto-camera setup
      if (first_vertex) { // No vertices loaded, use default camera
          scene_desc.camera_position = Vec3(0, 0, 3);
          scene_desc.camera_look_at = Vec3(0, 0, 0);
      } else {
          Vec3 center = (overall_min_bounds + overall_max_bounds) / 2.0;
          Vec3 extents = overall_max_bounds - overall_min_bounds;
          float radius = extents.length() / 2.0f;
          
          // Check if it looks like a Cornell Box (approx 2x2x2 centered at 0)
          // Cornell box usually -1 to 1 in dimensions
          bool is_cornell_box = (extents.x() > 1.8 && extents.x() < 2.2) &&
                                (extents.y() > 1.8 && extents.y() < 2.2) &&
                                (extents.z() > 1.8 && extents.z() < 2.2) &&
                                (std::abs(center.x()) < 0.2) &&
                                (std::abs(center.y()) < 0.2) &&
                                (std::abs(center.z()) < 0.2);

          if (is_cornell_box) {
              // Standard Cornell Box View
              scene_desc.camera_position = Vec3(0.0, 0.0, 3.8); // Adjusted for Z-forward/back
              scene_desc.camera_look_at = Vec3(0.0, 0.0, 0.0);
              scene_desc.camera_fov = 35.0f;
          } else {
              // Position camera to look at the center of the model, distance based on radius
              // Use an isometric-style view (offset in X, Y, Z) to show 3D structure
              scene_desc.camera_look_at = center;
              scene_desc.camera_position = center + Vec3(radius * 1.5f, radius * 1.0f, radius * 2.0f);
              scene_desc.camera_fov = 45.0f; // Default FOV for models
          }
      }

      // 4. Default light (if no lights in scene)
      if (!first_vertex && !has_emissive_material) {
          Vec3 center = (overall_min_bounds + overall_max_bounds) / 2.0;
          float size = (overall_max_bounds - overall_min_bounds).length();
          
          int light_mat = scene_desc.addMaterial(MaterialDesc::light(Vec3(15, 15, 15))); // Bright white light
          
          // Place light above and slightly in front
          Vec3 light_pos = center + Vec3(0, size * 1.5f, size * 0.5f);
          
          // Create a rectangle light
          scene_desc.addRectangle(light_pos, Vec3(size, 0, 0), Vec3(0, 0, size), light_mat);
      } else if (first_vertex) {
          // No vertices case
          int light_mat = scene_desc.addMaterial(MaterialDesc::light(Vec3(10, 10, 10)));
          scene_desc.addRectangle(Vec3(-1.0, 3.0, -2.0), Vec3(2.5, 0, 0), Vec3(0, 0, 1.5), light_mat);
      }

      scene_desc.use_bvh = true;
      scene_desc.buildBVH();
      
      return scene_desc;
   }

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
               cout << "Building BVH acceleration structure...\n";
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
      cout << "Creating default scene...\n";
      using namespace Scene;
      SceneDescription scene_desc;

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