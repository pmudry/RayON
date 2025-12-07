/**
 * @file yaml_scene_loader.cc
 * @brief Implementation of YAML scene loader
 */

#include "yaml_scene_loader.hpp"
#include "../../external/tiny_obj_loader.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

namespace Scene
{

// Utility functions
static string trimWhitespace(const string &str)
{
   size_t first = str.find_first_not_of(" \t\r\n");
   if (first == string::npos)
      return "";
   size_t last = str.find_last_not_of(" \t\r\n");
   return str.substr(first, last - first + 1);
}

static string removeQuotes(const string &str)
{
   string s = trimWhitespace(str);
   if (s.length() >= 2 && s[0] == '"' && s[s.length() - 1] == '"')
   {
      return s.substr(1, s.length() - 2);
   }
   return s;
}

static string removeComment(const string &line)
{
   size_t pos = line.find('#');
   if (pos != string::npos)
   {
      return line.substr(0, pos);
   }
   return line;
}

static Vec3 parseVec3Array(const string &str)
{
   string cleaned = trimWhitespace(str);
   if (cleaned.empty() || cleaned[0] != '[')
   {
      return Vec3(0, 0, 0);
   }

   // Remove brackets
   cleaned = cleaned.substr(1, cleaned.length() - 2);

   // Parse comma-separated values
   vector<double> values;
   stringstream ss(cleaned);
   string token;

   while (getline(ss, token, ','))
   {
      values.push_back(stod(trimWhitespace(token)));
   }

   if (values.size() >= 3)
   {
      return Vec3(values[0], values[1], values[2]);
   }
   return Vec3(0, 0, 0);
}

static MaterialType parseMaterialType(const string &type_str)
{
   if (type_str == "lambertian")
      return MaterialType::LAMBERTIAN;
   if (type_str == "metal")
      return MaterialType::METAL;
   if (type_str == "mirror")
      return MaterialType::MIRROR;
   if (type_str == "rough_mirror")
      return MaterialType::ROUGH_MIRROR;
   if (type_str == "glass")
      return MaterialType::GLASS;
   if (type_str == "dielectric")
      return MaterialType::DIELECTRIC;
   if (type_str == "light")
      return MaterialType::LIGHT;
   if (type_str == "constant")
      return MaterialType::CONSTANT;
   if (type_str == "show_normals")
      return MaterialType::SHOW_NORMALS;
   return MaterialType::LAMBERTIAN;
}

static ProceduralPattern parsePatternType(const string &type_str)
{
   if (type_str == "fibonacci_dots")
      return ProceduralPattern::FIBONACCI_DOTS;
   if (type_str == "checkerboard")
      return ProceduralPattern::CHECKERBOARD;
   if (type_str == "stripes")
      return ProceduralPattern::STRIPES;
   return ProceduralPattern::NONE;
}

static Vec3 rotatePoint(Vec3 p, Vec3 rot_deg)
{
   double rad_x = rot_deg.x() * M_PI / 180.0;
   double rad_y = rot_deg.y() * M_PI / 180.0;
   double rad_z = rot_deg.z() * M_PI / 180.0;

   // Rotate X
   double y1 = p.y() * cos(rad_x) - p.z() * sin(rad_x);
   double z1 = p.y() * sin(rad_x) + p.z() * cos(rad_x);
   p = Vec3(p.x(), y1, z1);

   // Rotate Y
   double x2 = p.x() * cos(rad_y) + p.z() * sin(rad_y);
   double z2 = -p.x() * sin(rad_y) + p.z() * cos(rad_y);
   p = Vec3(x2, p.y(), z2);

   // Rotate Z
   double x3 = p.x() * cos(rad_z) - p.y() * sin(rad_z);
   double y3 = p.x() * sin(rad_z) + p.y() * cos(rad_z);
   p = Vec3(x3, y3, p.z());

   return p;
}

static void loadObjGeometry(const string &filename, SceneDescription &scene, const Vec3 &pos, const Vec3 &rot,
                            const Vec3 &scale, int override_mat_id)
{
   tinyobj::attrib_t attrib;
   std::vector<tinyobj::shape_t> shapes;
   std::vector<tinyobj::material_t> materials;
   std::string warn, err;

   // Handle potential relative paths
   string full_path = filename; 

   // Extract directory for MTL search
   std::string mtl_dir;
   size_t last_slash = filename.find_last_of("/\\");
   if (last_slash != std::string::npos)
   {
      mtl_dir = filename.substr(0, last_slash) + "/";
   }

   bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, full_path.c_str(), mtl_dir.c_str());

   if (!warn.empty())
      std::cout << "OBJ Warning: " << warn << std::endl;
   if (!err.empty())
      std::cerr << "OBJ Error: " << err << std::endl;
   if (!ret)
      return;

   // Material mapping (only if not overridden)
   std::map<int, int> material_map;
   if (override_mat_id == -1)
   {
      // Load materials from OBJ to Scene
      for (size_t i = 0; i < materials.size(); ++i)
      {
         const auto &tmat = materials[i];
         MaterialDesc mat;
         mat.type = MaterialType::LAMBERTIAN;
         mat.albedo = Vec3(tmat.diffuse[0], tmat.diffuse[1], tmat.diffuse[2]);

         if (tmat.specular[0] > 0 || tmat.shininess > 0)
         {
            mat.type = MaterialType::METAL;
            mat.metallic = 1.0;
            mat.roughness = 1.0f - std::min(1.0f, tmat.shininess / 1000.0f);
         }
         if (tmat.ior > 1.0)
         {
            mat.type = MaterialType::GLASS;
            mat.refractive_index = tmat.ior;
         }
         int id = scene.addMaterial(mat);
         material_map[(int)i] = id;
      }
   }

   // Process shapes
   for (const auto &shape : shapes)
   {
      for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
      {
         if (shape.mesh.num_face_vertices[f] != 3)
            continue; // Triangles only

         GeometryDesc geom;
         geom.type = GeometryType::TRIANGLE;

         // Material
         if (override_mat_id != -1)
         {
            geom.material_id = override_mat_id;
         }
         else
         {
            int obj_mat_id = shape.mesh.material_ids[f];
            if (material_map.count(obj_mat_id))
            {
               geom.material_id = material_map[obj_mat_id];
            }
            else
            {
               geom.material_id = 0; // Fallback
            }
         }

         // Vertices
         Vec3 vertices[3];
         Vec3 normals[3];
         bool has_normals = false;

         for (int v = 0; v < 3; v++)
         {
            tinyobj::index_t idx = shape.mesh.indices[3 * f + v];

            // Position
            Vec3 p(attrib.vertices[3 * idx.vertex_index + 0], attrib.vertices[3 * idx.vertex_index + 1],
                   attrib.vertices[3 * idx.vertex_index + 2]);

            // Apply Transform
            // Scale
            p = Vec3(p.x() * scale.x(), p.y() * scale.y(), p.z() * scale.z());
            // Rotate
            p = rotatePoint(p, rot);
            // Translate
            p = p + pos;

            vertices[v] = p;

            // Normal
            if (idx.normal_index >= 0)
            {
               has_normals = true;
               Vec3 n(attrib.normals[3 * idx.normal_index + 0], attrib.normals[3 * idx.normal_index + 1],
                      attrib.normals[3 * idx.normal_index + 2]);
               // Rotate normal (no scale/translate)
               n = rotatePoint(n, rot);
               normals[v] = unit_vector(n);
            }
         }

         geom.data.triangle.v0 = vertices[0];
         geom.data.triangle.v1 = vertices[1];
         geom.data.triangle.v2 = vertices[2];
         geom.data.triangle.has_normals = has_normals;
         if (has_normals)
         {
            geom.data.triangle.n0 = normals[0];
            geom.data.triangle.n1 = normals[1];
            geom.data.triangle.n2 = normals[2];
         }

         // Bounds
         geom.bounds_min = Vec3(std::min({vertices[0].x(), vertices[1].x(), vertices[2].x()}),
                                std::min({vertices[0].y(), vertices[1].y(), vertices[2].y()}),
                                std::min({vertices[0].z(), vertices[1].z(), vertices[2].z()}));
         geom.bounds_max = Vec3(std::max({vertices[0].x(), vertices[1].x(), vertices[2].x()}),
                                std::max({vertices[0].y(), vertices[1].y(), vertices[2].y()}),
                                std::max({vertices[0].z(), vertices[1].z(), vertices[2].z()}));

         scene.addGeometry(geom);
      }
   }
}

/**
 * @brief Simple YAML parser for scene files
 */
class SimpleYAMLParser
{
 private:
   map<string, string> values_;

 public:
   bool parseFile(const string &filename)
   {
      ifstream file(filename);
      if (!file.is_open())
      {
         cerr << "ERROR: Cannot open scene file: " << filename << "\n";
         return false;
      }

      string line;
      vector<string> section_stack;
      int material_index = 0;
      int geometry_index = 0;
      bool in_materials = false;
      bool in_geometry = false;

      while (getline(file, line))
      {
         // Remove comments and trim
         line = removeComment(line);
         string trimmed = trimWhitespace(line);
         if (trimmed.empty())
            continue;

         // Calculate indentation
         int indent = 0;
         while (indent < (int)line.length() && (line[indent] == ' ' || line[indent] == '\t'))
         {
            indent++;
         }

         // Check for section headers
         if (trimmed == "materials:")
         {
            in_materials = true;
            in_geometry = false;
            material_index = 0;
            continue;
         }
         else if (trimmed == "geometry:")
         {
            in_materials = false;
            in_geometry = true;
            geometry_index = 0;
            continue;
         }
         else if (trimmed == "scene:")
         {
             in_materials = false;
             in_geometry = false;
             section_stack.clear();
             section_stack.push_back("scene");
             continue;
         }

         // Handle list items
         if (trimmed.length() >= 2 && trimmed[0] == '-' && trimmed[1] == ' ')
         {
            trimmed = trimWhitespace(trimmed.substr(2));
            if (in_materials)
            {
               section_stack.clear();
               section_stack.push_back("material" + to_string(material_index));
               material_index++;
            }
            else if (in_geometry)
            {
               section_stack.clear();
               section_stack.push_back("geom" + to_string(geometry_index));
               geometry_index++;
            }
         }

         // Parse key-value pairs
         size_t colon_pos = trimmed.find(':');
         if (colon_pos != string::npos)
         {
            string key = trimWhitespace(trimmed.substr(0, colon_pos));
            string value = trimWhitespace(trimmed.substr(colon_pos + 1));

            // Handle nested properties inside 'scene' block (e.g., scene.camera.position)
            if (!section_stack.empty() && section_stack[0] == "scene") {
                // Basic indentation-based hierarchy handling for 'scene' block
                // If we are in 'scene', check if the key is a new sub-section like "camera"
                if (value.empty()) {
                    // This is a subsection header (e.g., "camera:")
                    // For simplicity in this parser, we'll just append it to the stack or handle it via dot notation in keys
                    // A robust parser would track indentation levels.
                    // Here we assume keys are like "camera" or properties under it.
                    // Given the simple structure, we can cheat a bit:
                    // If indentation increases, it's a nested property.
                    // But we already trimmed 'line'.
                    // Let's stick to flat keys for simplicity or rely on the current simple logic.
                    // The current logic builds keys like "scene.camera" but doesn't handle deeper nesting well without indentation tracking.
                    
                    // IMPROVED LOGIC:
                    // If we are in the 'scene' block, we'll just manually construct keys like "scene.camera.position"
                    // based on the key name if it's unique, or we need a better parser.
                    // However, the parser above simply concatenates `section_stack`.
                    
                    // Hack for this specific format:
                    // If key is "camera", push to stack. If indentation decreases, pop.
                    // BUT, indentation tracking is tricky here.
                    // Let's assume the parser logic "Build full key path" below needs help.
                    
                    // Let's try to handle "camera:" line specifically if value is empty
                    if (key == "camera") {
                        if(section_stack.back() != "scene.camera") // Avoid duplicate push
                             section_stack.push_back("camera"); 
                        continue;
                    }
                }
            }

            // Build full key path
            string full_key = "";
            for (const auto &section : section_stack)
            {
               full_key += section + ".";
            }
            full_key += key;

            if (!value.empty())
            {
               values_[full_key] = value;
            }
         }
      }

      return true;
   }

   string getString(const string &key, const string &default_val = "") const
   {
      auto it = values_.find(key);
      return (it != values_.end()) ? it->second : default_val;
   }

   float getFloat(const string &key, float default_val = 0.0f) const
   {
      auto it = values_.find(key);
      if (it != values_.end())
      {
         return stof(it->second);
      }
      return default_val;
   }

   double getDouble(const string &key, double default_val = 0.0) const
   {
      auto it = values_.find(key);
      if (it != values_.end())
      {
         return stod(it->second);
      }
      return default_val;
   }

   int getInt(const string &key, int default_val = 0) const
   {
      auto it = values_.find(key);
      if (it != values_.end())
      {
         return stoi(it->second);
      }
      return default_val;
   }

   Vec3 getVec3(const string &key, const Vec3 &default_val = Vec3(0, 0, 0)) const
   {
      auto it = values_.find(key);
      if (it != values_.end())
      {
         return parseVec3Array(it->second);
      }
      return default_val;
   }

   bool getBool(const string &key, bool default_val = false) const
   {
      auto it = values_.find(key);
      if (it != values_.end())
      {
         string val = it->second;
         transform(val.begin(), val.end(), val.begin(), ::tolower);
         return (val == "true" || val == "yes" || val == "1");
      }
      return default_val;
   }

   bool hasKey(const string &key) const { return values_.find(key) != values_.end(); }
};

static bool loadMaterials(const SimpleYAMLParser &parser, SceneDescription &scene,
                          map<string, int> &material_name_to_id)
{
   // Try to load materials by index
   for (int i = 0; i < 100; ++i)
   { // Reasonable limit
      string prefix = "material" + to_string(i);

      if (!parser.hasKey(prefix + ".name"))
      {
         break; // No more materials
      }

      string mat_name = removeQuotes(parser.getString(prefix + ".name"));
      string mat_type = removeQuotes(parser.getString(prefix + ".type", "lambertian"));

      MaterialDesc mat;
      mat.type = parseMaterialType(mat_type);
      mat.albedo = parser.getVec3(prefix + ".albedo", Vec3(0.7, 0.7, 0.7));
      mat.emission = parser.getVec3(prefix + ".emission", Vec3(0, 0, 0));
      mat.roughness = parser.getFloat(prefix + ".roughness", 0.0f);
      mat.metallic = parser.getFloat(prefix + ".metallic", 0.0f);
      mat.refractive_index = parser.getFloat(prefix + ".refractive_index", 1.0f);
      mat.transmission = parser.getFloat(prefix + ".transmission", 0.0f);

      // Check for procedural pattern
      if (parser.hasKey(prefix + ".pattern.type"))
      {
         mat.pattern = parsePatternType(removeQuotes(parser.getString(prefix + ".pattern.type")));
         mat.pattern_color = parser.getVec3(prefix + ".pattern.color", Vec3(0, 0, 0));
         mat.pattern_param1 = parser.getFloat(prefix + ".pattern.dot_count", 0.0f);
         mat.pattern_param2 = parser.getFloat(prefix + ".pattern.dot_radius", 0.0f);
      }

      int mat_id = scene.addMaterial(mat);
      material_name_to_id[mat_name] = mat_id;

      cout << "  Loaded material: " << mat_name << " (ID: " << mat_id
           << ")"
              "\n";
   }

   return !scene.materials.empty();
}

static bool loadGeometry(const SimpleYAMLParser &parser, SceneDescription &scene,
                         const map<string, int> &material_name_to_id, const string& scene_filepath)
{
   // Try to load geometry by index
   for (int i = 0; i < 1000; ++i)
   { // Reasonable limit
      string prefix = "geom" + to_string(i);

      if (!parser.hasKey(prefix + ".type"))
      {
         break; // No more geometry
      }

      string geom_type = removeQuotes(parser.getString(prefix + ".type"));

      // Special case for OBJ: material is optional
      if (geom_type == "obj")
      {
         string filename = removeQuotes(parser.getString(prefix + ".filename"));
         
         // Resolve path relative to scene file if not found
         ifstream check_file(filename);
         if (!check_file.good()) {
             // File not found at raw path, try relative to scene file
             size_t last_slash = scene_filepath.find_last_of("/\\");
             if (last_slash != string::npos) {
                 string dir = scene_filepath.substr(0, last_slash + 1);
                 string relative_path = dir + filename;
                 ifstream check_relative(relative_path);
                 if (check_relative.good()) {
                     filename = relative_path;
                 }
             }
         }
         check_file.close();

         Vec3 pos = parser.getVec3(prefix + ".position", Vec3(0, 0, 0));
         Vec3 rot = parser.getVec3(prefix + ".rotation", Vec3(0, 0, 0));

         Vec3 scale(1, 1, 1);
         string scale_val = parser.getString(prefix + ".scale");
         if (!scale_val.empty())
         {
            if (scale_val.find('[') != string::npos)
            {
               scale = parseVec3Array(scale_val);
            }
            else
            {
               double s = stod(scale_val);
               scale = Vec3(s, s, s);
            }
         }

         int mat_id = -1;
         string mat_name = removeQuotes(parser.getString(prefix + ".material"));
         if (!mat_name.empty())
         {
            auto it = material_name_to_id.find(mat_name);
            if (it != material_name_to_id.end())
            {
               mat_id = it->second;
            }
            else
            {
               cerr << "WARNING: Unknown material for OBJ: " << mat_name << ". Using OBJ materials.\n";
            }
         }

         loadObjGeometry(filename, scene, pos, rot, scale, mat_id);
         cout << "  Loaded OBJ: " << filename << "\n";
         continue;
      }

      string mat_name = removeQuotes(parser.getString(prefix + ".material"));

      // Look up material ID
      auto it = material_name_to_id.find(mat_name);
      if (it == material_name_to_id.end())
      {
         cerr << "ERROR: Unknown material: " << mat_name << "\n";
         continue;
      }
      int mat_id = it->second;

      if (geom_type == "sphere")
      {
         Vec3 center = parser.getVec3(prefix + ".center");
         double radius = parser.getDouble(prefix + ".radius", 1.0);
         scene.addSphere(center, radius, mat_id);
      }
      else if (geom_type == "displaced_sphere")
      {
         Vec3 center = parser.getVec3(prefix + ".center");
         double radius = parser.getDouble(prefix + ".radius", 1.0);
         float disp_scale = parser.getFloat(prefix + ".displacement_scale", 0.2f);
         int pattern = parser.getInt(prefix + ".pattern_type", 0);
         scene.addDisplacedSphere(center, radius, mat_id, disp_scale, pattern);
      }
      else if (geom_type == "rectangle")
      {
         Vec3 corner = parser.getVec3(prefix + ".corner");
         Vec3 u = parser.getVec3(prefix + ".u");
         Vec3 v = parser.getVec3(prefix + ".v");
         scene.addRectangle(corner, u, v, mat_id);
      }

      cout << "  Loaded " << geom_type << " with material " << mat_name << "\n";
   }

   return !scene.geometries.empty();
}

bool loadSceneFromYAML(const char *filename, SceneDescription &scene)
{
   SimpleYAMLParser parser;

   cout << "Loading scene from: " << filename << "\n";

   if (!parser.parseFile(filename))
   {
      cerr << "ERROR: Failed to parse YAML file"
              "\n";
      return false;
   }

   // Load Camera Settings
   if (parser.hasKey("scene.camera.position"))
   {
       scene.camera_position = parser.getVec3("scene.camera.position");
       cout << "  Loaded camera position: " << scene.camera_position << endl;
   }
   if (parser.hasKey("scene.camera.look_at"))
   {
       scene.camera_look_at = parser.getVec3("scene.camera.look_at");
       cout << "  Loaded camera look_at: " << scene.camera_look_at << endl;
   }
   if (parser.hasKey("scene.camera.up"))
   {
       scene.camera_up = parser.getVec3("scene.camera.up");
   }
   if (parser.hasKey("scene.camera.fov"))
   {
       scene.camera_fov = parser.getFloat("scene.camera.fov");
       cout << "  Loaded camera fov: " << scene.camera_fov << endl;
   }

   // Clear existing scene
   scene.materials.clear();
   scene.geometries.clear();
   scene.meshes.clear();

   // Load materials first
   map<string, int> material_name_to_id;
   if (!loadMaterials(parser, scene, material_name_to_id))
   {
      cerr << "ERROR: Failed to load materials"
              "\n";
      return false;
   }

   // Then load geometry (which references materials)
   // Pass filename to resolve relative paths
   if (!loadGeometry(parser, scene, material_name_to_id, string(filename)))
   {
      cerr << "ERROR: Failed to load geometry"
              "\n";
      return false;
   }

   cout << "Scene loaded successfully:" << "\n";
   cout << "  - " << scene.materials.size() << " materials" << "\n";
   cout << "  - " << scene.geometries.size() << " objects" << "\n";

   return scene.validate();
}

bool saveSceneToYAML(const char *filename, const SceneDescription &scene)
{
   ofstream file(filename);
   if (!file.is_open())
   {
      cerr << "ERROR: Cannot open file for writing: " << filename << "\n";
      return false;
   }

   file << "# Scene exported from raytracer\n\n";
   file << "scene:\n";
   file << "  name: \"Exported Scene\"\n\n";

   // Write materials
   file << "materials:\n";
   for (size_t i = 0; i < scene.materials.size(); ++i)
   {
      const auto &mat = scene.materials[i];
      file << "  - name: \"material_" << i << "\"\n";

      // Material type
      file << "    type: ";
      switch (mat.type)
      {
      case MaterialType::LAMBERTIAN:
         file << "\"lambertian\"";
         break;
      case MaterialType::METAL:
         file << "\"metal\"";
         break;
      case MaterialType::MIRROR:
         file << "\"mirror\"";
         break;
      case MaterialType::ROUGH_MIRROR:
         file << "\"rough_mirror\"";
         break;
      case MaterialType::GLASS:
         file << "\"glass\"";
         break;
      case MaterialType::LIGHT:
         file << "\"light\"";
         break;
      default:
         file << "\"lambertian\"";
         break;
      }
      file << "\n";

      // Material properties
      file << "    albedo: [" << mat.albedo.x() << ", " << mat.albedo.y() << ", " << mat.albedo.z() << "]\n";
      if (mat.emission.x() > 0 || mat.emission.y() > 0 || mat.emission.z() > 0)
      {
         file << "    emission: [" << mat.emission.x() << ", " << mat.emission.y() << ", " << mat.emission.z() << "]\n";
      }
      if (mat.roughness > 0)
      {
         file << "    roughness: " << mat.roughness << "\n";
      }
      if (mat.metallic > 0)
      {
         file << "    metallic: " << mat.metallic << "\n";
      }
      if (mat.refractive_index != 1.0f)
      {
         file << "    refractive_index: " << mat.refractive_index << "\n";
      }
      if (mat.transmission > 0)
      {
         file << "    transmission: " << mat.transmission << "\n";
      }

      // Pattern
      if (mat.pattern != ProceduralPattern::NONE)
      {
         file << "    pattern:\n";
         file << "      type: ";
         switch (mat.pattern)
         {
         case ProceduralPattern::FIBONACCI_DOTS:
            file << "\"fibonacci_dots\"";
            break;
         case ProceduralPattern::CHECKERBOARD:
            file << "\"checkerboard\"";
            break;
         case ProceduralPattern::STRIPES:
            file << "\"stripes\"";
            break;
         default:
            break;
         }
         file << "\n";
         file << "      color: [" << mat.pattern_color.x() << ", " << mat.pattern_color.y() << ", "
              << mat.pattern_color.z() << "]\n";
         if (mat.pattern_param1 > 0)
            file << "      dot_count: " << (int)mat.pattern_param1 << "\n";
         if (mat.pattern_param2 > 0)
            file << "      dot_radius: " << mat.pattern_param2 << "\n";
      }
   }

   file << "\n";

   // Write geometry
   file << "geometry:\n";
   for (const auto &geom : scene.geometries)
   {
      file << "  - type: ";

      switch (geom.type)
      {
      case GeometryType::SPHERE:
         file << "\"sphere\"\n";
         file << "    material: \"material_" << geom.material_id << "\"\n";
         file << "    center: [" << geom.data.sphere.center.x() << ", " << geom.data.sphere.center.y() << ", "
              << geom.data.sphere.center.z() << "]\n";
         file << "    radius: " << geom.data.sphere.radius << "\n";
         break;

      case GeometryType::DISPLACED_SPHERE:
         file << "\"displaced_sphere\"\n";
         file << "    material: \"material_" << geom.material_id << "\"\n";
         file << "    center: [" << geom.data.displaced_sphere.center.x() << ", "
              << geom.data.displaced_sphere.center.y() << ", " << geom.data.displaced_sphere.center.z() << "]\n";
         file << "    radius: " << geom.data.displaced_sphere.radius << "\n";
         file << "    displacement_scale: " << geom.data.displaced_sphere.displacement_scale << "\n";
         file << "    pattern_type: " << geom.data.displaced_sphere.pattern_type << "\n";
         break;

      case GeometryType::RECTANGLE:
         file << "\"rectangle\"\n";
         file << "    material: \"material_" << geom.material_id << "\"\n";
         file << "    corner: [" << geom.data.rectangle.corner.x() << ", " << geom.data.rectangle.corner.y() << ", "
              << geom.data.rectangle.corner.z() << "]\n";
         file << "    u: [" << geom.data.rectangle.u.x() << ", " << geom.data.rectangle.u.y() << ", "
              << geom.data.rectangle.u.z() << "]\n";
         file << "    v: [" << geom.data.rectangle.v.x() << ", " << geom.data.rectangle.v.y() << ", "
              << geom.data.rectangle.v.z() << "]\n";
         break;

      default:
         file << "\"unknown\"\n";
         break;
      }
   }

   file.close();
   cout << "Scene saved to: " << filename << "\n";
   return true;
}

} // namespace Scene
