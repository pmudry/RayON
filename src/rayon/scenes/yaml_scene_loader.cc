/**
 * @file yaml_scene_loader.cc
 * @brief Implementation of YAML scene loader
 */

#include "yaml_scene_loader.hpp"
#include "obj_loader.hpp"
#include "volume_grid.hpp"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

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
   if (type_str == "anisotropic_metal")
      return MaterialType::ANISOTROPIC_METAL;
   if (type_str == "thin_film")
      return MaterialType::THIN_FILM;
   if (type_str == "clear_coat")
      return MaterialType::CLEAR_COAT;
   if (type_str == "volume" || type_str == "smoke" || type_str == "cloud")
      return MaterialType::VOLUME;
   return MaterialType::LAMBERTIAN;
}

static ProceduralPattern parsePatternType(const string &type_str)
{
   if (type_str == "fibonacci_dots")
      return ProceduralPattern::FIBONACCI_DOTS;
   if (type_str == "stripes")
      return ProceduralPattern::STRIPES;
   return ProceduralPattern::NONE;
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
            section_stack.clear();
            material_index = 0;
            continue;
         }
         else if (trimmed == "geometry:" || trimmed == "geometries:")
         {
            in_materials = false;
            in_geometry = true;
            section_stack.clear();
            geometry_index = 0;
            continue;
         }
         else if (trimmed == "camera:" || trimmed == "settings:")
         {
            in_materials = false;
            in_geometry = false;
            section_stack.clear();
            section_stack.push_back(trimmed.substr(0, trimmed.size() - 1)); // "camera" or "settings"
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

            // Handle flow-style inline maps: { key: val, key: val, ... }
            if (!trimmed.empty() && trimmed.front() == '{' && trimmed.back() == '}')
            {
               string inner = trimmed.substr(1, trimmed.size() - 2);
               // Split by comma, but respect brackets for arrays like [1,2,3]
               vector<string> pairs;
               int bracket_depth = 0;
               string current;
               for (char ch : inner)
               {
                  if (ch == '[')
                     bracket_depth++;
                  else if (ch == ']')
                     bracket_depth--;
                  if (ch == ',' && bracket_depth == 0)
                  {
                     pairs.push_back(current);
                     current.clear();
                  }
                  else
                  {
                     current += ch;
                  }
               }
               if (!current.empty())
                  pairs.push_back(current);

               for (const auto &pair : pairs)
               {
                  size_t cp = pair.find(':');
                  if (cp != string::npos)
                  {
                     string k = trimWhitespace(pair.substr(0, cp));
                     string v = trimWhitespace(pair.substr(cp + 1));
                     string fk = k;
                     for (const auto &s : section_stack)
                        fk = s + "." + fk;
                     if (!v.empty())
                        values_[fk] = v;
                  }
               }
               continue;
            }
         }

         // Parse key-value pairs
         size_t colon_pos = trimmed.find(':');
         if (colon_pos != string::npos)
         {
            string key = trimWhitespace(trimmed.substr(0, colon_pos));
            string value = trimWhitespace(trimmed.substr(colon_pos + 1));

            // Build full key path
            string full_key = key;
            for (const auto &section : section_stack)
            {
               full_key = section + "." + full_key;
            }

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
                          map<string, int> &material_name_to_id, const string &scene_dir)
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
      if (parser.hasKey(prefix + ".color") && parser.hasKey(prefix + ".emission_intensity"))
      {
         Vec3 color = parser.getVec3(prefix + ".color", Vec3(1, 1, 1));
         float intensity = parser.getFloat(prefix + ".emission_intensity", 1.0f);
         mat.emission = color * intensity;
      }
      else
      {
         mat.emission = parser.getVec3(prefix + ".emission", Vec3(0, 0, 0));
      }
      mat.roughness = parser.getFloat(prefix + ".roughness", 0.0f);
      mat.metallic = parser.getFloat(prefix + ".metallic", 0.0f);
      mat.refractive_index = parser.getFloat(prefix + ".refractive_index", 1.0f);
      mat.transmission = parser.getFloat(prefix + ".transmission", 0.0f);
      mat.anisotropy = parser.getFloat(prefix + ".anisotropy", 0.0f);
      mat.eta = parser.getVec3(prefix + ".eta", Vec3(0, 0, 0));
      mat.k = parser.getVec3(prefix + ".k", Vec3(0, 0, 0));
      mat.film_thickness = parser.getFloat(prefix + ".film_thickness", 400.0f);
      mat.film_ior = parser.getFloat(prefix + ".film_ior", 1.33f);
      // Participating medium: `g` is the usual name for the phase asymmetry, `anisotropy` also works
      mat.density = parser.getFloat(prefix + ".density", 2.0f);
      mat.noise_scale = parser.getFloat(prefix + ".noise_scale", 2.0f);
      mat.noise_strength = parser.getFloat(prefix + ".noise_strength", 0.3f);
      mat.edge = parser.getFloat(prefix + ".edge", 0.1f);
      mat.lumps = parser.getFloat(prefix + ".lumps", 0.3f);
      mat.billow = parser.getFloat(prefix + ".billow", 0.7f);
      if (parser.hasKey(prefix + ".g"))
         mat.anisotropy = parser.getFloat(prefix + ".g", 0.0f);
      // Density grid from data (tools/vdb_to_volume.py): replaces the mesh shape and the noise.
      // The loader adds the grid's box as the medium boundary once the geometry is read.
      if (mat.type == MaterialType::VOLUME && parser.hasKey(prefix + ".grid"))
      {
         string grid_path = removeQuotes(parser.getString(prefix + ".grid"));
         if (!grid_path.empty() && !scene_dir.empty() && !std::filesystem::path(grid_path).is_absolute())
            grid_path = scene_dir + "/" + grid_path;
         VolumeGridHeader header;
         if (!readVolumeGridHeader(grid_path, header))
         {
            cerr << "ERROR: cannot read the volume grid '" << grid_path << "' of material '" << mat_name
                 << "' (expected a .vol file written by tools/vdb_to_volume.py)\n";
            return false;
         }
         mat.grid_file     = grid_path;
         mat.grid_scale    = parser.getFloat(prefix + ".grid_scale", 1.0f);
         mat.grid_position = parser.getVec3(prefix + ".grid_position", Vec3(0, 0, 0));
         const double s = mat.grid_scale;
         mat.grid_min = Vec3(mat.grid_position.x() + header.bmin[0] * s, mat.grid_position.y() + header.bmin[1] * s,
                             mat.grid_position.z() + header.bmin[2] * s);
         mat.grid_max = Vec3(mat.grid_position.x() + header.bmax[0] * s, mat.grid_position.y() + header.bmax[1] * s,
                             mat.grid_position.z() + header.bmax[2] * s);
         // The grid is normalised to [0, 1] and holds all the detail: no interior noise by default
         // (the majorant is then exact) and no distance grid of the boundary
         mat.noise_strength = parser.getFloat(prefix + ".noise_strength", 0.0f);
         mat.edge           = 0.0f;
         cout << "Volume grid: " << grid_path << " (" << header.nx << "x" << header.ny << "x" << header.nz
              << " cells), box [" << mat.grid_min << "] - [" << mat.grid_max << "]\n";
      }

      // Support named metal presets for anisotropic_metal
      if (mat.type == MaterialType::ANISOTROPIC_METAL && parser.hasKey(prefix + ".preset"))
      {
         string preset = removeQuotes(parser.getString(prefix + ".preset"));
         if (preset == "gold") {
            mat.eta = Vec3(0.18, 0.42, 1.37); mat.k = Vec3(3.42, 2.35, 1.77);
         } else if (preset == "silver") {
            mat.eta = Vec3(0.05, 0.06, 0.05); mat.k = Vec3(4.18, 3.35, 2.58);
         } else if (preset == "copper") {
            mat.eta = Vec3(0.27, 0.68, 1.22); mat.k = Vec3(3.60, 2.63, 2.29);
         } else if (preset == "aluminum") {
            mat.eta = Vec3(1.35, 0.97, 0.53); mat.k = Vec3(7.47, 6.40, 5.28);
         }
      }

      // Check for procedural pattern
      if (parser.hasKey(prefix + ".pattern.type"))
      {
         mat.pattern = parsePatternType(removeQuotes(parser.getString(prefix + ".pattern.type")));
         mat.pattern_color = parser.getVec3(prefix + ".pattern.color", Vec3(0, 0, 0));
         mat.pattern_param1 = parser.getFloat(prefix + ".pattern.dot_count", 0.0f);
         mat.pattern_param2 = parser.getFloat(prefix + ".pattern.dot_radius", 0.0f);
      }

      // Check for diffuse texture
      if (parser.hasKey(prefix + ".texture"))
      {
         string tex_path = removeQuotes(parser.getString(prefix + ".texture"));
         if (!tex_path.empty())
         {
            // Resolve relative paths against the YAML file's directory
            if (!tex_path.empty() && !scene_dir.empty() &&
                !std::filesystem::path(tex_path).is_absolute())
               tex_path = scene_dir + "/" + tex_path;
            int tex_id = scene.addTexture(tex_path);
            mat.texture_id = tex_id;
         }
      }

      int mat_id = scene.addMaterial(mat);
      material_name_to_id[mat_name] = mat_id;
   }

   return !scene.materials.empty();
}

// Boundary of a volume read from a density grid: the grid's world box as 12 outward-wound
// triangles carrying the material (the GPU back ends trace it like any volume mesh)
static void addVolumeGridBox(SceneDescription &scene, int mat_id)
{
   const Vec3 lo = scene.materials[mat_id].grid_min, hi = scene.materials[mat_id].grid_max;
   const Vec3 c[8] = {Vec3(lo.x(), lo.y(), lo.z()), Vec3(hi.x(), lo.y(), lo.z()), Vec3(hi.x(), hi.y(), lo.z()),
                      Vec3(lo.x(), hi.y(), lo.z()), Vec3(lo.x(), lo.y(), hi.z()), Vec3(hi.x(), lo.y(), hi.z()),
                      Vec3(hi.x(), hi.y(), hi.z()), Vec3(lo.x(), hi.y(), hi.z())};
   // Faces as quads (a, b, c, d) wound counter-clockwise seen from outside
   const int faces[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {3, 7, 6, 2}, {0, 4, 7, 3}, {1, 2, 6, 5}};
   for (const auto &f : faces)
   {
      scene.addTriangle(c[f[0]], c[f[1]], c[f[2]], mat_id);
      scene.addTriangle(c[f[0]], c[f[2]], c[f[3]], mat_id);
   }
}

static bool loadGeometry(const SimpleYAMLParser &parser, SceneDescription &scene,
                         const map<string, int> &material_name_to_id,
                         const string &scene_dir)
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
      string mat_name = removeQuotes(parser.getString(prefix + ".material"));

      // Look up material ID — optional for OBJ entries (MTL file provides materials)
      int mat_id = -1;
      if (!mat_name.empty())
      {
         auto it = material_name_to_id.find(mat_name);
         if (it != material_name_to_id.end())
         {
            mat_id = it->second;
         }
         else
         {
            cerr << "ERROR: Unknown material: " << mat_name << "\n";
            if (geom_type != "obj")
               continue;
            // OBJ: fallback to -1 (MTL-only mode)
         }
      }
      else if (geom_type != "obj")
      {
         cerr << "ERROR: Missing 'material' key for " << geom_type << " geometry[" << i << "]\n";
         continue;
      }

      // Check visibility flag (default: true)
      string vis_str = removeQuotes(parser.getString(prefix + ".visible", "true"));
      bool visible = (vis_str != "false" && vis_str != "0");

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
      else if (geom_type == "sdf")
      {
         // Ray-marched signed distance primitive (CUDA backend only). The shape picks the
         // meaning of the three parameters; named keys fill them, `params` sets them raw.
         static const struct
         {
            const char *name;
            SDFType     type;
            const char *keys[3];
         } shapes[] = {
            {"sphere", SDFType::SPHERE, {"radius", nullptr, nullptr}},
            {"box", SDFType::BOX, {nullptr, nullptr, nullptr}}, // half_extents vec3
            {"torus", SDFType::TORUS, {"major_radius", "minor_radius", nullptr}},
            {"capsule", SDFType::CAPSULE, {"radius", "height", nullptr}},
            {"cylinder", SDFType::CYLINDER, {"radius", "height", nullptr}},
            {"mandelbulb", SDFType::MANDELBULB, {"power", "iterations", nullptr}},
            {"death_star", SDFType::DEATH_STAR, {"radius", "cutout_radius", "cutout_distance"}},
            {"cut_hollow_sphere", SDFType::CUT_HOLLOW_SPHERE, {"radius", "cut_height", "thickness"}},
            {"octahedron", SDFType::OCTAHEDRON, {"size", nullptr, nullptr}},
            {"pyramid", SDFType::PYRAMID, {"height", nullptr, nullptr}},
         };
         const string shape = removeQuotes(parser.getString(prefix + ".shape"));
         const auto  *def   = (const decltype(&shapes[0]))nullptr;
         for (const auto &s : shapes)
            if (shape == s.name)
               def = &s;
         if (!def)
         {
            cerr << "ERROR: Unknown sdf shape '" << shape << "' for geometry[" << i << "]\n";
            continue;
         }
         Vec3 center = parser.getVec3(prefix + ".center");
         Vec3 params = parser.getVec3(prefix + ".params");
         if (def->type == SDFType::BOX && parser.hasKey(prefix + ".half_extents"))
            params = parser.getVec3(prefix + ".half_extents");
         double p[3] = {params.x(), params.y(), params.z()};
         for (int k = 0; k < 3; ++k)
            if (def->keys[k] && parser.hasKey(prefix + "." + def->keys[k]))
               p[k] = parser.getDouble(prefix + "." + def->keys[k], p[k]);
         // Rotation given in degrees (X, Y, Z Euler order), stored in radians
         Vec3 rot_deg = parser.getVec3(prefix + ".rotation");
         const double d2r = M_PI / 180.0;
         scene.addSDFPrimitive(def->type, center, Vec3(p[0], p[1], p[2]), mat_id,
                               Vec3(rot_deg.x() * d2r, rot_deg.y() * d2r, rot_deg.z() * d2r));
      }
      else if (geom_type == "rectangle")
      {
         Vec3 corner = parser.getVec3(prefix + ".corner");
         Vec3 u = parser.getVec3(prefix + ".u");
         Vec3 v = parser.getVec3(prefix + ".v");
         scene.addRectangle(corner, u, v, mat_id);
      }
      else if (geom_type == "triangle")
      {
         Vec3 v0 = parser.getVec3(prefix + ".v0");
         Vec3 v1 = parser.getVec3(prefix + ".v1");
         Vec3 v2 = parser.getVec3(prefix + ".v2");

         if (parser.hasKey(prefix + ".n0"))
         {
            Vec3 n0 = parser.getVec3(prefix + ".n0");
            Vec3 n1 = parser.getVec3(prefix + ".n1");
            Vec3 n2 = parser.getVec3(prefix + ".n2");
            scene.addTriangleWithNormals(v0, v1, v2, n0, n1, n2, mat_id);
         }
         else
         {
            scene.addTriangle(v0, v1, v2, mat_id);
         }
      }
      else if (geom_type == "obj")
      {
         string obj_file = removeQuotes(parser.getString(prefix + ".file"));
         Vec3 obj_position = parser.getVec3(prefix + ".position", Vec3(0, 0, 0));
         Vec3 obj_scale = parser.getVec3(prefix + ".scale", Vec3(1, 1, 1));

         // Resolve path relative to scene file directory
         string obj_path = obj_file;
         if (!obj_file.empty() && obj_file[0] != '/')
            obj_path = scene_dir + "/" + obj_file;

         // Record the start index so we can apply visibility to the whole range
         int geom_start = static_cast<int>(scene.geometries.size());
         int tri_count = OBJLoader::loadOBJ(obj_path, scene, mat_id, obj_position, obj_scale);
         if (tri_count < 0)
         {
            cerr << "ERROR: Failed to load OBJ file: " << obj_path << "\n";
            continue; // Skip success log and generic visibility assignment
         }
         if (!visible)
         {
            // Apply visibility to all triangles added by this OBJ
            for (int gi = geom_start; gi < static_cast<int>(scene.geometries.size()); ++gi)
               scene.geometries[gi].visible = false;
         }
         // Skip the generic visibility assignment below (already handled)
         continue;
      }

      // Apply visibility flag to the last added geometry
      if (!scene.geometries.empty())
         scene.geometries.back().visible = visible;
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

   // Clear existing scene
   scene.materials.clear();
   scene.geometries.clear();
   scene.meshes.clear();

   scene.source_file = filename;

   // Load camera settings (optional)
   if (parser.hasKey("camera.position"))
      scene.camera_position = parser.getVec3("camera.position");
   if (parser.hasKey("camera.look_at"))
      scene.camera_look_at = parser.getVec3("camera.look_at");
   if (parser.hasKey("camera.up"))
      scene.camera_up = parser.getVec3("camera.up");
   if (parser.hasKey("camera.fov"))
      scene.camera_fov = parser.getFloat("camera.fov", 90.0f);

   // Load scene settings (optional)
   if (parser.hasKey("settings.background_color"))
      scene.background_color = parser.getVec3("settings.background_color");
   if (parser.hasKey("settings.background_intensity"))
      scene.background_intensity = parser.getFloat("settings.background_intensity", 1.0f);
   if (parser.hasKey("settings.light_intensity"))
      scene.light_intensity = parser.getFloat("settings.light_intensity", 1.0f);
   if (parser.hasKey("settings.light_falloff"))
      scene.light_falloff = std::max(0.0f, parser.getFloat("settings.light_falloff", 2.0f));
   if (parser.hasKey("settings.samples_per_batch"))
      scene.samples_per_batch = std::max(0, parser.getInt("settings.samples_per_batch", 0));
   // Accept both "settings.use_bvh" (canonical) and "scene.use_bvh" (legacy)
   {
      string bvh_raw;
      if (parser.hasKey("settings.use_bvh"))
         bvh_raw = parser.getString("settings.use_bvh", "false");
      else if (parser.hasKey("scene.use_bvh"))
         bvh_raw = parser.getString("scene.use_bvh", "false");
      if (!bvh_raw.empty())
         scene.use_bvh = (removeQuotes(bvh_raw) == "true" || removeQuotes(bvh_raw) == "1");
   }
   if (parser.hasKey("settings.adaptive_sampling"))
   {
      string val = removeQuotes(parser.getString("settings.adaptive_sampling", "false"));
      scene.adaptive_sampling = (val == "true" || val == "1");
   }

   // Derive scene directory for resolving relative OBJ paths
   string scene_file(filename);
   string scene_dir = ".";
   size_t last_slash = scene_file.find_last_of("/\\");
   if (last_slash != string::npos)
      scene_dir = scene_file.substr(0, last_slash);

   // Load materials first
   map<string, int> material_name_to_id;
   if (!loadMaterials(parser, scene, material_name_to_id, scene_dir))
   {
      cerr << "ERROR: Failed to load materials"
              "\n";
      return false;
   }

   // Then load geometry (which references materials)
   if (!loadGeometry(parser, scene, material_name_to_id, scene_dir))
   {
      cerr << "ERROR: Failed to load geometry"
              "\n";
      return false;
   }
   // Volumes read from a density grid need no geometry of their own: their box is the boundary
   for (int i = 0; i < static_cast<int>(scene.materials.size()); ++i)
      if (!scene.materials[i].grid_file.empty())
         addVolumeGridBox(scene, i);

   cout << "Scene loaded successfully:" << "\n";
   cout << "  - " << scene.materials.size() << " materials" << "\n";
   cout << "  - " << scene.geometries.size() << " objects" << "\n";

   return scene.validate();
}

namespace
{
// Rewrite the given `key: value` lines of a top-level YAML block in place (lines are added
// after the block header when missing, the block itself is appended when missing); every
// other line of the file is kept as is.
void setBlockValues(vector<string> &lines, const string &block, const vector<pair<string, string>> &values)
{
   auto indent_of = [](const string &l) { return l.find_first_not_of(' '); };
   auto key_of    = [](const string &l)
   {
      const size_t first = l.find_first_not_of(' ');
      string t = l.substr(first == string::npos ? 0 : first);
      const size_t colon = t.find(':');
      return colon == string::npos ? string() : t.substr(0, colon);
   };
   size_t begin = lines.size();
   for (size_t i = 0; i < lines.size(); ++i)
      if (indent_of(lines[i]) == 0 && key_of(lines[i]) == block)
      {
         begin = i;
         break;
      }
   if (begin == lines.size())
   {
      if (!lines.empty() && !lines.back().empty())
         lines.push_back("");
      lines.push_back(block + ":");
      for (const auto &kv : values)
         lines.push_back("  " + kv.first + ": " + kv.second);
      return;
   }
   size_t end = begin + 1;
   while (end < lines.size())
   {
      const size_t ind = indent_of(lines[end]);
      if (ind == 0 && !lines[end].empty() && lines[end][0] != '#')
         break; // next top-level key
      ++end;
   }
   vector<bool> done(values.size(), false);
   for (size_t i = begin + 1; i < end; ++i)
   {
      const size_t ind = indent_of(lines[i]);
      if (ind == string::npos || lines[i][ind] == '#')
         continue;
      const string key = key_of(lines[i]);
      for (size_t k = 0; k < values.size(); ++k)
         if (key == values[k].first)
         {
            // Keep a trailing comment if there is one
            const size_t hash    = lines[i].find(" #");
            const string comment = hash == string::npos ? string() : lines[i].substr(hash);
            lines[i] = "  " + key + ": " + values[k].second + comment;
            done[k]  = true;
         }
   }
   // Missing keys go right after the block header, in the given order
   for (size_t k = values.size(); k-- > 0;)
      if (!done[k])
         lines.insert(lines.begin() + begin + 1, "  " + values[k].first + ": " + values[k].second);
}

string fmtNumber(double v)
{
   ostringstream os;
   os.setf(ios::fixed);
   os.precision(3);
   os << v;
   string t = os.str(); // trim trailing zeros: 2.100 -> 2.1, 40.000 -> 40
   t.erase(t.find_last_not_of('0') + 1);
   if (!t.empty() && t.back() == '.')
      t.pop_back();
   return t == "-0" ? string("0") : t;
}
} // namespace

bool saveViewToYAML(const char *filename, const Vec3 &position, const Vec3 &look_at, float fov,
                    float background_intensity, float light_intensity, float light_falloff)
{
   ifstream in(filename);
   if (!in)
   {
      cerr << "ERROR: Cannot open scene file for update: " << filename << "\n";
      return false;
   }
   vector<string> lines;
   for (string line; getline(in, line);)
      lines.push_back(line);
   in.close();

   auto vec = [](const Vec3 &v)
   { return "[" + fmtNumber(v.x()) + ", " + fmtNumber(v.y()) + ", " + fmtNumber(v.z()) + "]"; };
   setBlockValues(lines, "camera", {{"position", vec(position)}, {"look_at", vec(look_at)}, {"fov", fmtNumber(fov)}});
   setBlockValues(lines, "settings",
                  {{"background_intensity", fmtNumber(background_intensity)},
                   {"light_intensity", fmtNumber(light_intensity)},
                   {"light_falloff", fmtNumber(light_falloff)}});

   ofstream out(filename, ios::trunc);
   if (!out)
   {
      cerr << "ERROR: Cannot write scene file: " << filename << "\n";
      return false;
   }
   for (const auto &l : lines)
      out << l << "\n";
   cout << "Scene view saved to " << filename << ": camera " << vec(position) << " -> " << vec(look_at) << ", fov "
        << fmtNumber(fov) << ", background_intensity " << fmtNumber(background_intensity) << ", light_intensity "
        << fmtNumber(light_intensity) << ", light_falloff " << fmtNumber(light_falloff) << "\n";
   return true;
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
      case MaterialType::ANISOTROPIC_METAL:
         file << "\"anisotropic_metal\"";
         break;
      case MaterialType::THIN_FILM:
         file << "\"thin_film\"";
         break;
      case MaterialType::CLEAR_COAT:
         file << "\"clear_coat\"";
         break;
      case MaterialType::VOLUME:
         file << "\"volume\"";
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
      if (mat.anisotropy > 0)
      {
         file << "    anisotropy: " << mat.anisotropy << "\n";
      }
      if (mat.eta.x() > 0 || mat.eta.y() > 0 || mat.eta.z() > 0)
      {
         file << "    eta: [" << mat.eta.x() << ", " << mat.eta.y() << ", " << mat.eta.z() << "]\n";
         file << "    k: [" << mat.k.x() << ", " << mat.k.y() << ", " << mat.k.z() << "]\n";
      }

      // Participating medium
      if (mat.type == MaterialType::VOLUME)
      {
         file << "    density: " << mat.density << "\n";
         file << "    g: " << mat.anisotropy << "\n";
         file << "    noise_scale: " << mat.noise_scale << "\n";
         file << "    noise_strength: " << mat.noise_strength << "\n";
         if (!mat.grid_file.empty())
         {
            file << "    grid: \"" << mat.grid_file << "\"\n";
            file << "    grid_scale: " << mat.grid_scale << "\n";
            file << "    grid_position: [" << mat.grid_position.x() << ", " << mat.grid_position.y() << ", "
                 << mat.grid_position.z() << "]\n";
         }
         else
         {
            file << "    edge: " << mat.edge << "\n";
            file << "    lumps: " << mat.lumps << "\n";
            file << "    billow: " << mat.billow << "\n";
         }
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

      case GeometryType::SDF_PRIMITIVE:
      {
         static const char *const shape_names[] = {"sphere",     "box",        "torus",             "capsule",
                                                   "cylinder",   "mandelbulb", "death_star",        "cut_hollow_sphere",
                                                   "octahedron", "pyramid",    "custom"};
         const auto &sdf = geom.data.sdf;
         const int   t   = static_cast<int>(sdf.sdf_type);
         file << "\"sdf\"\n";
         file << "    material: \"material_" << geom.material_id << "\"\n";
         file << "    shape: \"" << (t >= 0 && t < 11 ? shape_names[t] : "custom") << "\"\n";
         file << "    center: [" << sdf.position.x() << ", " << sdf.position.y() << ", " << sdf.position.z() << "]\n";
         file << "    params: [" << sdf.parameters.x() << ", " << sdf.parameters.y() << ", " << sdf.parameters.z()
              << "]\n";
         const double r2d = 180.0 / M_PI;
         file << "    rotation: [" << sdf.rotation.x() * r2d << ", " << sdf.rotation.y() * r2d << ", "
              << sdf.rotation.z() * r2d << "]\n";
         break;
      }

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
