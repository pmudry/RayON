/**
 * @file obj_loader.hpp
 * @brief Wavefront OBJ file loader for triangle meshes
 *
 * Lightweight parser supporting vertices, vertex normals, texture coordinates,
 * face indices, mtllib references, and usemtl group assignments.
 * Per-group materials come from the referenced .mtl file; the caller-supplied
 * mat_id acts as a fallback for groups that have no usemtl directive.
 */
#pragma once

#include "mtl_loader.hpp"
#include "scene_description.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace Scene
{

class OBJLoader
{
 public:
   /**
    * @brief Load an OBJ file and add its triangles to the scene.
    *
    * Materials are resolved from the .mtl file referenced by mtllib.
    * The caller-supplied @p fallback_mat_id is used only for faces that
    * belong to a group with no usemtl directive (or when no .mtl is found).
    * Pass -1 for @p fallback_mat_id to require all materials to come from
    * the .mtl file (faces without usemtl are then skipped with a one-time warning).
    *
    * @param filename         Path to .obj file
    * @param scene            Scene to add triangles to
    * @param fallback_mat_id  Material ID for faces without usemtl (-1 = MTL-only)
    * @param position         Translation offset
    * @param scale            Per-axis scale
    * @return Number of triangles loaded, or -1 on failure
    */
   static int loadOBJ(const std::string &filename, SceneDescription &scene, int fallback_mat_id,
                       const Vec3 &position = Vec3(0, 0, 0), const Vec3 &scale = Vec3(1, 1, 1))
   {
      std::ifstream file(filename);
      if (!file.is_open())
      {
         std::cerr << "OBJ Loader: Cannot open file: " << filename << "\n";
         return -1;
      }

      // Determine directory of the OBJ file for relative path resolution
      std::string obj_dir;
      auto last_slash = filename.find_last_of("/\\");
      if (last_slash != std::string::npos)
         obj_dir = filename.substr(0, last_slash);

      std::vector<Vec3> vertices;
      std::vector<Vec3> normals;
      std::vector<Vec3> texcoords;         // UV stored as (u, v, 0)
      int triangle_count = 0;

      // MTL data (populated when mtllib line is found)
      std::map<std::string, MtlEntry> mtl_materials;
      // Map from MTL material name → scene material id (lazily registered)
      std::map<std::string, int> mtl_name_to_scene_id;

      int active_mat_id = fallback_mat_id; // Current material for faces
      bool face_skip_warned = false;       // One-time warning when faces are skipped without a material

      auto resolveMtlMaterial = [&](const std::string &name) -> int {
         auto cached = mtl_name_to_scene_id.find(name);
         if (cached != mtl_name_to_scene_id.end())
            return cached->second;

         auto entry_it = mtl_materials.find(name);
         if (entry_it == mtl_materials.end())
         {
            std::cerr << "OBJ Loader: Unknown MTL material '" << name << "', using fallback\n";
            return fallback_mat_id;
         }

         MtlEntry &entry = entry_it->second;
         // Load texture if present
         if (!entry.diffuse_tex_path.empty())
         {
            int tex_id = scene.addTexture(entry.diffuse_tex_path);
            entry.mat.texture_id = tex_id;
         }
         int scene_id = scene.addMaterial(entry.mat);
         mtl_name_to_scene_id[name] = scene_id;
         return scene_id;
      };

      std::string line;
      while (std::getline(file, line))
      {
         // Strip comment
         auto hash = line.find('#');
         if (hash != std::string::npos) line.resize(hash);

         if (line.empty()) continue;

         std::istringstream iss(line);
         std::string prefix;
         iss >> prefix;

         if (prefix == "v")
         {
            double x, y, z;
            if (iss >> x >> y >> z)
               vertices.emplace_back(x * scale.x() + position.x(),
                                     y * scale.y() + position.y(),
                                     z * scale.z() + position.z());
         }
         else if (prefix == "vn")
         {
            double x, y, z;
            if (iss >> x >> y >> z)
               normals.emplace_back(x, y, z);
         }
         else if (prefix == "vt")
         {
            double u, v;
            if (iss >> u >> v)
               texcoords.emplace_back(u, v, 0.0);
         }
         else if (prefix == "mtllib")
         {
            std::string mtl_file;
            std::getline(iss, mtl_file);
            // Trim
            size_t s = mtl_file.find_first_not_of(" \t");
            if (s != std::string::npos) mtl_file = mtl_file.substr(s);
            size_t e = mtl_file.find_last_not_of(" \t\r\n");
            if (e != std::string::npos) mtl_file = mtl_file.substr(0, e + 1);

            std::string mtl_path = mtl_file;
            if (!obj_dir.empty() && !mtl_file.empty() && !std::filesystem::path(mtl_file).is_absolute())
               mtl_path = obj_dir + "/" + mtl_file;

            // Compute the directory containing the .mtl file itself (may differ
            // from obj_dir when mtllib has a subdirectory prefix, e.g. "mtl/foo.mtl").
            std::string mtl_dir = obj_dir;
            auto mtl_slash = mtl_path.find_last_of("/\\");
            if (mtl_slash != std::string::npos)
               mtl_dir = mtl_path.substr(0, mtl_slash);

            mtl_materials = MTLLoader::load(mtl_path, mtl_dir);
         }
         else if (prefix == "usemtl")
         {
            std::string mat_name;
            std::getline(iss, mat_name);
            size_t s = mat_name.find_first_not_of(" \t");
            if (s != std::string::npos) mat_name = mat_name.substr(s);
            size_t e = mat_name.find_last_not_of(" \t\r\n");
            if (e != std::string::npos) mat_name = mat_name.substr(0, e + 1);

            if (!mat_name.empty())
               active_mat_id = resolveMtlMaterial(mat_name);
         }
         else if (prefix == "f")
         {
            if (active_mat_id < 0)
            {
               // No material yet and no fallback — skip with a one-time warning per file
               if (!face_skip_warned)
               {
                  std::cerr << "OBJ Loader: face(s) skipped — no 'usemtl' encountered and no fallback material provided\n";
                  face_skip_warned = true;
               }
               continue;
            }

            // Face — support triangles and quads (fan triangulation)
            std::vector<int> v_idx, vt_idx, vn_idx;
            std::string token;

            while (iss >> token)
            {
               int vi = 0, vti = 0, vni = 0;
               parseFaceVertex(token, vi, vti, vni);

               if (vi > 0) vi -= 1;
               else if (vi < 0) vi = static_cast<int>(vertices.size()) + vi;

               if (vti > 0) vti -= 1;
               else if (vti < 0) vti = static_cast<int>(texcoords.size()) + vti;
               else vti = -1; // 0 means absent

               if (vni > 0) vni -= 1;
               else if (vni < 0) vni = static_cast<int>(normals.size()) + vni;
               else vni = -1;

               v_idx.push_back(vi);
               vt_idx.push_back(vti);
               vn_idx.push_back(vni);
            }

            // Fan triangulation for polygons with 3+ vertices
            for (size_t i = 1; i + 1 < v_idx.size(); ++i)
            {
               int i0 = v_idx[0], i1 = v_idx[i], i2 = v_idx[i + 1];
               if (i0 < 0 || i0 >= (int)vertices.size() ||
                   i1 < 0 || i1 >= (int)vertices.size() ||
                   i2 < 0 || i2 >= (int)vertices.size())
                  continue;

               bool has_norms = (vn_idx[0] >= 0 && vn_idx[0] < (int)normals.size() &&
                                 vn_idx[i] >= 0 && vn_idx[i] < (int)normals.size() &&
                                 vn_idx[i + 1] >= 0 && vn_idx[i + 1] < (int)normals.size());

               bool has_tex = (vt_idx[0] >= 0 && vt_idx[0] < (int)texcoords.size() &&
                               vt_idx[i] >= 0 && vt_idx[i] < (int)texcoords.size() &&
                               vt_idx[i + 1] >= 0 && vt_idx[i + 1] < (int)texcoords.size());

               if (has_norms && has_tex)
               {
                  scene.addTriangleWithNormalsAndUVs(
                      vertices[i0], vertices[i1], vertices[i2],
                      normals[vn_idx[0]], normals[vn_idx[i]], normals[vn_idx[i + 1]],
                      texcoords[vt_idx[0]], texcoords[vt_idx[i]], texcoords[vt_idx[i + 1]],
                      active_mat_id);
               }
               else if (has_norms)
               {
                  scene.addTriangleWithNormals(
                      vertices[i0], vertices[i1], vertices[i2],
                      normals[vn_idx[0]], normals[vn_idx[i]], normals[vn_idx[i + 1]],
                      active_mat_id);
               }
               else if (has_tex)
               {
                  scene.addTriangleWithUVs(
                      vertices[i0], vertices[i1], vertices[i2],
                      texcoords[vt_idx[0]], texcoords[vt_idx[i]], texcoords[vt_idx[i + 1]],
                      active_mat_id);
               }
               else
               {
                  scene.addTriangle(vertices[i0], vertices[i1], vertices[i2], active_mat_id);
               }
               ++triangle_count;
            }
         }
         // Silently skip: s, g, o, l, etc.
      }

      std::cout << "OBJ Loader: Loaded " << triangle_count << " triangles from " << filename
                << " (" << vertices.size() << " verts, " << normals.size() << " normals, "
                << texcoords.size() << " UVs)\n";

      return triangle_count;
   }

 private:
   /**
    * @brief Parse a face vertex token: v, v/vt, v/vt/vn, or v//vn
    *        Returns 0 for absent components (caller converts to -1 after checking).
    */
   static void parseFaceVertex(const std::string &token, int &vi, int &vti, int &vni)
   {
      vi = vti = vni = 0;
      std::istringstream stream(token);
      std::string part;

      if (std::getline(stream, part, '/'))
         if (!part.empty()) vi = std::stoi(part);

      if (std::getline(stream, part, '/'))
         if (!part.empty()) vti = std::stoi(part);

      if (std::getline(stream, part, '/'))
         if (!part.empty()) vni = std::stoi(part);
   }
};

} // namespace Scene
