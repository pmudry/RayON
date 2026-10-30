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
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <string_view>
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
      // The whole file in memory, parsed in place: a scan of a million faces spends its time in
      // per-line stream objects otherwise
      std::string text;
      {
         std::ifstream file(filename, std::ios::binary);
         if (!file.is_open())
         {
            std::cerr << "OBJ Loader: Cannot open file: " << filename << "\n";
            return -1;
         }
         text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
      }

      // One geometry per triangle: reserve for the faces (at least one triangle each) up front
      {
         size_t faces = (text.compare(0, 2, "f ") == 0) ? 1 : 0;
         for (const char *q = text.c_str(); (q = std::strstr(q, "\nf ")) != nullptr; q += 3)
            ++faces;
         scene.geometries.reserve(scene.geometries.size() + faces);
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

      std::vector<int> v_idx, vt_idx, vn_idx; // face corners, reused from face to face
      size_t pos = 0;
      while (pos < text.size())
      {
         size_t eol = text.find('\n', pos);
         if (eol == std::string::npos) eol = text.size();
         std::string_view line(text.data() + pos, eol - pos);
         pos = eol + 1;

         // Strip comment
         auto hash = line.find('#');
         if (hash != std::string_view::npos) line = line.substr(0, hash);

         const char *cur = line.data();
         const char *end = cur + line.size();
         const std::string_view prefix = nextToken(cur, end);
         if (prefix.empty()) continue;

         if (prefix == "v")
         {
            double x, y, z;
            if (parseDouble(cur, end, x) && parseDouble(cur, end, y) && parseDouble(cur, end, z))
               vertices.emplace_back(x * scale.x() + position.x(),
                                     y * scale.y() + position.y(),
                                     z * scale.z() + position.z());
         }
         else if (prefix == "vn")
         {
            double x, y, z;
            if (parseDouble(cur, end, x) && parseDouble(cur, end, y) && parseDouble(cur, end, z))
               normals.emplace_back(x, y, z);
         }
         else if (prefix == "vt")
         {
            double u, v = 0.0; // a one-component "vt u" is legal: keep it, or every later index shifts
            if (parseDouble(cur, end, u))
            {
               parseDouble(cur, end, v);
               texcoords.emplace_back(u, v, 0.0);
            }
         }
         else if (prefix == "mtllib")
         {
            std::string mtl_file = trimmed(std::string_view(cur, end - cur));

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
            std::string mat_name = trimmed(std::string_view(cur, end - cur));

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
            v_idx.clear();
            vt_idx.clear();
            vn_idx.clear();

            for (std::string_view token = nextToken(cur, end); !token.empty(); token = nextToken(cur, end))
            {
               int vi = 0, vti = 0, vni = 0;
               parseFaceVertex(token, vi, vti, vni);

               if (vi > 0) vi -= 1;
               else if (vi < 0) vi = static_cast<int>(vertices.size()) + vi;
               else vi = -1; // 0 or unreadable: not a vertex (the triangle is dropped below)

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

      return triangle_count;
   }

 private:
   static bool isBlank(char c) { return c == ' ' || c == '\t' || c == '\r'; }

   /// Next whitespace-separated token of [cur, end), advancing cur past it (empty at the end)
   static std::string_view nextToken(const char *&cur, const char *end)
   {
      while (cur < end && isBlank(*cur)) ++cur;
      const char *start = cur;
      while (cur < end && !isBlank(*cur)) ++cur;
      return std::string_view(start, cur - start);
   }

   /// Read one number from [cur, end); false when there is none
   static bool parseDouble(const char *&cur, const char *end, double &out)
   {
      while (cur < end && isBlank(*cur)) ++cur;
      if (cur < end && *cur == '+') ++cur; // from_chars does not take a leading '+'
      const auto res = std::from_chars(cur, end, out);
      if (res.ec != std::errc()) return false;
      cur = res.ptr;
      return true;
   }

   static std::string trimmed(std::string_view s)
   {
      while (!s.empty() && isBlank(s.front())) s.remove_prefix(1);
      while (!s.empty() && isBlank(s.back())) s.remove_suffix(1);
      return std::string(s);
   }

   /**
    * @brief Parse a face vertex token: v, v/vt, v/vt/vn, or v//vn
    *        Returns 0 for absent or unreadable components (caller converts to -1 after checking).
    */
   static void parseFaceVertex(std::string_view token, int &vi, int &vti, int &vni)
   {
      vi = vti = vni = 0;
      int *parts[3] = {&vi, &vti, &vni};
      const char *cur = token.data();
      const char *end = cur + token.size();
      for (int k = 0; k < 3; ++k)
      {
         const char *slash = static_cast<const char *>(std::memchr(cur, '/', end - cur));
         const char *stop = slash ? slash : end;
         if (stop > cur)
         {
            const char *num = (*cur == '+') ? cur + 1 : cur;
            if (std::from_chars(num, stop, *parts[k]).ec != std::errc()) *parts[k] = 0;
         }
         if (!slash) break;
         cur = slash + 1;
      }
   }
};

} // namespace Scene
