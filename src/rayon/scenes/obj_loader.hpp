/**
 * @file obj_loader.hpp
 * @brief Wavefront OBJ file loader for triangle meshes
 *
 * Lightweight parser supporting vertices, vertex normals, texture coordinates,
 * and face indices. Converts OBJ data into TriangleMesh for the scene system.
 */
#pragma once

#include "scene_description.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace Scene
{

class OBJLoader
{
 public:
   /**
    * @brief Load an OBJ file and add its triangles to the scene
    * @param filename Path to .obj file
    * @param scene Scene to add triangles to
    * @param mat_id Material ID to apply to all triangles
    * @param position Translation offset
    * @param scale Uniform or per-axis scale
    * @return Number of triangles loaded, or -1 on failure
    */
   static int loadOBJ(const std::string &filename, SceneDescription &scene, int mat_id,
                       const Vec3 &position = Vec3(0, 0, 0), const Vec3 &scale = Vec3(1, 1, 1))
   {
      std::ifstream file(filename);
      if (!file.is_open())
      {
         std::cerr << "OBJ Loader: Cannot open file: " << filename << "\n";
         return -1;
      }

      std::vector<Vec3> vertices;
      std::vector<Vec3> normals;
      std::vector<Vec3> texcoords;
      int triangle_count = 0;

      std::string line;
      while (std::getline(file, line))
      {
         // Skip empty lines and comments
         if (line.empty() || line[0] == '#')
            continue;

         std::istringstream iss(line);
         std::string prefix;
         iss >> prefix;

         if (prefix == "v")
         {
            // Vertex position
            double x, y, z;
            if (iss >> x >> y >> z)
               vertices.emplace_back(x * scale.x() + position.x(), y * scale.y() + position.y(),
                                     z * scale.z() + position.z());
         }
         else if (prefix == "vn")
         {
            // Vertex normal
            double x, y, z;
            if (iss >> x >> y >> z)
               normals.emplace_back(x, y, z);
         }
         else if (prefix == "vt")
         {
            // Texture coordinate (stored for future use)
            double u, v;
            if (iss >> u >> v)
               texcoords.emplace_back(u, v, 0);
         }
         else if (prefix == "f")
         {
            // Face — support triangles and quads (fan triangulation for polygons)
            std::vector<int> v_indices, vn_indices;
            std::string token;

            while (iss >> token)
            {
               int vi = 0, vti = 0, vni = 0;
               parseFaceVertex(token, vi, vti, vni);

               // OBJ indices are 1-based; convert to 0-based. Negative = relative.
               if (vi > 0)
                  vi -= 1;
               else if (vi < 0)
                  vi = static_cast<int>(vertices.size()) + vi;

               if (vni > 0)
                  vni -= 1;
               else if (vni < 0)
                  vni = static_cast<int>(normals.size()) + vni;

               v_indices.push_back(vi);
               vn_indices.push_back(vni);
            }

            // Fan triangulation for polygons with 3+ vertices
            for (size_t i = 1; i + 1 < v_indices.size(); ++i)
            {
               int i0 = v_indices[0], i1 = v_indices[i], i2 = v_indices[i + 1];
               if (i0 < 0 || i0 >= (int)vertices.size() || i1 < 0 || i1 >= (int)vertices.size() || i2 < 0 ||
                   i2 >= (int)vertices.size())
                  continue;

               bool has_norms = !normals.empty() && vn_indices[0] >= 0 && vn_indices[0] < (int)normals.size() &&
                                vn_indices[i] >= 0 && vn_indices[i] < (int)normals.size() && vn_indices[i + 1] >= 0 &&
                                vn_indices[i + 1] < (int)normals.size();

               if (has_norms)
               {
                  scene.addTriangleWithNormals(vertices[i0], vertices[i1], vertices[i2], normals[vn_indices[0]],
                                               normals[vn_indices[i]], normals[vn_indices[i + 1]], mat_id);
               }
               else
               {
                  scene.addTriangle(vertices[i0], vertices[i1], vertices[i2], mat_id);
               }
               ++triangle_count;
            }
         }
         // Silently skip: mtllib, usemtl, s, g, o, etc.
      }

      std::cout << "OBJ Loader: Loaded " << triangle_count << " triangles from " << filename << " (" << vertices.size()
                << " vertices, " << normals.size() << " normals)\n";

      return triangle_count;
   }

 private:
   /**
    * @brief Parse a face vertex token in format: v, v/vt, v/vt/vn, or v//vn
    */
   static void parseFaceVertex(const std::string &token, int &vi, int &vti, int &vni)
   {
      vi = vti = vni = 0;
      std::istringstream stream(token);
      std::string part;

      // First component: vertex index (always present)
      if (std::getline(stream, part, '/'))
      {
         if (!part.empty())
            vi = std::stoi(part);
      }

      // Second component: texture coordinate index (optional)
      if (std::getline(stream, part, '/'))
      {
         if (!part.empty())
            vti = std::stoi(part);
      }

      // Third component: normal index (optional)
      if (std::getline(stream, part, '/'))
      {
         if (!part.empty())
            vni = std::stoi(part);
      }
   }
};

} // namespace Scene
