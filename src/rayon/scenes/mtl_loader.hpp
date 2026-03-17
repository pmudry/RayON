/**
 * @file mtl_loader.hpp
 * @brief Wavefront MTL (material library) parser
 *
 * Parses .mtl files referenced by .obj files and converts them into
 * MaterialDesc entries. Supports diffuse color (Kd), emission (Ke),
 * roughness (Ns→roughness), IOR (Ni), transmission (d/Tr), illumination
 * model heuristics, and map_Kd diffuse texture maps.
 */
#pragma once

#include "scene_description.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace Scene
{

/**
 * @brief Intermediate MTL entry before registering in the scene
 */
struct MtlEntry
{
   MaterialDesc mat;
   std::string diffuse_tex_path; // Empty if no map_Kd
};

class MTLLoader
{
 public:
   /**
    * @brief Parse a .mtl file and return a map of material-name → MtlEntry
    * @param filename  Absolute path to the .mtl file
    * @param mtl_dir   Directory of the .mtl file (for relative texture paths)
    * @return map of name → MtlEntry (empty map on failure)
    */
   static std::map<std::string, MtlEntry> load(const std::string &filename, const std::string &mtl_dir)
   {
      std::map<std::string, MtlEntry> result;

      std::ifstream file(filename);
      if (!file.is_open())
      {
         std::cerr << "MTL Loader: Cannot open '" << filename << "'\n";
         return result;
      }

      std::string current_name;
      MtlEntry current_entry;
      bool has_current = false;

      // Working variables per material
      float Kd[3] = {0.7f, 0.7f, 0.7f};
      float Ke[3] = {0.0f, 0.0f, 0.0f};
      float Ns = 100.0f; // specular exponent
      float Ni = 1.0f;
      float d = 1.0f;    // opacity (1 = opaque)
      int illum = 2;
      std::string map_Kd;

      auto commitMaterial = [&]() {
         if (!has_current) return;

         // Compute roughness from Ns (specular exponent, Blinn-Phong)
         // Ns ∈ [0, 1000]; roughness ≈ 1 - sqrt(Ns/1000)
         float roughness = 1.0f - std::sqrt(std::max(0.0f, std::min(Ns / 1000.0f, 1.0f)));

         // Determine material type from illum and Ke
         float ke_len = Ke[0] + Ke[1] + Ke[2];
         float transmission = 1.0f - d;

         MaterialDesc &mat = current_entry.mat;
         if (ke_len > 0.3f)
         {
            mat = MaterialDesc::light(Vec3(Ke[0], Ke[1], Ke[2]));
         }
         else if (illum == 5 || illum == 7 || transmission > 0.5f)
         {
            mat = MaterialDesc::glass(Ni > 1.0f ? Ni : 1.45f);
            mat.transmission = transmission;
         }
         else if (roughness < 0.05f && illum >= 2)
         {
            mat = MaterialDesc::roughMirror(Vec3(Kd[0], Kd[1], Kd[2]), roughness);
         }
         else
         {
            mat = MaterialDesc::lambertian(Vec3(Kd[0], Kd[1], Kd[2]));
            if (roughness > 0.01f && illum >= 2)
            {
               mat = MaterialDesc::roughMirror(Vec3(Kd[0], Kd[1], Kd[2]), roughness);
            }
         }

         result[current_name] = current_entry;
      };

      auto resetWorkingVars = [&]() {
         Kd[0] = Kd[1] = Kd[2] = 0.7f;
         Ke[0] = Ke[1] = Ke[2] = 0.0f;
         Ns = 100.0f;
         Ni = 1.0f;
         d = 1.0f;
         illum = 2;
         map_Kd.clear();
         current_entry = MtlEntry{};
      };

      std::string line;
      while (std::getline(file, line))
      {
         // Strip comment
         auto hash = line.find('#');
         if (hash != std::string::npos) line.resize(hash);

         // Trim
         size_t first = line.find_first_not_of(" \t\r\n");
         if (first == std::string::npos) continue;
         line = line.substr(first);

         std::istringstream iss(line);
         std::string keyword;
         iss >> keyword;
         // normalise keyword to lowercase
         std::transform(keyword.begin(), keyword.end(), keyword.begin(),
                        [](unsigned char c) { return std::tolower(c); });

         if (keyword == "newmtl")
         {
            commitMaterial();
            resetWorkingVars();
            std::getline(iss, current_name);
            // Trim whitespace from name
            size_t s = current_name.find_first_not_of(" \t");
            if (s != std::string::npos) current_name = current_name.substr(s);
            size_t e = current_name.find_last_not_of(" \t\r\n");
            if (e != std::string::npos) current_name = current_name.substr(0, e + 1);
            has_current = !current_name.empty();
         }
         else if (keyword == "kd")
         {
            iss >> Kd[0] >> Kd[1] >> Kd[2];
         }
         else if (keyword == "ke")
         {
            iss >> Ke[0] >> Ke[1] >> Ke[2];
         }
         else if (keyword == "ns")
         {
            iss >> Ns;
         }
         else if (keyword == "ni")
         {
            iss >> Ni;
         }
         else if (keyword == "d")
         {
            iss >> d;
         }
         else if (keyword == "tr")
         {
            float tr = 0.0f;
            iss >> tr;
            d = 1.0f - tr;
         }
         else if (keyword == "illum")
         {
            iss >> illum;
         }
         else if (keyword == "map_kd")
         {
            std::string tex_path;
            // map_Kd may have options starting with '-'; skip them
            std::string token;
            while (iss >> token)
            {
               if (token[0] == '-')
               {
                  iss >> token; // skip option value
               }
               else
               {
                  tex_path = token;
               }
            }
            if (!tex_path.empty())
            {
               // Resolve relative to mtl directory
               if (!mtl_dir.empty() && !std::filesystem::path(tex_path).is_absolute())
                  tex_path = mtl_dir + "/" + tex_path;
               map_Kd = tex_path;
               current_entry.diffuse_tex_path = map_Kd;
            }
         }
         // Silently ignore: Ka, Ks, map_Ks, bump, refl, etc.
      }

      commitMaterial();

      std::cout << "MTL Loader: Loaded " << result.size() << " material(s) from '" << filename << "'\n";
      return result;
   }
};

} // namespace Scene
