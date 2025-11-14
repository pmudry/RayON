/**
 * @file scene_loader.h
 * @brief YAML scene loader for SceneDescription
 * 
 * Loads scenes from YAML files into the unified SceneDescription format.
 * Uses a simple, dependency-free YAML parser.
 */

#pragma once

// Include scene_description first, before any "using namespace std" contamination
#include "scene_description.hpp"

#include <fstream>
#include <sstream>
#include <map>
#include <cctype>
#include <algorithm>
#include <iostream>

namespace Scene {

/**
 * @brief Simple YAML-like scene file parser
 * 
 * Parses a subset of YAML suitable for scene files:
 * - Key: value pairs
 * - Arrays using [x, y, z] notation  
 * - List items with '- ' prefix
 * - Comments starting with '#'
 * - Indentation-based nesting
 */
class SimpleSceneParser {
private:
    std::map<std::string, std::string> values_;
    
    static std::string trimWhitespace(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, last - first + 1);
    }
    
    static std::string removeComment(const std::string& line) {
        size_t pos = line.find('#');
        if (pos != std::string::npos) {
            return line.substr(0, pos);
        }
        return line;
    }
    
public:
    bool parseFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "ERROR: Cannot open scene file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        std::vector<std::string> section_stack;
        int current_indent = -1;
        int material_index = 0;
        int geometry_index = 0;
        bool in_materials = false;
        bool in_geometry = false;
        
        while (std::getline(file, line)) {
            // Remove comments and trim
            line = removeComment(line);
            std::string trimmed = trimWhitespace(line);
            if (trimmed.empty()) continue;
            
            // Calculate indentation
            int indent = 0;
            while (indent < (int)line.length() && (line[indent] == ' ' || line[indent] == '\t')) {
                indent++;
            }
            
            // Check for section headers
            if (trimmed == "materials:") {
                in_materials = true;
                in_geometry = false;
                material_index = 0;
                continue;
            } else if (trimmed == "geometry:") {
                in_materials = false;
                in_geometry = true;
                geometry_index = 0;
                continue;
            }
            
            // Handle list items
            if (trimmed[0] == '-' && trimmed[1] == ' ') {
                trimmed = trimWhitespace(trimmed.substr(2));
                if (in_materials) {
                    section_stack.clear();
                    section_stack.push_back("material" + std::to_string(material_index));
                    material_index++;
                } else if (in_geometry) {
                    section_stack.clear();
                    section_stack.push_back("geom" + std::to_string(geometry_index));
                    geometry_index++;
                }
            }
            
            // Parse key-value pairs
            size_t colon_pos = trimmed.find(':');
            if (colon_pos != std::string::npos) {
                std::string key = trimWhitespace(trimmed.substr(0, colon_pos));
                std::string value = trimWhitespace(trimmed.substr(colon_pos + 1));
                
                // Build full key path
                std::string full_key = key;
                for (const auto& section : section_stack) {
                    full_key = section + "." + full_key;
                }
                
                if (!value.empty()) {
                    values_[full_key] = value;
                }
            }
        }
        
        return true;
    }
    
    std::string getString(const std::string& key, const std::string& default_val = "") const {
        auto it = values_.find(key);
        return (it != values_.end()) ? it->second : default_val;
    }
    
    float getFloat(const std::string& key, float default_val = 0.0f) const {
        auto it = values_.find(key);
        if (it != values_.end()) {
            return std::stof(it->second);
        }
        return default_val;
    }
    
    double getDouble(const std::string& key, double default_val = 0.0) const {
        auto it = values_.find(key);
        if (it != values_.end()) {
            return std::stod(it->second);
        }
        return default_val;
    }
    
    int getInt(const std::string& key, int default_val = 0) const {
        auto it = values_.find(key);
        if (it != values_.end()) {
            return std::stoi(it->second);
        }
        return default_val;
    }
    
    Vec3 getVec3(const std::string& key, const Vec3& default_val = Vec3(0, 0, 0)) const {
        auto it = values_.find(key);
        if (it != values_.end()) {
            return parseVec3Array(it->second);
        }
        return default_val;
    }
    
    bool getBool(const std::string& key, bool default_val = false) const {
        auto it = values_.find(key);
        if (it != values_.end()) {
            std::string val = it->second;
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            return (val == "true" || val == "yes" || val == "1");
        }
        return default_val;
    }
    
    bool hasKey(const std::string& key) const {
        return values_.find(key) != values_.end();
    }
    
private:
    static Vec3 parseVec3Array(const std::string& str) {
        // Parse [x, y, z] format
        std::string cleaned = trimWhitespace(str);
        if (cleaned.empty() || cleaned[0] != '[') {
            return Vec3(0, 0, 0);
        }
        
        // Remove brackets
        cleaned = cleaned.substr(1, cleaned.length() - 2);
        
        // Parse comma-separated values
        std::vector<double> values;
        std::stringstream ss(cleaned);
        std::string token;
        
        while (std::getline(ss, token, ',')) {
            values.push_back(std::stod(trimWhitespace(token)));
        }
        
        if (values.size() >= 3) {
            return Vec3(values[0], values[1], values[2]);
        }
        return Vec3(0, 0, 0);
    }
};

/**
 * @brief Scene loader - converts YAML files to SceneDescription
 */
class SceneLoader {
public:
    /**
     * @brief Load a scene from YAML file
     * @param filename Path to YAML scene file
     * @param scene Output SceneDescription to populate
     * @return true if successful
     */
    static bool loadFromYAML(const std::string& filename, SceneDescription& scene) {
        SimpleSceneParser parser;
        
        std::cout << "Loading scene from: " << filename << std::endl;
        
        if (!parser.parseFile(filename)) {
            std::cerr << "ERROR: Failed to parse YAML file" << std::endl;
            return false;
        }
        
        // Clear existing scene
        scene.materials.clear();
        scene.geometries.clear();
        scene.meshes.clear();
        
        // Load materials first
        std::map<std::string, int> material_name_to_id;
        if (!loadMaterials(parser, scene, material_name_to_id)) {
            std::cerr << "ERROR: Failed to load materials" << std::endl;
            return false;
        }
        
        // Then load geometry (which references materials)
        if (!loadGeometry(parser, scene, material_name_to_id)) {
            std::cerr << "ERROR: Failed to load geometry" << std::endl;
            return false;
        }
        
        std::cout << "Scene loaded successfully:" << std::endl;
        std::cout << "  - " << scene.materials.size() << " materials" << std::endl;
        std::cout << "  - " << scene.geometries.size() << " objects" << std::endl;
        
        return scene.validate();
    }
    
private:
    static MaterialType parseMaterialType(const std::string& type_str) {
        if (type_str == "lambertian") return MaterialType::LAMBERTIAN;
        if (type_str == "metal") return MaterialType::METAL;
        if (type_str == "mirror") return MaterialType::MIRROR;
        if (type_str == "rough_mirror") return MaterialType::ROUGH_MIRROR;
        if (type_str == "glass") return MaterialType::GLASS;
        if (type_str == "dielectric") return MaterialType::DIELECTRIC;
        if (type_str == "light") return MaterialType::LIGHT;
        if (type_str == "constant") return MaterialType::CONSTANT;
        if (type_str == "show_normals") return MaterialType::SHOW_NORMALS;
        return MaterialType::LAMBERTIAN;
    }
    
    static ProceduralPattern parsePatternType(const std::string& type_str) {
        if (type_str == "fibonacci_dots") return ProceduralPattern::FIBONACCI_DOTS;
        if (type_str == "checkerboard") return ProceduralPattern::CHECKERBOARD;
        if (type_str == "stripes") return ProceduralPattern::STRIPES;
        return ProceduralPattern::NONE;
    }
    
    static bool loadMaterials(const SimpleSceneParser& parser, SceneDescription& scene,
                             std::map<std::string, int>& material_name_to_id) {
        // Try to load materials by index
        for (int i = 0; i < 100; ++i) {  // Reasonable limit
            std::string prefix = "material" + std::to_string(i);
            
            if (!parser.hasKey(prefix + ".name")) {
                break;  // No more materials
            }
            
            std::string mat_name = parser.getString(prefix + ".name");
            std::string mat_type = parser.getString(prefix + ".type", "lambertian");
            
            MaterialDesc mat;
            mat.type = parseMaterialType(mat_type);
            mat.albedo = parser.getVec3(prefix + ".albedo", Vec3(0.7, 0.7, 0.7));
            mat.emission = parser.getVec3(prefix + ".emission", Vec3(0, 0, 0));
            mat.roughness = parser.getFloat(prefix + ".roughness", 0.0f);
            mat.metallic = parser.getFloat(prefix + ".metallic", 0.0f);
            mat.refractive_index = parser.getFloat(prefix + ".refractive_index", 1.0f);
            mat.transmission = parser.getFloat(prefix + ".transmission", 0.0f);
            
            // Check for procedural pattern
            if (parser.hasKey(prefix + ".pattern.type")) {
                mat.pattern = parsePatternType(parser.getString(prefix + ".pattern.type"));
                mat.pattern_color = parser.getVec3(prefix + ".pattern.color", Vec3(0, 0, 0));
                mat.pattern_param1 = parser.getFloat(prefix + ".pattern.dot_count", 0.0f);
                mat.pattern_param2 = parser.getFloat(prefix + ".pattern.dot_radius", 0.0f);
            }
            
            int mat_id = scene.addMaterial(mat);
            material_name_to_id[mat_name] = mat_id;
            
            std::cout << "  Loaded material: " << mat_name << " (ID: " << mat_id << ")" << std::endl;
        }
        
        return !scene.materials.empty();
    }
    
    static bool loadGeometry(const SimpleSceneParser& parser, SceneDescription& scene,
                            const std::map<std::string, int>& material_name_to_id) {
        // Try to load geometry by index
        for (int i = 0; i < 1000; ++i) {  // Reasonable limit
            std::string prefix = "geom" + std::to_string(i);
            
            if (!parser.hasKey(prefix + ".type")) {
                break;  // No more geometry
            }
            
            std::string geom_type = parser.getString(prefix + ".type");
            std::string mat_name = parser.getString(prefix + ".material");
            
            // Look up material ID
            auto it = material_name_to_id.find(mat_name);
            if (it == material_name_to_id.end()) {
                std::cerr << "ERROR: Unknown material: " << mat_name << std::endl;
                continue;
            }
            int mat_id = it->second;
            
            if (geom_type == "sphere") {
                Vec3 center = parser.getVec3(prefix + ".center");
                double radius = parser.getDouble(prefix + ".radius", 1.0);
                scene.addSphere(center, radius, mat_id);
            }
            else if (geom_type == "displaced_sphere") {
                Vec3 center = parser.getVec3(prefix + ".center");
                double radius = parser.getDouble(prefix + ".radius", 1.0);
                float disp_scale = parser.getFloat(prefix + ".displacement_scale", 0.2f);
                int pattern = parser.getInt(prefix + ".pattern_type", 0);
                scene.addDisplacedSphere(center, radius, mat_id, disp_scale, pattern);
            }
            else if (geom_type == "rectangle") {
                Vec3 corner = parser.getVec3(prefix + ".corner");
                Vec3 u = parser.getVec3(prefix + ".u");
                Vec3 v = parser.getVec3(prefix + ".v");
                scene.addRectangle(corner, u, v, mat_id);
            }
            
            std::cout << "  Loaded " << geom_type << " with material " << mat_name << std::endl;
        }
        
        return !scene.geometries.empty();
    }
};

} // namespace Scene
