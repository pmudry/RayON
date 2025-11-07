/**
 * @file yaml_scene_loader.h
 * @brief YAML scene loader for SceneDescription
 * 
 * Loads scenes from YAML files. Uses minimal dependencies.
 * Note: Works around "using namespace std" pollution from other headers.
 */

#pragma once

#include "scene_description.h"

// Forward declarations to avoid std:: namespace pollution issues
#include <iosfwd>

namespace Scene {

/**
 * @brief Load scene from YAML file
 * @param filename Path to YAML scene file
 * @param scene SceneDescription to populate
 * @return true if successful
 */
bool loadSceneFromYAML(const char* filename, SceneDescription& scene);

/**
 * @brief Save scene to YAML file  
 * @param filename Output YAML filename
 * @param scene SceneDescription to save
 * @return true if successful
 */
bool saveSceneToYAML(const char* filename, const SceneDescription& scene);

} // namespace Scene
