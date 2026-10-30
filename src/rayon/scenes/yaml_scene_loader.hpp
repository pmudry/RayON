/**
 * @file yaml_scene_loader.hpp
 * @brief YAML scene loader for SceneDescription
 * 
 * Loads scenes from YAML files. Uses minimal dependencies.
 * Note: Works around "using namespace std" pollution from other headers.
 */

#pragma once

#include "scene_description.hpp"

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

/**
 * @brief Write a camera (position, look_at, fov) into an existing YAML scene file, in place.
 *
 * Only the `camera:` block is touched: its position / look_at / fov lines are rewritten (or
 * added), every other line of the file, comments included, is kept as is. A file without a
 * camera block gets one appended.
 * @return true if the file was rewritten
 */
bool saveCameraToYAML(const char* filename, const Vec3& position, const Vec3& look_at, float fov);

} // namespace Scene
