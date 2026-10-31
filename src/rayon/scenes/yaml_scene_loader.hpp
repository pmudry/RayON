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
 * @brief Write the current view into an existing YAML scene file, in place: the camera
 * (position, look_at, fov) and the light settings (background_intensity, light_intensity,
 * light_falloff, and indirect_clamp when it is given: a negative value leaves that key alone; likewise
 * tone_mapping and exposure, written only when they differ from the defaults or the file has them).
 *
 * Only those lines of the `camera:` and `settings:` blocks are rewritten (or added, the block
 * itself being appended when missing); every other line, comments included, is kept as is.
 * @return true if the file was rewritten
 */
bool saveViewToYAML(const char* filename, const Vec3& position, const Vec3& look_at, float fov,
                    float background_intensity, float light_intensity, float light_falloff = 2.0f,
                    float indirect_clamp = -1.0f, int tone_mapping = -1, float exposure = 1.0f);

} // namespace Scene
