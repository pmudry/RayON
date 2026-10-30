/**
 * @file scene_builder.hpp
 * @brief Converts SceneDescription to renderer-specific formats
 *
 * This file provides builders that convert the unified SceneDescription format to
 * GPU format: flat arrays in device memory.
 *
 * Note: The CPU scene builder (CPUSceneBuilder) has been moved to the
 * legacy/cpu-renderer branch.
 */

#pragma once

#include "scene_description.hpp"

// Forward declaration for CUDA scene
namespace CudaScene
{
struct Scene;
}

namespace Scene
{

//==============================================================================
// CUDA SCENE BUILDER - Declared here, implemented in scene_builder_cuda.cu
//==============================================================================

class CudaSceneBuilder
{
 public:
   static CudaScene::Scene *buildGPUScene(const SceneDescription &desc);
   static void freeGPUScene(CudaScene::Scene *scene);
   /// Rewrite one material of an uploaded scene in place (GUI edits); no rebuild
   static void updateMaterial(CudaScene::Scene *scene, int idx, const MaterialDesc &desc);
};

} // namespace Scene
