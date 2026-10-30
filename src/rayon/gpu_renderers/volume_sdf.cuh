#pragma once
/**
 * @file volume_sdf.cuh
 * @brief Signed distance grid of a closed triangle mesh, used to shape VOLUME materials
 * (edge fall-off and noise lumps need to know how deep a point is inside the mesh).
 */

#include "volume_params.h"
#include <string>
#include <vector>

/**
 * @brief Build the signed distance grid of a mesh and bind it to `vp` as a 3D texture.
 * @param tris       Triangle vertices in world space, 3 per triangle
 * @param max_cells  Cells along the longest side of the (padded) bounding box (resolution)
 * @param pad        Margin around the mesh covered by the grid (room for the lumps)
 * @return false when there is nothing to build (vp.sdf stays 0)
 *
 * Brute force on the GPU: every voxel tests every triangle (exact distance, sign by ray
 * parity), which is fine for a few tens of thousands of triangles at 64 cells.
 */
bool buildVolumeSDF(const std::vector<float3> &tris, int max_cells, float pad, VolumeParams &vp);

/**
 * @brief Bake the extinction multiplier field (shape x interior noise) of `vp` over its distance
 * grid box into vp.field, at `max_cells` along the longest side. Call again after any parameter
 * change (a few milliseconds). Needs vp.sdf; a no-op otherwise.
 */
void bakeVolumeField(VolumeParams &vp, int max_cells);

/**
 * @brief Bind a density grid read from a `.vol` file (tools/vdb_to_volume.py) as the extinction
 * field of `vp`, over the world box [lo, hi]. The values are already in [0, 1]; outside the box
 * the field reads 0. No distance grid is built (vp.sdf stays 0: no empty-space skipping), and
 * bakeVolumeField() leaves the grid alone, so the GUI's noise sliders have no effect on it.
 * @return false when the file cannot be read or the texture allocated (vp.field stays 0)
 */
bool loadVolumeGrid(const std::string &path, float3 lo, float3 hi, VolumeParams &vp);

/// Release the textures and arrays bound by buildVolumeSDF / bakeVolumeField
void freeVolumeSDF(VolumeParams &vp);
