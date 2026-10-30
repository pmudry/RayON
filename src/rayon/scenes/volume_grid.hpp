#pragma once
/**
 * @file volume_grid.hpp
 * @brief Reader of RayON's dense volume grid files (`.vol`), written by tools/vdb_to_volume.py
 * from OpenVDB data (OpenVDB sample models, Blender smoke caches). Host side only.
 *
 * Layout (little endian): "RAYONVOL", int32 version = 1, int32 nx ny nz, float bmin[3] bmax[3]
 * (world box of the grid in the source's units), float max_value (the source values were
 * divided by it), then nx*ny*nz float values in [0, 1], x fastest.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace Scene
{

struct VolumeGridHeader
{
   int32_t nx = 0, ny = 0, nz = 0;
   float   bmin[3] = {0.0f, 0.0f, 0.0f};
   float   bmax[3] = {0.0f, 0.0f, 0.0f};
   float   max_value = 1.0f;
};

/// Read the header of an open `.vol` file (leaves the stream at the first value)
inline bool readVolumeGridHeader(FILE *f, VolumeGridHeader &h)
{
   char    magic[8];
   int32_t version = 0;
   if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "RAYONVOL", 8) != 0)
      return false;
   if (fread(&version, sizeof(int32_t), 1, f) != 1 || version != 1)
      return false;
   int32_t dims[3];
   if (fread(dims, sizeof(int32_t), 3, f) != 3)
      return false;
   h.nx = dims[0];
   h.ny = dims[1];
   h.nz = dims[2];
   if (fread(h.bmin, sizeof(float), 3, f) != 3 || fread(h.bmax, sizeof(float), 3, f) != 3 ||
       fread(&h.max_value, sizeof(float), 1, f) != 1)
      return false;
   return h.nx > 0 && h.ny > 0 && h.nz > 0;
}

/// Read only the header of a `.vol` file (box and resolution)
inline bool readVolumeGridHeader(const std::string &path, VolumeGridHeader &h)
{
   FILE *f = fopen(path.c_str(), "rb");
   if (!f)
      return false;
   const bool ok = readVolumeGridHeader(f, h);
   fclose(f);
   return ok;
}

/// Read a whole `.vol` file: header plus the nx*ny*nz values (x fastest)
inline bool readVolumeGrid(const std::string &path, VolumeGridHeader &h, std::vector<float> &values)
{
   FILE *f = fopen(path.c_str(), "rb");
   if (!f)
      return false;
   bool ok = readVolumeGridHeader(f, h);
   if (ok)
   {
      const size_t n = static_cast<size_t>(h.nx) * h.ny * h.nz;
      values.resize(n);
      ok = fread(values.data(), sizeof(float), n, f) == n;
   }
   fclose(f);
   return ok;
}

} // namespace Scene
