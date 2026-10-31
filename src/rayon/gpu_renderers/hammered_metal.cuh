#pragma once
/**
 * @file hammered_metal.cuh
 * @brief Dimples of the HAMMERED material, shared by CUDA and OptiX.
 *
 * Hammered metal: a rough metal (the ROUGH_MIRROR lobe) whose shading normal is tilted by
 * procedural hammer marks, like the dimples of the golf ball but on any surface and without
 * touching the geometry. Space is cut into jittered cells (Worley / Voronoi, cell size `size`); the
 * mark around a hit is the cell of the nearest feature point, a shallow bowl centred on it: the
 * normal tilts towards the centre in proportion to the distance, so neighbouring marks meet in
 * sharp ridges. The tilt uses the tangential part of the offset only.
 */

#ifdef __CUDACC__
#define HAMMER_HD __host__ __device__
#else
#define HAMMER_HD
#endif

/// Jitter of a cell's feature point in [0, 1)^3 (integer hash of the cell coordinates)
HAMMER_HD inline void hammer_cell_jitter(int cx, int cy, int cz, float &jx, float &jy, float &jz)
{
   unsigned int h = (unsigned int)cx * 73856093u ^ (unsigned int)cy * 19349663u ^ (unsigned int)cz * 83492791u;
   auto next = [&h]()
   {
      h ^= h >> 16;
      h *= 0x7feb352du;
      h ^= h >> 15;
      h *= 0x846ca68bu;
      h ^= h >> 16;
      return (float)(h >> 8) * (1.0f / 16777216.0f);
   };
   jx = next();
   jy = next();
   jz = next();
}

/// Offset (in units of the cell size) from the nearest feature point to the point p / size
HAMMER_HD inline void hammer_nearest_offset(float px, float py, float pz, float size, float &vx, float &vy,
                                            float &vz)
{
   const float qx = px / size, qy = py / size, qz = pz / size;
   const int   cx = (int)floorf(qx), cy = (int)floorf(qy), cz = (int)floorf(qz);
   float       best = 1e30f;
   vx = vy = vz = 0.0f;
   for (int dz = -1; dz <= 1; ++dz)
      for (int dy = -1; dy <= 1; ++dy)
         for (int dx = -1; dx <= 1; ++dx)
         {
            float jx, jy, jz;
            hammer_cell_jitter(cx + dx, cy + dy, cz + dz, jx, jy, jz);
            const float ox = qx - ((float)(cx + dx) + jx);
            const float oy = qy - ((float)(cy + dy) + jy);
            const float oz = qz - ((float)(cz + dz) + jz);
            const float d2 = ox * ox + oy * oy + oz * oz;
            if (d2 < best)
            {
               best = d2;
               vx   = ox;
               vy   = oy;
               vz   = oz;
            }
         }
}

/// Shading normal of a hammered surface at p, from the geometric normal (nx, ny, nz), in place
HAMMER_HD inline void hammer_normal(float px, float py, float pz, float size, float depth, float &nx, float &ny,
                                    float &nz)
{
   if (size <= 0.0f || depth <= 0.0f)
      return;
   float vx, vy, vz;
   hammer_nearest_offset(px, py, pz, size, vx, vy, vz);
   // Tangential part of the offset: the bowl's slope at the hit
   const float vn = vx * nx + vy * ny + vz * nz;
   vx -= vn * nx;
   vy -= vn * ny;
   vz -= vn * nz;
   // Concave bowl: the normal leans back towards the centre of the mark
   float mx = nx - depth * vx, my = ny - depth * vy, mz = nz - depth * vz;
   const float len = sqrtf(mx * mx + my * my + mz * mz);
   if (len > 1e-6f)
   {
      nx = mx / len;
      ny = my / len;
      nz = mz / len;
   }
}
