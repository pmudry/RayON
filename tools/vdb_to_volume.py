#!/usr/bin/env python3
"""
Resample an OpenVDB float grid into RayON's dense volume format (`.vol`), the "raw grid" route
of explanations/VOLUMETRIC_CLOUDS.md: the path tracer reads a dense density grid straight into
its 3D texture, no OpenVDB dependency on the C++ side.

    python3 tools/vdb_to_volume.py bunny_cloud.vdb resources/volumes/bunny_cloud.vol --cells 300

Needs the OpenVDB Python bindings (Ubuntu: `sudo apt install python3-openvdb`) and numpy.
Blender's smoke caches (`fluid_data_XXXX.vdb`, grid `density`) work the same way.

File layout (little endian):
    char[8]  "RAYONVOL"
    int32    version (1)
    int32    nx, ny, nz          grid resolution
    float32  bmin[3], bmax[3]    world-space box covered by the grid (voxel edges), VDB units
    float32  max_value           largest value of the source grid (the data is divided by it)
    float32  nx*ny*nz values in [0, 1], x fastest, then y, then z
"""

import argparse
import math
import struct
import sys

import numpy as np
import pyopenvdb as vdb


def main() -> int:
   ap = argparse.ArgumentParser(description="OpenVDB float grid -> RayON dense .vol grid")
   ap.add_argument("input", help="source .vdb file")
   ap.add_argument("output", help="destination .vol file")
   ap.add_argument("--grid", default="density", help="name of the float grid to convert (default: density)")
   ap.add_argument("--cells", type=int, default=256,
                   help="largest side of the output grid; the source is box-filtered by an integer factor "
                        "so that it fits (default: 256)")
   args = ap.parse_args()

   grid = vdb.read(args.input, args.grid)
   (i0, j0, k0), (i1, j1, k1) = grid.evalActiveVoxelBoundingBox()
   dims = (i1 - i0 + 1, j1 - j0 + 1, k1 - k0 + 1)
   if min(dims) <= 0:
      print("empty grid", file=sys.stderr)
      return 1
   print(f"{args.input}: grid '{args.grid}' {dims[0]}x{dims[1]}x{dims[2]} voxels, "
         f"voxel size {grid.transform.voxelSize()[0]:.4g}")

   dense = np.zeros(dims, dtype=np.float32)  # indexed [i, j, k] = [x, y, z]
   grid.copyToArray(dense, ijk=(i0, j0, k0))

   # Integer box filter so that the longest side fits in --cells
   f = max(1, math.ceil(max(dims) / args.cells))
   if f > 1:
      padded = [math.ceil(d / f) * f for d in dims]
      tmp = np.zeros(padded, dtype=np.float32)
      tmp[: dims[0], : dims[1], : dims[2]] = dense
      dense = tmp.reshape(padded[0] // f, f, padded[1] // f, f, padded[2] // f, f).mean(axis=(1, 3, 5))
   out_dims = dense.shape

   max_value = float(dense.max())
   if max_value <= 0.0:
      print("grid holds no density", file=sys.stderr)
      return 1
   dense /= max_value

   # World box: voxel (i, j, k) is centred on indexToWorld(i, j, k) and spans +-half a voxel;
   # the padded voxels at the high end extend the box by whole (filtered) cells.
   vs = grid.transform.voxelSize()
   lo = grid.transform.indexToWorld((i0, j0, k0))
   bmin = [lo[a] - 0.5 * vs[a] for a in range(3)]
   bmax = [bmin[a] + out_dims[a] * f * vs[a] for a in range(3)]

   with open(args.output, "wb") as fh:
      fh.write(struct.pack("<8siiiiffffff f".replace(" ", ""), b"RAYONVOL", 1, *out_dims, *bmin, *bmax, max_value))
      # RayON's 3D texture wants x fastest: memory order [z][y][x]
      np.ascontiguousarray(dense.transpose(2, 1, 0)).tofile(fh)

   print(f"{args.output}: {out_dims[0]}x{out_dims[1]}x{out_dims[2]} cells (filter {f}), "
         f"box [{bmin[0]:.3f}, {bmin[1]:.3f}, {bmin[2]:.3f}] - [{bmax[0]:.3f}, {bmax[1]:.3f}, {bmax[2]:.3f}], "
         f"max value {max_value:.4g}, {out_dims[0] * out_dims[1] * out_dims[2] * 4 / 1e6:.1f} MB")
   return 0


if __name__ == "__main__":
   sys.exit(main())
