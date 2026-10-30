# Volume grids

Dense density grids (`.vol`) read by `volume` materials through the `grid` key (see
`resources/scenes/SCENE_FORMAT.md`). They are large (tens of MB) and git-ignored: regenerate
them with `tools/vdb_to_volume.py` from an OpenVDB file.

```bash
sudo apt install python3-openvdb          # OpenVDB Python bindings (Ubuntu / Debian)
curl -L -o /tmp/bunny_cloud.vdb \
  https://media.githubusercontent.com/media/AcademySoftwareFoundation/openvdb-website/master/download/models/bunny_cloud.vdb
python3 tools/vdb_to_volume.py /tmp/bunny_cloud.vdb resources/volumes/bunny_cloud.vol --cells 300
./build/rayon -m 5 --scene resources/scenes/19_vdb_bunny.yaml
```

`bunny_cloud.vdb` (OpenVDB sample models, <https://www.openvdb.org/download/>) is the grid
behind pbrt-v4's `bunny-cloud` scene: 577×572×438 voxels of 0.08 units, grid `density`.
`--cells 300` box-filters it by 2 to 289×286×219 cells (72 MB as float32).

## File format

Little endian: `"RAYONVOL"`, `int32` version (1), `int32` nx ny nz, `float32` bmin[3] bmax[3]
(world box of the grid in the source's units), `float32` max_value (the data was divided by
it), then `nx*ny*nz` float32 values in [0, 1], x fastest. The scene's `density` scales the
values back to an extinction coefficient; `grid_scale` and `grid_position` place the box.

## Volumes from Blender

Any Blender smoke simulation exports to OpenVDB and goes through the same tool:

1. Select a mesh, *Object > Quick Effects > Quick Smoke*. Blender creates the domain and makes
   the mesh an emitter.
2. Domain: *Resolution Divisions* 128–256, *Noise* on (*Upres* 2) for fine detail. To keep the
   shape of the mesh, set *Gas > Buoyancy Density* and *Heat* to 0, *Vorticity* 0, and the flow
   *Behavior* to *Geometry* (emitted once, barely moving).
3. *Cache*: *Type* All, *Volumes Format* OpenVDB, then *Bake*. The cache directory holds
   `data/fluid_data_XXXX.vdb` with the grids `density`, `temperature`, `flame`; pick a frame
   before the smoke dissipates (10–20).
4. `python3 tools/vdb_to_volume.py fluid_data_0015.vdb resources/volumes/smoke.vol --grid density`

Without a simulation, a *Volume > Empty* object with the *Mesh to Volume* and *Volume Displace*
modifiers gives a fog volume of any mesh (what `bunny_cloud.vdb` is), but Blender has no
OpenVDB export for volume objects: write it with `pyopenvdb` from a Python script, or build it
without Blender with OpenVDB's `vdb_tool`:

```bash
vdb_tool -read bunny.obj -mesh2ls voxel=0.005 -ls2fog -write bunny_fog.vdb
```
