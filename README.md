# A realt-time ray-tracer in CPP/CUDA for CS302 HPC

Based on <https://github.com/RayTracing/raytracing.github.io/tree/release/src/InOneWeekend>, an amazing resource ! 

This is a complete hand-made re-implementation of the `InOneWeekend` version, to get started and understand how it works. There are multiple implementations : 

- single-threaded
- multi-threaded
- CUDA accelerated

It uses [single-file public domain (or MIT licensed) libraries for C/C++](https://github.com/nothings/stb/tree/master).

The project is declined in another version that serves as a starting point for the studnets.

# How to compile and environment setup in VSCode

## Using `cmake` to generate compilation files and build using `make`

Create manually via

```bash
mkdir build
cd build
cmake .. --fresh
make -j
```

This will generate the appropriate `compile_commands.json` for `clangd` so that you get syntax highlighting, code completion etc. in VS Code. You can then run from the `build directory`

```bash
./302_raytracer --help
```

Or, all at once : 

## Running
```bash
make -j && ./302_raytracer -m 2
```

Rendered frames are written to `rendered_images/` with timestamped filenames such as `output_2025-11-15_14-22-09.png`. Each run produces a new PNG (timestamp uses local time, second precision), so you can sort files chronologically without overwriting previous renders.

## Within VSCode

Install extension `clangd` from `LLVM`. **WARNING** to make `clangd` work you must have `compile_commands.json` (which are now generated automatically from `CMakeLists.txt` called during the `cmake` phase above). :warning: In addition, the headers that are in C++ **MUST** be named with a `.hpp` extension (not `.h`). I spent I whole day trying to figure this out.

This is all you need. There are *tasks* created in the `.vscode` folder that can be launched with `CTRL+Shift+P` -> Tasks and then you can build, and run. You can even setup key bindings for that.

## Build documentation

If required, the documentation can be built with `doxygen`, which should be run in the main directory. The results are not saved in the git repository to save space.
