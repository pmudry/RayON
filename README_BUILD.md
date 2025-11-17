# Build Instructions

## Building with Different Configurations

### Method 1: Using CMake directly

**For Release build (optimized, fast):**
```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j 12
```

**For Debug build (with debug symbols, slower):**
```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j 12
```

### Method 3: Switching between builds

If you want to switch from Release to Debug (or vice versa), you should clean first:
```bash
cd build
make clean
cmake .. -DCMAKE_BUILD_TYPE=Debug  # or Release
make -j8
```

## Default Build Type

If you don't specify a build type, **Release** is used by default for maximum performance.

## Build Type Differences

- **Release**: `-O3` optimization, no debug symbols, fastest execution
- **Debug**: `-g` debug symbols, `-G` CUDA device debug, easier debugging, slower execution

## Automated Include Directory Management

The CMakeLists.txt automatically discovers all subdirectories under `src/` and adds them to the include paths for compilation. This eliminates the need to manually list include directories when adding new subfolders.

Additionally, CMake updates the `.clangd` file's `Add:` section with the corresponding `-I` flags for clangd language server support. The rest of `.clangd` remains unchanged.

When adding or removing subdirectories under `src/`, simply re-run `cmake .. --fresh` to update the includes automatically.

## Static Analysis with clang-tidy

If `clang-tidy` is installed on the system, CMake will automatically enable it during the build process to perform static analysis on C++ code. This helps detect potential bugs, style issues, and performance problems.

The `.clang-tidy` file configures the checks to run. To disable clang-tidy for a specific build, set the environment variable `CMAKE_CXX_CLANG_TIDY` to an empty string or remove the clang-tidy package.
