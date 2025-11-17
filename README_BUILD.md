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
