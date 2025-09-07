# A simple yet pretty cool ray-tracer in CPP for CS302 HPC

Based on https://github.com/RayTracing/raytracing.github.io/tree/release/src/InOneWeekend, an amazing resource ! 

This is a complete hand-made re-implementation of the `InOneWeekend` version, to get started and understand how it works.

This version is single-threaded, with no GPU acceleration.

It uses [single-file public domain (or MIT licensed) libraries for C/C++](https://github.com/nothings/stb/tree/master).

## How to compile

## Using CMAKE

```bash
cmake .
make -j 
./v0_single_threaded
```

Or, all at once : 

```bash
cmake . && make -j 24 && ./v0_single_threaded
```

## Whitin VSCode

Install the `CMake` extension, click here and there and then launch directly with `CTRL + SHIFT + F5`, just like in the old days ;)

