# A simple yet pretty cool ray-tracer in CPP for CS302 HPC

Based on <https://github.com/RayTracing/raytracing.github.io/tree/release/src/InOneWeekend>, an amazing resource ! 

This is a complete hand-made re-implementation of the `InOneWeekend` version, to get started and understand how it works. There are multiple implementations : 

- single-threaded
- multi-threaded
- CUDA accelerated

It uses [single-file public domain (or MIT licensed) libraries for C/C++](https://github.com/nothings/stb/tree/master).

The project is declined in another version that serves as a starting point for the studnets.

## How to compile and environment setup in VSCode

## Using CMAKE to generate compilation scripts

Create manually via

```bash
mkdir build
cd build
cmake .. --fresh -G 'Unix Makefiles'
make -j
```

This will generate the appropriate `compile_commands.json` for `clangd` so that you get syntax highlighting, code completion etc. in VS Code.

You can then run from the `build directory`

```bash
./302_raytracer
```

Or, all at once : 

## Running
```bash
make -j && echo "2" | ./302_raytracer
```

Rendered frames are written to `rendered_images/` with timestamped filenames such as `output_2025-11-15_14-22-09.png`. Each run produces a new PNG (timestamp uses local time, second precision), so you can sort files chronologically without overwriting previous renders.

## Within VSCode

Install extension `clangd` from `LLVM`. **WARNING** to make `clangd` work you must have `compile_commands.json` and the headers that are in C++ **MUST** be named with a `.hpp` extension (not `.h`). I spent I whole day trying to figure this out.

This is all you need. There are *tasks* created in the `.vscode` folder that can be launched with `CTRL+Shift+P` -> Tasks and then you can build, and run. You can even setup key bindings for that.

## Build documentation

If required, the documentation can be built with `doxygen`, which should be run in the main directory. The results are not saved in the git repository to save space.

# Notes for PA
- Trucs cools : ImGUI à installer ?
- Ce serait probablement une idée qu'ils se fassent la main en regardant comment fonctionne le code pour faire le rendu d'une sphère dans un premier temps, sans vibe code.
- Ensuite rendu des normales
- Ajouter l'anti-aliasing, vraiment assez simple
- Ajouter les ombres, c'est rigolo à faire
- Ensuite vibe code du // cpu directement pour chaque image pour un cube qui tourne
- Rajouter miroir, verre, textures
- Vibe code du checkerboard pour le sol
- Passer sur GPU

- C'est très très simple de // la génération d'images, voir dans l'historique (avec future sur les images). A mon avis c'est un truc à montrer aux étudiants comment utiliser 72 cpus d'un coup, c'est assez incroyable en fait.
- Aussi trivial de // le rendu de la caméra
- Est-ce que l'on demanderait pas de vibecoder aussi AABB optimisation
- Claude n'arrive pas à implémenter les balles de golf correctement