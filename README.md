# A simple yet pretty cool ray-tracer in CPP for CS302 HPC

Based on https://github.com/RayTracing/raytracing.github.io/tree/release/src/InOneWeekend, an amazing resource ! 

This is a complete hand-made re-implementation of the `InOneWeekend` version, to get started and understand how it works.

This version is single-threaded, with no GPU acceleration.

It uses [single-file public domain (or MIT licensed) libraries for C/C++](https://github.com/nothings/stb/tree/master).

## How to compile

## Using CMAKE

```bash
cmake 
make -j 
./v0_single_threaded
```

Or, all at once : 

```bash
cmake . && make -j && ./v0_single_threaded
```

## Whitin VSCode

Install the `CMake` extension, click here and there and then launch directly with `CTRL + SHIFT + F5`, just like in the old days ;)

# Notes for PA

- Ce serait probablement une idée qu'ils se fassent la main en regardant comment fonctionne le code pour faire le rendu d'une sphère dans un premier temps, sans vibe code.
- Ensuite rendu des normales
- Ajouter l'anti-aliasing, vraiment assez simple
- Ensuite vibe code du cube qui tourne
- Ajouter les ombres, c'est rigolo à faire
- Ensuite vibe code du // cpu directement pour chaque image pour un cube qui tourne
- Rajouter miroir, verre, textures
- Vibe code du checkerboard pour le sol
- Passer sur GPU

- C'est très très simple de // la génération d'images, voir dans l'historique (avec future sur les images). A mon avis c'est un truc à montrer aux étudiants comment utiliser 72 cpus d'un coup, c'est assez incroyable en fait.
- Aussi trivial de // le rendu de la caméra

- Est-ce que l'on demanderait pas de vibecoder aussi AABB optimisation