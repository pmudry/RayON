# Progressive SDL Rendering Feature

## Overview

This branch adds an interactive SDL2-based progressive rendering mode to the raytracer with **real-time camera controls**. This feature allows you to see your render in real-time as it progressively improves in quality through multiple sample stages, while also being able to interactively adjust the camera position, rotation, and zoom.

## What is Progressive Rendering?

Progressive rendering displays the image while it's being rendered, starting with low quality (few samples per pixel) and gradually increasing quality. This allows you to:

- **Preview quickly**: See your scene in seconds with low sample counts
- **Stop early**: If the preview looks wrong, you can stop and adjust parameters
- **Watch progress**: See the image quality improve in real-time
- **Interactive experience**: Get visual feedback during the rendering process
- **Adjust camera**: Rotate, pan, and zoom the camera to explore different angles

## Features

### Interactive Camera Controls 🎮

The SDL window provides full interactive camera control:

- **Left Mouse Button (LMB)**: Rotate camera (orbit around the look-at point)
  - Drag horizontally to rotate around the scene
  - Drag vertically to change elevation angle
  - Camera maintains its distance from the look-at point
  
- **Right Mouse Button (RMB)**: Pan camera (move the look-at point)
  - Move the entire camera and look-at point together
  - Useful for centering on different parts of the scene
  
- **Mouse Wheel**: Zoom in/out
  - Scroll up to zoom in (move camera closer)
  - Scroll down to zoom out (move camera further away)
  - Maintains current viewing angles

- **Space Bar**: Re-render with current camera position
  - Triggers a new progressive render from stage 1
  - Useful after making multiple camera adjustments
  - Automatically interrupts any ongoing render

- **ESC Key / Close Window**: Exit the interactive session

### Sample Stages

The progressive rendering goes through multiple quality stages:
- **Stage 1**: 8 samples per pixel (instant preview)
- **Stage 2**: 16 samples per pixel (quick improvement)
- **Stage 3**: 32 samples per pixel (cleaner result)
- **Stage 4**: 64 samples per pixel (good quality)
- **Stage 5**: 128 samples per pixel (high quality)
- **Stage 6**: 256 samples per pixel (very high quality)

**Block-Based Tiled Rendering**: To keep the viewer responsive during high-sample renders, each frame is split into a fixed **8×8 grid of rectangular blocks (64 tiles total)**. This provides:

- **64 event checks per frame** at all quality levels
- **Immediate visual feedback** - see tiles appear as they complete
- **Consistent visual appearance** - same tile boundaries across all stages eliminates artifacts
- **Highly responsive** - can interrupt renders almost instantly at any quality level

**Stage Delay**: After completing each quality stage, there's a 500ms pause before starting the next stage. This allows you to clearly see each quality level and interrupt if satisfied with the current result.

### Interactive Controls

- **Left Mouse Button (LMB)**: Rotate camera (orbit around the look-at point)
- **Right Mouse Button (RMB)**: Pan camera (move the look-at point and camera)
- **Mouse Wheel**: Zoom in/out (change camera distance)
- **Any camera movement**: Automatically interrupts current render and restarts from stage 1
- **ESC Key / Close Window**: Exit the interactive session
- **Space Bar**: Re-render with current camera position
- **ESC key**: Stop rendering and exit
- **Close window**: Stop rendering and exit

### Camera Behavior

The camera system uses an orbit model:
- The camera orbits around a "look-at" point (center of interest)
- Rotation keeps the camera at a constant distance from the look-at point
- Panning moves both the camera and look-at point together
- Zooming changes the distance while maintaining viewing angles
- Real-time re-rendering shows the scene from new camera positions

## How to Use

### Prerequisites

Make sure SDL2 is installed on your system:

```bash
# Ubuntu/Debian
sudo apt-get install libsdl2-dev

# macOS
brew install sdl2

# Windows
# Download from https://www.libsdl.org/download-2.0.php
```

### Building

The feature is automatically compiled if SDL2 is detected:

```bash
cd build
cmake .. --fresh
make -j12
```

If SDL2 is found, you'll see:
```
-- SDL2 found and linked for real-time display
```

### Running

When you run the raytracer, you'll see a new rendering option:

```bash
./302_raytracer -r 1080 -s 1024
```

The menu will show:
```
Choose rendering method:
	0. CPU sequential
	1. CPU parallel
	2. CUDA GPU (default)
	3. CUDA GPU with progressive SDL display
Enter choice (0, 1, 2, or 3): 
```

Select option **3** for progressive SDL rendering.

### Example Usage

```bash
# Progressive render at 1080p
echo 3 | ./302_raytracer -r 1080

# Progressive render at 4K
echo 3 | ./302_raytracer -r 2160

# Progressive render at 720p with custom samples
echo 3 | ./302_raytracer -r 720 -s 512
```

## Technical Details

### Implementation

The progressive rendering feature is implemented in `camera.h`:
- **Method**: `renderPixelsSDLProgressive()`
- **Conditional compilation**: Only available when `SDL2_FOUND` is defined
- **Backend**: Uses CUDA for each rendering stage (fast GPU rendering)
- **Display**: Updates SDL texture after each stage completion

### Rendering Flow

1. Initialize SDL window, renderer, and texture
2. For each sample stage:
   - Render with CUDA at current sample count
   - Update SDL texture with the new image data
   - Display in the window
   - Check for user interruption (ESC or window close)
3. Wait for user to acknowledge completion
4. Clean up SDL resources
5. Return final image for saving

### Customization

You can customize the sample stages in `main.cc`:

```cpp
// Default stages
c.renderPixelsSDLProgressive(localImage, {1, 4, 16, 64, 256, 1024});

// Custom stages - more granular
c.renderPixelsSDLProgressive(localImage, {1, 2, 4, 8, 16, 32, 64, 128, 256});

// Quick preview stages
c.renderPixelsSDLProgressive(localImage, {1, 4, 16});
```

## Performance Considerations

- **GPU Required**: Uses CUDA for rendering each stage, so a CUDA-capable GPU is required
- **Memory**: Keeps one full-resolution image buffer per stage
- **Display overhead**: Minimal - SDL texture updates are fast
- **Total time**: Roughly equivalent to rendering with the highest sample count directly

## Benefits Over Standard Rendering

1. **Immediate feedback**: See your scene within seconds
2. **Error detection**: Catch scene setup mistakes early
3. **Quality control**: Stop when quality is sufficient for your needs
4. **Visual satisfaction**: Watch the rendering improve in real-time
5. **Flexibility**: Adjust quality vs. speed tradeoff interactively
6. **Interactive exploration**: Rotate, pan, and zoom to find the perfect camera angle
7. **Real-time adjustment**: Change camera position and immediately see results
8. **Iterative workflow**: Quickly test multiple camera positions without restarting

## Typical Workflow

1. **Start the program** with option 3 (SDL progressive rendering)
2. **Initial render** begins automatically showing progressive quality stages
3. **Explore the scene**:
   - Use left mouse to rotate and view from different angles
   - Use right mouse to pan and focus on different areas
   - Use mouse wheel to get closer or further from objects
4. **Re-render as needed**: Press Space to trigger new render with current camera
5. **Find the perfect shot**: Iterate until satisfied with composition
6. **Wait for final quality**: Let the highest sample stage complete
7. **Exit and save**: Close window to save the final render to disk

## Troubleshooting

### SDL2 Not Found

If option 3 doesn't appear:
- SDL2 is not installed or not found by CMake
- Reinstall SDL2 and run `cmake .. --fresh` in the build directory

### Window Doesn't Appear

- Check that you're not in a headless environment
- Verify SDL2 is properly linked: `ldd ./302_raytracer | grep SDL`

### Slow Performance

- Progressive rendering uses CUDA, ensure your GPU drivers are up to date
- Each stage requires a complete render at that sample count
- For slower GPUs, consider using fewer or lower sample stages

## Future Enhancements

Potential improvements for this feature:
- Adjustable sample stages via command-line arguments
- Real-time statistics overlay (FPS, rays/sec, current stage)
- Pause/resume functionality
- Save intermediate stages
- Tile-based progressive rendering for huge resolutions
- Multi-GPU support for faster stage completion

## Code References

- **Main implementation**: `src/302_raytracer/camera.h` - `renderPixelsSDLProgressive()`
- **Menu integration**: `src/302_raytracer/main.cc` - rendering method selection
- **CMake configuration**: `CMakeLists.txt` - SDL2 detection and linking
- **CUDA backend**: `src/302_raytracer/camera_cuda.cu` - `renderPixelsCUDA()`
