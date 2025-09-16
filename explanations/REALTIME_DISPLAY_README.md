# Real-time CUDA Ray Tracer Display

This document explains how to use the real-time display feature of the CUDA ray tracer.

## Prerequisites

To enable real-time display, you need to install SDL2:

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install libsdl2-dev
```

### macOS
```bash
brew install sdl2
```

### Windows
Download the development libraries from: https://www.libsdl.org/download-2.0.php

## How It Works

The real-time display feature renders the image in small tiles (64x64 pixels by default) and updates the display after each tile is completed. This allows you to see the rendering progress in real-time rather than waiting for the entire image to finish.

### Key Features:
- **Progressive Rendering**: Watch the image build up tile by tile
- **Interactive Window**: Close the window anytime with the X button
- **Same Quality**: Uses the same CUDA rendering engine as the offline version
- **Memory Efficient**: Only renders one tile at a time on the GPU

## Usage

1. **Build the project** (after installing SDL2):
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

2. **Run the program**:
   ```bash
   ./v0_single_threaded
   ```

3. **Choose rendering mode**:
   - Option 1: CPU Parallel (original)
   - Option 2: CUDA GPU (original)
   - Option 3: **CUDA GPU with Real-time Display** (new)

4. **Watch the rendering**:
   - A window will open showing the progressive rendering
   - Each tile appears as it's completed
   - The window remains open after rendering is complete
   - Close the window to exit

## Technical Details

### Tile-Based Rendering
- **Tile Size**: 64x64 pixels (configurable in `camera.h`)
- **Rendering Order**: Left-to-right, top-to-bottom
- **Memory Management**: Full image buffer maintained on both CPU and GPU
- **Synchronization**: Each tile is synchronized before display update

### Performance Considerations
- **GPU Memory**: Uses same amount as full rendering (not tile-optimized yet)
- **Display Updates**: Small delay (10ms) between tiles for visibility
- **Event Handling**: Window remains responsive during rendering

### Code Structure
- `renderPixelsCUDART()`: Main real-time rendering method in `camera.h`
- `renderPixelsCUDATile()`: Host function for tile rendering in `camera_cuda.cu`
- `renderPixelsTileKernel()`: CUDA kernel for tile-based rendering

## Troubleshooting

### SDL2 Not Found
If you see warnings about SDL2 not being found:
1. Install SDL2 using the commands above
2. Re-run `cmake ..` and `make`
3. The real-time option will be automatically enabled

### Window Doesn't Appear
- Check that your display is properly configured
- Try running from a graphical terminal
- On Linux, ensure X11 forwarding is working if using SSH

### Performance Issues
- Reduce tile size in `camera.h` for more frequent updates
- Increase samples per pixel for better quality
- The real-time version uses the same CUDA optimization as the offline version

## Future Improvements

Potential enhancements for the real-time display:
- Adaptive tile sizing based on GPU memory
- Parallel tile rendering for faster updates
- Preview mode with lower quality tiles
- Progress bar and timing estimates
- Save intermediate results