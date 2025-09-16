# Explanation of `renderPixelsCUDA` and `renderPixelsKernel`

This document provides a detailed explanation of the CUDA implementation for rendering pixels in the ray tracer, focusing on the `renderPixelsCUDA` host function and the `renderPixelsKernel` device kernel found in `src/v0_single_threaded/camera_cuda.cu`.

## Overview

The rendering process is parallelized using CUDA by assigning the computation of each pixel's color to a separate GPU thread. This allows for a massive speedup compared to a single-threaded CPU implementation. The process involves three main stages:

1.  **Setup on the Host (CPU)**: The C++ code prepares the rendering parameters and allocates memory on the GPU.
2.  **Execution on the Device (GPU)**: A CUDA kernel is launched, where thousands of threads concurrently trace rays to compute pixel colors.
3.  **Data Transfer back to Host (CPU)**: The final rendered image is copied from GPU memory back to CPU memory to be saved.

---

## Host Function: `renderPixelsCUDA`

This function is the C++ entry point that orchestrates the entire CUDA rendering pipeline. It runs on the CPU.

```cpp
extern "C" unsigned long long renderPixelsCUDA(unsigned char* image, int width, int height,
                                               double cam_center_x, double cam_center_y, double cam_center_z,
                                               double pixel00_x, double pixel00_y, double pixel00_z,
                                               double delta_u_x, double delta_u_y, double delta_u_z,
                                               double delta_v_x, double delta_v_y, double delta_v_z,
                                               int samples_per_pixel, int max_depth)
```

### Key Steps and CUDA Specifics:

1.  **GPU Memory Allocation (`cudaMalloc`)**:
    *   `cudaMalloc(&d_image, image_size)`: Allocates a buffer on the GPU's VRAM to store the final image data. `d_image` is a pointer to this device memory.
    *   `cudaMalloc(&d_ray_count, sizeof(unsigned long long))`: Allocates memory for a single 64-bit integer on the device. This will be used as an atomic counter to track the total number of rays traced by all threads.

2.  **GPU Memory Initialization (`cudaMemset`)**:
    *   `cudaMemset(d_ray_count, 0, ...)`: Initializes the ray counter on the device to zero.
    *   `cudaMemset(d_image, 0, ...)`: Clears the device image buffer.

3.  **Execution Configuration (Grid and Block Dimensions)**:
    *   `dim3 block_size(32, 4);`: Defines the dimensions of a **thread block**. Here, each block contains `32 * 4 = 128` threads arranged in a 2D layout. Using rectangular blocks is a common heuristic to improve performance by optimizing memory access patterns and avoiding artifacts that can sometimes arise from perfectly square configurations.
    *   `dim3 grid_size(...)`: Defines the dimensions of the **grid of blocks**. The calculation `(width + block_size.x - 1) / block_size.x` is a standard CUDA idiom to ensure enough blocks are launched to cover every pixel of the image, even if the image dimensions are not perfect multiples of the block dimensions.

4.  **Kernel Launch (`<<<...>>>`)**:
    *   `renderPixelsKernel<<<grid_size, block_size>>>(...);`: This is the most critical part. It launches the `renderPixelsKernel` function on the GPU.
    *   The `<<<grid_size, block_size>>>` syntax tells the CUDA runtime how many threads to launch and how to group them. In this case, it launches a 2D grid of thread blocks.
    *   All parameters (camera data, image dimensions, device pointers) are passed from the host to the kernel. Note that `double` precision values from the host are cast to `float`, as the kernel is optimized to use single-precision arithmetic, which is much faster on most consumer GPUs.

5.  **Synchronization and Error Checking**:
    *   `cudaGetLastError()`: Since kernel launches are asynchronous (the CPU code continues immediately without waiting for the GPU to finish), this function is called to check for any errors that might have occurred when launching the kernel.
    *   `cudaDeviceSynchronize()`: This is a blocking call that pauses the CPU thread until all previously issued commands on the GPU have completed. This is essential to ensure the rendering is finished before we try to copy the results back.

6.  **Data Transfer from Device to Host (`cudaMemcpy`)**:
    *   `cudaMemcpy(image, d_image, ..., cudaMemcpyDeviceToHost)`: Copies the rendered pixel data from the GPU's memory (`d_image`) back to the host's main memory (`image`).
    *   `cudaMemcpy(&host_ray_count, d_ray_count, ...)`: Copies the final ray count from the GPU back to a host variable.

7.  **Cleanup (`cudaFree`)**:
    *   `cudaFree(d_image)` and `cudaFree(d_ray_count)`: Releases the memory that was allocated on the GPU, preventing memory leaks in VRAM.

---

## Device Kernel: `renderPixelsKernel`

This function runs on the GPU. A separate instance of this kernel (a thread) is executed for each pixel in the output image.

```cpp
__global__ void renderPixelsKernel(unsigned char* image, int width, int height, ..., unsigned long long* ray_count)
```

### Key Steps and CUDA Specifics:

1.  **`__global__` Specifier**: This keyword declares the function as a "kernel" that can be called from the host (CPU) and is executed on the device (GPU).

2.  **Global Thread-to-Pixel Mapping**:
    *   `int x = blockIdx.x * blockDim.x + threadIdx.x;`
    *   `int y = blockIdx.y * blockDim.y + threadIdx.y;`
    *   This is the standard CUDA pattern for computing a unique global ID for each thread. `blockIdx` gives the ID of the current block in the grid, `blockDim` gives the size of the block, and `threadIdx` gives the ID of the current thread within its block. This calculation maps each thread to a unique `(x, y)` pixel coordinate.

3.  **Random State Initialization (`curand_init`)**:
    *   Each thread must have its own independent random number generator state to avoid visual artifacts.
    *   `curand_init(...)`: Initializes the cuRAND library's state for the current thread. The seed is made unique for each pixel by combining its coordinates and the system clock, ensuring that each pixel's anti-aliasing and material scattering calculations are statistically independent.

4.  **Ray Tracing Loop**:
    *   For each sample per pixel, the thread calculates a unique ray direction with a random offset for anti-aliasing.
    *   It calls the `ray_color` device function, which recursively traces the ray through the scene.

5.  **Atomic Operations (`atomicAdd`)**:
    *   Inside `ray_color`, the global ray counter is incremented using `atomicAdd(ray_count, 1)`.
    *   An atomic operation is crucial here because thousands of threads are trying to increment the same memory location (`d_ray_count`) simultaneously. `atomicAdd` ensures that these operations are serialized, preventing race conditions and guaranteeing a correct final count.

6.  **Writing Output**:
    *   After accumulating the color from all samples, the thread performs gamma correction and converts the final floating-point color value to an 8-bit RGB triplet.
    *   It then writes these three bytes directly to the correct location in the global image buffer (`d_image`). Since each thread is responsible for a unique pixel, there are no write conflicts between threads at this stage.
