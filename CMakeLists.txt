#---------------------------------------------------------------------------------------------------
# CMake Build Configuration for the Ray Tracing Weekend Series
#
# See README.md for guidance.
#---------------------------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.1.0...4.0.0 )

# Option to choose compiler (set before project())
# Usage: cmake .. -DUSE_CLANG=OFF
option(USE_CLANG "Use Clang compiler instead of GCC" ON)

if(USE_CLANG)
    set(CMAKE_C_COMPILER "clang")
    set(CMAKE_CXX_COMPILER "clang++")
    message(STATUS "Using Clang compiler")
endif()

project ( 302_RAYTRACE LANGUAGES CXX CUDA )

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set ( CMAKE_CXX_STANDARD          17 )
set ( CMAKE_CXX_STANDARD_REQUIRED ON )
set ( CMAKE_CXX_EXTENSIONS        OFF )

# CUDA settings
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    
    # Enable parallel CUDA compilation for faster builds
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -t 0")
    
    message(STATUS "CUDA found and enabled")
    set(CUDA_FOUND TRUE)
else()
    message(WARNING "CUDA not found. CUDA functionality will be disabled.")
    set(CUDA_FOUND FALSE)
endif()

# Source
set ( EXTERNAL
  src/external/stb_image_write.h
  src/external/stb_image.h
)

set ( SOURCE_302_RAYTRACER
  # Main source
  src/302_raytracer/main.cc
  
  # Core headers
  src/302_raytracer/utils.h
  src/302_raytracer/vec3.h
  src/302_raytracer/color.h
  src/302_raytracer/ray.h
  src/302_raytracer/interval.h
  src/302_raytracer/constants.h
  src/302_raytracer/rnd_gen.h
  
  # Geometry
  src/302_raytracer/hittable.h
  src/302_raytracer/hittable_list.h
  src/302_raytracer/sphere.h
  src/302_raytracer/cube.h
  src/302_raytracer/rectangle.h
  
  # Materials
  src/302_raytracer/material.h
  
  # Scene description (NEW)
  src/302_raytracer/scene_description.h
  src/302_raytracer/scene_builder.h
  
  # Camera and renderers
  src/302_raytracer/camera/camera.h
  src/302_raytracer/camera/camera_base.h  
  
  # SDL renderers (if SDL2 is available)
  src/302_raytracer/camera/sdl/camera_progressive_sdl_cuda.h
  src/302_raytracer/camera/sdl/sdl_gui_handler.h
  src/302_raytracer/camera/sdl/sdl_gui_controls.h
  
  # CPU renderers
  src/302_raytracer/cpu_renderers/cpu_ray_tracer.h
  src/302_raytracer/cpu_renderers/renderer_cpu.h
  src/302_raytracer/cpu_renderers/renderer_cpu_parallel.h
  
  # GPU renderers
  src/302_raytracer/gpu_renderers/renderer_cuda.h
  src/302_raytracer/gpu_renderers/renderer_cuda.cu
  src/302_raytracer/gpu_renderers/cuda_float3.cuh
  src/302_raytracer/gpu_renderers/cuda_utils.cuh
  src/302_raytracer/gpu_renderers/cuda_scene.cuh
  src/302_raytracer/gpu_renderers/scene_builder_cuda.cu
)

# Add include directories to avoid ugly ../../ paths
include_directories(src)
include_directories(src/302_raytracer)
include_directories(src/302_raytracer/camera)
include_directories(src/302_raytracer/camera/sdl)
include_directories(src/302_raytracer/cpu_renderers)
include_directories(src/302_raytracer/gpu_renderers)

# Ensure compile_commands.json is generated in the source directory
set(CMAKE_COMPILE_COMMANDS_OUTPUT_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Specific compiler flags below. We're not going to add options for all possible compilers, but if
# you're new to CMake, the following may be a helpful example if you're using a
# different compiler or want to set different compiler options.
message (STATUS "Compiler ID: " ${CMAKE_CXX_COMPILER_ID})
message (STATUS "Release flags: " ${CMAKE_CXX_FLAGS_RELEASE})
message (STATUS "Debug flags: " ${CMAKE_CXX_FLAGS_DEBUG})

# Set to C++17
include(ProcessorCount)
ProcessorCount(N)
message(STATUS "Number of processor cores: " ${N})
if(NOT N EQUAL 0)
    math(EXPR PARALLEL_CORES "${N} - 2")
    if(PARALLEL_CORES LESS 1)
        set(CMAKE_BUILD_PARALLEL_LEVEL 1)
    else()
        set(CMAKE_BUILD_PARALLEL_LEVEL ${PARALLEL_CORES})
    endif()
    message(STATUS "Setting parallel build level to: " ${CMAKE_BUILD_PARALLEL_LEVEL})
endif()

# Ninja-specific optimizations
if(CMAKE_GENERATOR STREQUAL "Ninja")
    message(STATUS "Using Ninja generator - enabling optimizations")
    # Enable colored output for Ninja builds (CXX only, not CUDA)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-color=always>")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fcolor-diagnostics>")
    endif()
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Apply these warnings only to C++ files, not CUDA
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wnon-virtual-dtor>") # Class has virtual functions, but its destructor is not virtual
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wreorder>") # Data member will be initialized after [other] data member
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wmaybe-uninitialized>") # Variable improperly initialized
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wunused-variable>") # Variable is defined but unused
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Apply these warnings only to C++ files, not CUDA
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wnon-virtual-dtor>") # Class has virtual functions, but its destructor is not virtual
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wreorder>") # Data member will be initialized after [other] data member
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wsometimes-uninitialized>") # Variable improperly initialized
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wunused-variable>") # Variable is defined but unused
endif()

# Executables
add_executable(302_raytracer ${EXTERNAL} ${SOURCE_302_RAYTRACER})

# Find and link SDL2 for real-time display (optional)
find_package(SDL2 QUIET)
if(SDL2_FOUND)
    include_directories(${SDL2_INCLUDE_DIRS})
    target_link_libraries(302_raytracer ${SDL2_LIBRARIES})
    add_definitions(-DSDL2_FOUND)
    message(STATUS "SDL2 found and linked for real-time display")
    
    # Try to find SDL2_ttf for TrueType font rendering
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(SDL2_TTF QUIET SDL2_ttf)
        if(SDL2_TTF_FOUND)
            include_directories(${SDL2_TTF_INCLUDE_DIRS})
            target_link_libraries(302_raytracer ${SDL2_TTF_LIBRARIES})
            add_definitions(-DSDL2_TTF_FOUND)
            message(STATUS "SDL2_ttf found and linked for TrueType font rendering")
        else()
            message(WARNING "SDL2_ttf not found. TrueType font rendering will be disabled.")
            message(WARNING "On Ubuntu/Debian: sudo apt-get install libsdl2-ttf-dev")
            message(WARNING "On macOS: brew install sdl2_ttf")
        endif()
    endif()
else()
    message(WARNING "SDL2 not found. Real-time display will be disabled. Install SDL2 to enable real-time rendering.")
    message(WARNING "On Ubuntu/Debian: sudo apt-get install libsdl2-dev")
    message(WARNING "On macOS: brew install sdl2")
    message(WARNING "On Windows: Download from https://www.libsdl.org/download-2.0.php")    
    message(STATUS "Continuing without SDL2...")
endif()

# Create an empty directory for storing the images
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/res)

if(CUDA_FOUND)
    set_property(TARGET 302_raytracer PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(302_raytracer curand)
    
    # CUDA optimization flags
    # Set CUDA architectures (adjust based on your GPU - this covers common GPUs)
    set(CMAKE_CUDA_ARCHITECTURES "all-major")
    
    # CUDA compiler flags for optimization and warning suppression
    target_compile_options(302_raytracer PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:            
            -Xcompiler=-fno-strict-aliasing -rdynamic -lineinfo # Optimize memory access
            --expt-relaxed-constexpr     # Allow relaxed constexpr
            --expt-extended-lambda       # Enable extended lambda support
            #-w                           # Suppress all CUDA warnings
            $<$<CONFIG:Release>:-O3>     # Maximum optimization for Release
            #$<$<CONFIG:Release>:--use_fast_math>  # Fast math for Release only
            $<$<CONFIG:Debug>:-O0>       # No optimization for Debug
            $<$<CONFIG:Debug>:-g>        # Generate host debug info
            $<$<CONFIG:Debug>:-lineinfo> # Generate line-number information (safer than -G)
        >
    )
    
    # Set device link options to suppress nvlink warnings
    set_target_properties(302_raytracer PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_LINK_OPTIONS_INIT "-Xnvlink=--suppress-stack-size-warning,-w"
    )
    
    # Additional nvlink flags to suppress all warnings
    target_link_options(302_raytracer PRIVATE
        $<DEVICE_LINK:-Xnvlink=--suppress-stack-size-warning,-w>
    )
endif()