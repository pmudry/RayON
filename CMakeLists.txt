#---------------------------------------------------------------------------------------------------
# CMake Build Configuration for 302_RAYTRACE
# See README.md for guidance.
#---------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.8)

# Option to choose compiler (set before project())
option(USE_CLANG "Use Clang compiler instead of GCC" ON)

project(302_RAYTRACE LANGUAGES CXX CUDA)

if(USE_CLANG)
    set(CMAKE_C_COMPILER "clang")
    set(CMAKE_CXX_COMPILER "clang++")
    message(STATUS "Using Clang compiler")
else()
    set(CMAKE_C_COMPILER "gcc")
    set(CMAKE_CXX_COMPILER "g++")
    message(STATUS "Using gcc / g++ compiler")
endif()


# Ensure compile_commands.json is generated in the source directory
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA settings
include(CheckLanguage)

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_ARCHITECTURES 90) # Set CUDA architectures (adjust based on your GPU - this covers common GPUs)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -t 0") # Enable parallel CUDA compilation for faster builds
    set(CUDA_FOUND TRUE)
else()
    message(WARNING "CUDA not found. CUDA functionality will be disabled.")
    set(CUDA_FOUND FALSE)
endif()

# Source
set(EXTERNAL
    src/external/stb_image_write.h
    src/external/stb_image.h
)

set(SOURCE_302_RAYTRACER
    # Main source, only CC and CU files here
    src/302_raytracer/main.cc
    src/302_raytracer/scenes/yaml_scene_loader.cc
    src/302_raytracer/gpu_renderers/renderer_cuda.cu
    src/302_raytracer/gpu_renderers/cuda_utils.cu
    src/302_raytracer/gpu_renderers/scene_builder_cuda.cu
    src/302_raytracer/gpu_renderers/shaders/render_acc_kernel.cu
    src/302_raytracer/gpu_renderers/shaders/shader_golf.cu
)

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

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Apply these warnings only to C++ files, not CUDA
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wnon-virtual-dtor>") # Class has virtual functions, but its destructor is not virtual
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wreorder>") # Data member will be initialized after [other] data member
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wuninitialized>") # Variable improperly initialized
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wunused-variable>") # Variable is defined but unused
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Apply these warnings only to C++ files, not CUDA
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wnon-virtual-dtor>") # Class has virtual functions, but its destructor is not virtual
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wreorder>") # Data member will be initialized after [other] data member
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wsometimes-uninitialized>") # Variable improperly initialized
endif()

# Executables
add_executable(302_raytracer ${EXTERNAL} ${SOURCE_302_RAYTRACER})

# Add include directories using modern CMake target_include_directories
# This is the proper way - affects only this target and propagates to clangd via compile_commands.json
target_include_directories(302_raytracer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/external

    ${CMAKE_SOURCE_DIR}/src/302_raytracer
    ${CMAKE_SOURCE_DIR}/src/302_raytracer/camera
    ${CMAKE_SOURCE_DIR}/src/302_raytracer/camera/sdl
    ${CMAKE_SOURCE_DIR}/src/302_raytracer/scenes
    ${CMAKE_SOURCE_DIR}/src/302_raytracer/data_structures
    
    # CPU renderers
    ${CMAKE_SOURCE_DIR}/src/302_raytracer/cpu_renderers
    ${CMAKE_SOURCE_DIR}/src/302_raytracer/cpu_renderers/data_structures
    ${CMAKE_SOURCE_DIR}/src/302_raytracer/cpu_renderers/utils
    ${CMAKE_SOURCE_DIR}/src/302_raytracer/cpu_renderers/cpu_shapes
    
    # GPU renderers
    ${CMAKE_SOURCE_DIR}/src/302_raytracer/gpu_renderers
    ${CMAKE_SOURCE_DIR}/src/302_raytracer/gpu_renderers/shaders
    ${CMAKE_SOURCE_DIR}/src/302_raytracer/gpu_renderers/materials
    ${CMAKE_SOURCE_DIR}/src/302_raytracer/gpu_renderers/materials/legacy
)

target_compile_definitions(302_raytracer PRIVATE
    RT_BUILD_TYPE_STRING="$<IF:$<BOOL:${CMAKE_BUILD_TYPE}>,${CMAKE_BUILD_TYPE},$<CONFIG>>"
)

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

# Create an output directory for rendered images
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/rendered_images)

if(CUDA_FOUND)
    set_property(TARGET 302_raytracer PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(302_raytracer curand)

    # CUDA compiler flags for optimization and warning suppression
    target_compile_options(302_raytracer PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -maxrregcount 80               # Limit max registers per thread for better occupancy (TO BE TUNED)
        -Xcompiler=-fno-strict-aliasing -rdynamic -lineinfo # Optimize memory access
        --expt-relaxed-constexpr # Allow relaxed constexpr
        --expt-extended-lambda # Enable extended lambda support
        #-w                           # Suppress all CUDA warnings
        $<$<CONFIG:Release>:-O3> # Maximum optimization for Release
        #$<$<CONFIG:Release>:--use_fast_math>  # Fast math for Release only
        $<$<CONFIG:Debug>:-O0> # No optimization for Debug
        $<$<CONFIG:Debug>:-g> # Generate host debug info
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
