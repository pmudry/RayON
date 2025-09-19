#---------------------------------------------------------------------------------------------------
# CMake Build Configuration for the Ray Tracing Weekend Series
#
# See README.md for guidance.
#---------------------------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.1.0...4.0.0 )

project ( 302_RAYTRACE LANGUAGES CXX CUDA )

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
  src/302_raytracer/utils.h
  src/302_raytracer/main.cc
  src/302_raytracer/vec3.h
  src/302_raytracer/color.h
  src/302_raytracer/sphere.h
  src/302_raytracer/cube.h
  src/302_raytracer/rectangle.h
  src/302_raytracer/hittable.h
  src/302_raytracer/hittable_list.h
  src/302_raytracer/interval.h
  src/302_raytracer/rnd_gen.h
  src/302_raytracer/camera.h
  src/302_raytracer/camera_cuda.cu
  src/302_raytracer/camera_cuda.h
)

include_directories(src)

# Ensure compile_commands.json is generated in the source directory
set(CMAKE_COMPILE_COMMANDS_OUTPUT_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Specific compiler flags below. We're not going to add options for all possible compilers, but if
# you're new to CMake (like we are), the following may be a helpful example if you're using a
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
    add_compile_options(-Wnon-virtual-dtor) # Class has virtual functions, but its destructor is not virtual
    add_compile_options(-Wreorder) # Data member will be initialized after [other] data member
    add_compile_options(-Wmaybe-uninitialized) # Variable improperly initialized
    add_compile_options(-Wunused-variable) # Variable is defined but unused
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wnon-virtual-dtor) # Class has virtual functions, but its destructor is not virtual
    #add_compile_options(-Wreorder) # Data member will be initialized after [other] data member
    # add_compile_options(-Wsometimes-uninitialized) # Variable improperly initialized
    add_compile_options(-Wunused-variable) # Variable is defined but unused
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
endif()