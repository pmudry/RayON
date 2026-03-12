#---------------------------------------------------------------------------------------------------
# CMake Build Configuration for RayON Raytracer
# See README.md for guidance.
#---------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.8)

# Option to choose compiler (set before project())
option(USE_CLANG "Use Clang compiler instead of GCC" ON)

if(USE_CLANG)
    set(CMAKE_C_COMPILER "clang")
    set(CMAKE_CXX_COMPILER "clang++")
    message(STATUS "Using Clang compiler")
else()
    set(CMAKE_C_COMPILER "gcc")
    set(CMAKE_CXX_COMPILER "g++")
    message(STATUS "Using gcc / g++ compiler")
endif()

project(RAYON_RAYTRACER LANGUAGES CXX CUDA)

# Option to enable diagnostic output
option(ENABLE_DIAGS "Enable diagnostic output in CUDA and CPU renderers" OFF)

# Ensure compile_commands.json is generated in the source directory
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Option to enable/disable clang-tidy static analysis
option(ENABLE_CLANG_TIDY "Enable clang-tidy static analysis during build" OFF)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable clang-tidy if available and requested
if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE clang-tidy)
    if(CLANG_TIDY_EXE)
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-header-filter=^((?!src/external).)*$ --config")
        message(STATUS "clang-tidy found and enabled (ENABLE_CLANG_TIDY=ON)")
    else()
        message(WARNING "clang-tidy enabled via ENABLE_CLANG_TIDY but clang-tidy executable not found")
    endif()
else()
    message(STATUS "clang-tidy disabled (set ENABLE_CLANG_TIDY=ON to enable)")
endif()

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_ARCHITECTURES "native") # Set CUDA architecture to the machine where it's built
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
    src/external/imgui/imgui.cpp
    src/external/imgui/imgui_demo.cpp
    src/external/imgui/imgui_draw.cpp
    src/external/imgui/imgui_tables.cpp
    src/external/imgui/imgui_widgets.cpp
    src/external/imgui/imgui_impl_sdl2.cpp
    src/external/imgui/imgui_impl_sdlrenderer2.cpp
)

set(SOURCE_RAYON
    # Main source, only CC and CU files here
    src/rayon/main.cc
    src/rayon/render/render_coordinator.cpp
    src/rayon/scenes/yaml_scene_loader.cc
    src/rayon/gpu_renderers/renderer_cuda_device.cu
    src/rayon/gpu_renderers/cuda_raytracer.cu
    src/rayon/gpu_renderers/cuda_utils.cu
    src/rayon/gpu_renderers/scene_builder_cuda.cu
    src/rayon/gpu_renderers/shaders/render_acc_kernel.cu
    src/rayon/gpu_renderers/shaders/shader_golf.cu
)

# Enable colored compiler output for Make builds
if(CMAKE_GENERATOR STREQUAL "Unix Makefiles")
    message(STATUS "Using Make generator - enabling colored output")
    # Force Make to always show colors via environment variable
    set(CMAKE_COLOR_MAKEFILE ON)
    # Add color flags for compilers (safe options that work with both GCC and Clang)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-color=always>")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fcolor-diagnostics>")
    endif()
elseif(CMAKE_GENERATOR STREQUAL "Ninja")
    message(STATUS "Using Ninja generator - colors enabled by default")
    # Ninja automatically supports colored output
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
add_executable(rayon ${EXTERNAL} ${SOURCE_RAYON})

# Recursively find all subdirectories under src/
# This automates the include directory listing for both CMake and .clangd
# No need to manually add new subdirectories here - they will be included automatically
file(GLOB_RECURSE ALL_ENTRIES LIST_DIRECTORIES true "${CMAKE_SOURCE_DIR}/src/*")
set(INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/src")
foreach(entry ${ALL_ENTRIES})
    if(IS_DIRECTORY ${entry})
        list(APPEND INCLUDE_DIRS ${entry})
    endif()
endforeach()

# Add include directories using modern CMake target_include_directories
# This is the proper way - affects only this target and propagates to clangd via compile_commands.json
target_include_directories(rayon PRIVATE ${INCLUDE_DIRS})

target_compile_definitions(rayon PRIVATE
    RT_BUILD_TYPE_STRING="$<IF:$<BOOL:${CMAKE_BUILD_TYPE}>,${CMAKE_BUILD_TYPE},$<CONFIG>>"
)

# Add DIAGS definition if enabled
if(ENABLE_DIAGS)
    target_compile_definitions(rayon PRIVATE DIAGS)
    message(STATUS "Diagnostics enabled (ENABLE_DIAGS=ON)")
else()
    message(STATUS "Diagnostics disabled (set ENABLE_DIAGS=ON to enable)")
endif()

# Generate .clangd with -I flags
# This updates the .clangd file's Add section with the current include directories
# The rest of .clangd remains untouched
file(READ ${CMAKE_SOURCE_DIR}/.clangd CLANGD_CONTENT)
string(FIND "${CLANGD_CONTENT}" "Add:" ADD_POS)
string(FIND "${CLANGD_CONTENT}" "Diagnostics:" DIAG_POS)
string(SUBSTRING "${CLANGD_CONTENT}" 0 ${ADD_POS} BEFORE_ADD)
string(SUBSTRING "${CLANGD_CONTENT}" ${DIAG_POS} -1 AFTER_ADD)
set(NEW_ADD "Add:\n    - --no-cuda-version-check\n    # The include section is automatically generated my cmake at build, do NOT modify.\n")
foreach(dir ${INCLUDE_DIRS})
    set(NEW_ADD "${NEW_ADD}    - -I${dir}\n")
endforeach()
set(NEW_CLANGD "${BEFORE_ADD}${NEW_ADD}${AFTER_ADD}")
file(WRITE ${CMAKE_SOURCE_DIR}/.clangd "${NEW_CLANGD}")

# Find and link SDL2 for real-time display (optional)
find_package(SDL2 QUIET)
if(SDL2_FOUND)
    include_directories(${SDL2_INCLUDE_DIRS})
    target_link_libraries(rayon ${SDL2_LIBRARIES})
    add_definitions(-DSDL2_FOUND)
    message(STATUS "SDL2 found and linked for real-time display")

    # Try to find SDL2_ttf for TrueType font rendering
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(SDL2_TTF QUIET SDL2_ttf)
        if(SDL2_TTF_FOUND)
            include_directories(${SDL2_TTF_INCLUDE_DIRS})
            target_link_libraries(rayon ${SDL2_TTF_LIBRARIES})
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
    set_property(TARGET rayon PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(rayon curand)

    # CUDA compiler flags for optimization and warning suppression
    target_compile_options(rayon PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -maxrregcount 80               # Profiled: 64 regs gave 67% occupancy but spill hurt at high res
        -Xcompiler=-fno-strict-aliasing -rdynamic -lineinfo # Optimize memory access
        --expt-relaxed-constexpr # Allow relaxed constexpr
        --expt-extended-lambda # Enable extended lambda support
        #-w                           # Suppress all CUDA warnings
        $<$<CONFIG:Release>:-O3> # Maximum optimization for Release
        $<$<CONFIG:Release>:--use_fast_math>  # Fast math for Release (fast rsqrtf, FMA, relaxed denormals)
        $<$<CONFIG:Debug>:-O0> # No optimization for Debug
        $<$<CONFIG:Debug>:-g> # Generate host debug info
        $<$<CONFIG:Debug>:-lineinfo> # Generate line-number information (safer than -G)
        >
    )

    # Set device link options to suppress nvlink warnings
    set_target_properties(rayon PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_LINK_OPTIONS_INIT "-Xnvlink=--suppress-stack-size-warning,-w"
    )

    # Additional nvlink flags to suppress all warnings
    target_link_options(rayon PRIVATE
        $<DEVICE_LINK:-Xnvlink=--suppress-stack-size-warning,-w>
    )
endif()
