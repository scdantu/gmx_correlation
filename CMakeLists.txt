cmake_minimum_required(VERSION 3.28)
cmake_policy(VERSION 3.28)

project(gmx_correlation CXX)

# ── Build type ────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type." FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ── Options ───────────────────────────────────────────────────────────────────
option(GMX_DOUBLE              "Use double precision GROMACS"                              OFF)
option(GMX_CORRELATION_MPI     "Enable MPI work sharing for the correlation matrix"        OFF)
option(GMX_CORRELATION_CUDA    "Enable CUDA GPU acceleration (Linux/NVIDIA only)"         OFF)
option(GMX_CORRELATION_METAL   "Enable Metal GPU acceleration (macOS only)"               OFF)

# Installation prefix for the compiled binary and Python script
set(GMX_CORRELATION_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/bin"
    CACHE PATH "Directory where gmx_correlation binary is installed")

# Auto-enable Metal on macOS when no GPU backend is explicitly chosen
if(APPLE AND NOT GMX_CORRELATION_CUDA AND NOT GMX_CORRELATION_METAL)
    set(GMX_CORRELATION_METAL ON CACHE BOOL "Auto-enabled on macOS" FORCE)
endif()

set(GMX_SUFFIX "" CACHE STRING "Suffix for the GROMACS installation to use")
if(GMX_DOUBLE AND NOT GMX_SUFFIX)
    set(GROMACS_SUFFIX "_d")
else()
    set(GROMACS_SUFFIX ${GMX_SUFFIX})
endif()

# ── GROMACS ───────────────────────────────────────────────────────────────────
find_package(GROMACS 2025 REQUIRED)

if(APPLE AND TARGET Gromacs::libgromacs)
    # Some Homebrew GROMACS builds export a now-missing SDK path for libm.tbd;
    # replace it with the plain link item so the build doesn't fail.
    get_target_property(_gmx_link_libraries Gromacs::libgromacs INTERFACE_LINK_LIBRARIES)
    if(_gmx_link_libraries)
        set(_gmx_fixed "")
        foreach(_lib IN LISTS _gmx_link_libraries)
            if(_lib MATCHES "/MacOSX[^/]+\\.sdk/usr/lib/libm\\.tbd$" AND NOT EXISTS "${_lib}")
                list(APPEND _gmx_fixed m)
            else()
                list(APPEND _gmx_fixed "${_lib}")
            endif()
        endforeach()
        set_target_properties(Gromacs::libgromacs PROPERTIES
                              INTERFACE_LINK_LIBRARIES "${_gmx_fixed}")
        unset(_gmx_fixed)
    endif()
    unset(_gmx_link_libraries)
endif()

gromacs_check_double(GMX_DOUBLE)
gromacs_check_compiler(CXX)

# lib/ headers must be visible to src/ and to each other
include_directories(
    ${GROMACS_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_definitions(${GROMACS_DEFINITIONS})

# ── CUDA (Linux / NVIDIA) ────────────────────────────────────────────────────
if(GMX_CORRELATION_CUDA)
    if(APPLE)
        message(WARNING "CUDA is not supported on macOS since CUDA 10.2. "
                        "Disabling GMX_CORRELATION_CUDA.")
        set(GMX_CORRELATION_CUDA OFF CACHE BOOL "" FORCE)
    else()
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
    endif()
endif()

# ── Metal (macOS) ─────────────────────────────────────────────────────────────
if(GMX_CORRELATION_METAL)
    if(NOT APPLE)
        message(WARNING "Metal is only available on macOS. Disabling GMX_CORRELATION_METAL.")
        set(GMX_CORRELATION_METAL OFF CACHE BOOL "" FORCE)
    else()
        enable_language(OBJCXX)
        find_library(METAL_FRAMEWORK     Metal     REQUIRED)
        find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    endif()
endif()

# ── Source list ───────────────────────────────────────────────────────────────
set(GMX_CORRELATION_SOURCES
    src/gmx_correlation.cpp
    lib/correlation_core.cpp
    lib/kraskov.cpp
    lib/gcmi.cpp
    lib/transfer_entropy.cpp)

if(GMX_CORRELATION_CUDA)
    list(APPEND GMX_CORRELATION_SOURCES lib/kraskov_gpu.cu)
endif()

if(GMX_CORRELATION_METAL)
    list(APPEND GMX_CORRELATION_SOURCES
         lib/kraskov_metal.mm
         lib/gcmi_metal.mm
         lib/te_metal.mm)
    set_source_files_properties(
         lib/kraskov_metal.mm
         lib/gcmi_metal.mm
         lib/te_metal.mm
         PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(gmx_correlation ${GMX_CORRELATION_SOURCES})

if(GMX_CORRELATION_CUDA)
    set_source_files_properties(lib/kraskov_gpu.cu PROPERTIES LANGUAGE CUDA)
endif()

set_target_properties(gmx_correlation PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    COMPILE_FLAGS "${GROMACS_CXX_FLAGS}")

# ── GPU link ──────────────────────────────────────────────────────────────────
if(GMX_CORRELATION_CUDA)
    set_target_properties(gmx_correlation PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF)
    target_compile_definitions(gmx_correlation PRIVATE GMX_CORRELATION_USE_CUDA)
    target_link_libraries(gmx_correlation PRIVATE CUDA::cudart)
    message(STATUS "GPU backend: CUDA  (lib/kraskov_gpu.cu)")
elseif(GMX_CORRELATION_METAL)
    target_compile_definitions(gmx_correlation PRIVATE GMX_CORRELATION_USE_METAL)
    target_link_libraries(gmx_correlation PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    message(STATUS "GPU backend: Metal (lib/kraskov_metal.mm, gcmi_metal.mm, te_metal.mm) — float32")
else()
    message(STATUS "GPU backend: none  (add -DGMX_CORRELATION_METAL=ON on macOS or -DGMX_CORRELATION_CUDA=ON on Linux)")
endif()

target_link_libraries(gmx_correlation PRIVATE ${GROMACS_LIBRARIES})

# ── OpenMP ────────────────────────────────────────────────────────────────────
find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gmx_correlation PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "CPU threading: OpenMP ${OpenMP_CXX_VERSION}")
else()
    message(WARNING "OpenMP not found — CPU path will use 1 thread")
endif()

# ── MPI ───────────────────────────────────────────────────────────────────────
if(GMX_CORRELATION_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    target_compile_definitions(gmx_correlation PRIVATE GMX_CORRELATION_USE_MPI)
    target_link_libraries(gmx_correlation PRIVATE MPI::MPI_CXX)
    message(STATUS "MPI: enabled")
endif()

# ── Install targets ───────────────────────────────────────────────────────────
install(TARGETS gmx_correlation RUNTIME DESTINATION bin)
install(PROGRAMS bin/analyze_matrix.py DESTINATION bin
        RENAME gmx_analyze_matrix)
