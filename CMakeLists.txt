cmake_minimum_required(VERSION 3.28)
cmake_policy(VERSION 3.28)

project(gmx_correlation CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type." FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(GMX_DOUBLE "Use double precision GROMACS" OFF)
option(GMX_CORRELATION_MPI "Enable MPI work sharing for the correlation matrix" OFF)
option(GMX_CORRELATION_CUDA "Enable CUDA GPU acceleration for the Kraskov estimator (Linux/NVIDIA only)" OFF)
option(GMX_CORRELATION_METAL "Enable Metal GPU acceleration for the Kraskov estimator (macOS only)" OFF)

# Auto-enable Metal on macOS if neither GPU backend is explicitly requested
if(APPLE AND NOT GMX_CORRELATION_CUDA AND NOT GMX_CORRELATION_METAL)
    set(GMX_CORRELATION_METAL ON CACHE BOOL "Auto-enabled on macOS" FORCE)
endif()
set(GMX_SUFFIX "" CACHE STRING "Suffix for the GROMACS installation to use")

if(GMX_DOUBLE AND NOT GMX_SUFFIX)
    set(GROMACS_SUFFIX "_d")
else()
    set(GROMACS_SUFFIX ${GMX_SUFFIX})
endif()

find_package(GROMACS 2025 REQUIRED)

if(APPLE AND TARGET Gromacs::libgromacs)
    # Some Homebrew GROMACS packages have exported an absolute SDK path for
    # libm.tbd. When that SDK is no longer installed, make treats it as a
    # missing build prerequisite. Replacing only dead SDK libm paths with the
    # normal `m` link item keeps the workaround local to affected macOS systems.
    get_target_property(_gmx_link_libraries Gromacs::libgromacs INTERFACE_LINK_LIBRARIES)
    if(_gmx_link_libraries)
        set(_gmx_fixed_link_libraries "")
        foreach(_gmx_link_library IN LISTS _gmx_link_libraries)
            if(_gmx_link_library MATCHES "/MacOSX[^/]+\\.sdk/usr/lib/libm\\.tbd$" AND NOT EXISTS "${_gmx_link_library}")
                list(APPEND _gmx_fixed_link_libraries m)
            else()
                list(APPEND _gmx_fixed_link_libraries "${_gmx_link_library}")
            endif()
        endforeach()
        set_target_properties(Gromacs::libgromacs PROPERTIES
                              INTERFACE_LINK_LIBRARIES "${_gmx_fixed_link_libraries}")
    endif()
    unset(_gmx_fixed_link_libraries)
    unset(_gmx_link_libraries)
endif()

gromacs_check_double(GMX_DOUBLE)
gromacs_check_compiler(CXX)

include_directories(${GROMACS_INCLUDE_DIRS})
add_definitions(${GROMACS_DEFINITIONS})

# ---- CUDA (Linux / NVIDIA) ----
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

# ---- Metal (macOS) ----
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

# ---- Source list ----
set(GMX_CORRELATION_SOURCES
    gmx_correlation.cpp
    correlation_core.cpp
    kraskov.cpp
    gcmi.cpp
    transfer_entropy.cpp)

if(GMX_CORRELATION_CUDA)
    list(APPEND GMX_CORRELATION_SOURCES kraskov_gpu.cu)
endif()
if(GMX_CORRELATION_METAL)
    list(APPEND GMX_CORRELATION_SOURCES kraskov_metal.mm gcmi_metal.mm te_metal.mm)
    set_source_files_properties(kraskov_metal.mm gcmi_metal.mm te_metal.mm PROPERTIES
        COMPILE_FLAGS "-fobjc-arc")
endif()

add_executable(gmx_correlation ${GMX_CORRELATION_SOURCES})

if(GMX_CORRELATION_CUDA)
    set_source_files_properties(kraskov_gpu.cu PROPERTIES LANGUAGE CUDA)
endif()

set_target_properties(gmx_correlation PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    COMPILE_FLAGS "${GROMACS_CXX_FLAGS}")

if(GMX_CORRELATION_CUDA)
    set_target_properties(gmx_correlation PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF)
    target_compile_definitions(gmx_correlation PRIVATE GMX_CORRELATION_USE_CUDA)
    target_link_libraries(gmx_correlation PRIVATE CUDA::cudart)
    message(STATUS "GPU backend: CUDA (kraskov_gpu.cu)")
elseif(GMX_CORRELATION_METAL)
    target_compile_definitions(gmx_correlation PRIVATE GMX_CORRELATION_USE_METAL)
    target_link_libraries(gmx_correlation PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    message(STATUS "GPU backend: Metal (kraskov_metal.mm) — float32 kernel on Apple GPU")
else()
    message(STATUS "GPU backend: none (use -DGMX_CORRELATION_METAL=ON on macOS, -DGMX_CORRELATION_CUDA=ON on Linux)")
endif()

target_link_libraries(gmx_correlation PRIVATE ${GROMACS_LIBRARIES})

# ---- OpenMP (CPU threading for Kraskov pair loop) ----
find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gmx_correlation PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "CPU threading: OpenMP ${OpenMP_CXX_VERSION}")
else()
    message(WARNING "OpenMP not found — CPU path will use 1 thread")
endif()

if(GMX_CORRELATION_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    target_compile_definitions(gmx_correlation PRIVATE GMX_CORRELATION_USE_MPI)
    target_link_libraries(gmx_correlation PRIVATE MPI::MPI_CXX)
endif()
