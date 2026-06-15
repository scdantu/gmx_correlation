cmake_minimum_required(VERSION 3.28)
cmake_policy(VERSION 3.28)

project(gmx_correlation CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type." FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(GMX_DOUBLE "Use double precision GROMACS" OFF)
option(GMX_CORRELATION_MPI "Enable MPI work sharing for the correlation matrix" OFF)
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

add_executable(gmx_correlation
    gmx_correlation.cpp
    correlation_core.cpp
    kraskov.cpp)

set_target_properties(gmx_correlation PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    COMPILE_FLAGS "${GROMACS_CXX_FLAGS}")

target_link_libraries(gmx_correlation PRIVATE ${GROMACS_LIBRARIES})

if(GMX_CORRELATION_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    target_compile_definitions(gmx_correlation PRIVATE GMX_CORRELATION_USE_MPI)
    target_link_libraries(gmx_correlation PRIVATE MPI::MPI_CXX)
endif()
