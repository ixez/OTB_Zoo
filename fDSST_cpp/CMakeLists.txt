cmake_minimum_required(VERSION 2.8)
project(fDSST)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(OpenCV REQUIRED)

set(fDSST_include_path "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(fDSST_include
        "${fDSST_include_path}/fdssttracker.hpp"
        "${fDSST_include_path}/fhog.h"
        "${fDSST_include_path}/ffttools.hpp"
        "${fDSST_include_path}/labdata.hpp"
        "${fDSST_include_path}/recttools.hpp"
        "${fDSST_include_path}/sse.hpp"
        "${fDSST_include_path}/SSE2NEON.h"
        "${fDSST_include_path}/TrackTask.h"
)
include_directories(${fDSST_include_path})

set(fDSST_src_path "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(fDSST_src
        "${fDSST_src_path}/fdssttracker.cpp"
        "${fDSST_src_path}/fhog.cpp"
        "${fDSST_src_path}/TrackTask.cpp"
        "${fDSST_src_path}/main.cpp"
)
add_executable(fDSST ${fDSST_src} ${fDSST_include})
target_link_libraries(fDSST ${OpenCV_LIBS})

set_target_properties(fDSST PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)