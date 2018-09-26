cmake_minimum_required(VERSION 3.4.1)
project(MTA)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(OpenCV REQUIRED)
if(NOT DEFINED EIGEN3_INCLUDE_DIRS)
    find_package(Eigen3 REQUIRED)
endif()
include_directories(${EIGEN3_INCLUDE_DIRS})

set(MTA_include_path "${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories(${MTA_include_path})
set(MTA_include 
    "${MTA_include_path}/Backgrounds.h"
    "${MTA_include_path}/Config.h"
    "${MTA_include_path}/Features.h"
    "${MTA_include_path}/HaarFeature.h"
    "${MTA_include_path}/HaarFeatures.h"
    "${MTA_include_path}/HistogramFeatures.h"
    "${MTA_include_path}/HOGFeatures.h"
    "${MTA_include_path}/ImageRep.h"
    "${MTA_include_path}/IterTracker.h"
    "${MTA_include_path}/Kernels.h"
    "${MTA_include_path}/LaRank.h"
    "${MTA_include_path}/MultiFeatures.h"
    "${MTA_include_path}/RawFeatures.h"
    "${MTA_include_path}/Rect.h"
    "${MTA_include_path}/Sample.h"
    "${MTA_include_path}/Sampler.h"
)

set(MTA_src_path "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(MTA_src
    "${MTA_src_path}/Backgrounds.cpp"
    "${MTA_src_path}/Config.cpp"
    "${MTA_src_path}/Features.cpp"
    "${MTA_src_path}/HaarFeature.cpp"
    "${MTA_src_path}/HaarFeatures.cpp"
    "${MTA_src_path}/HistogramFeatures.cpp"
    "${MTA_src_path}/HOGFeature.cpp"
    "${MTA_src_path}/ImageRep.cpp"
    "${MTA_src_path}/IterTracker.cpp"
    "${MTA_src_path}/LaRank.cpp"
    "${MTA_src_path}/MultiFeatures.cpp"
    "${MTA_src_path}/RawFeatures.cpp"
    "${MTA_src_path}/Sampler.cpp"
    "${MTA_src_path}/main.cpp"
)

add_executable(MTA ${MTA_include} ${MTA_src})
target_link_libraries(MTA ${OpenCV_LIBS})

set_target_properties(MTA PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)