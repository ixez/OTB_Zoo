cmake_minimum_required(VERSION 2.8)
project(DSST)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# add OpenCV
set(OPENCV_DIR_HINT "/home/zeke/Libs/opencv/opencv-3.4.3/build")
find_package(OpenCV REQUIRED HINTS ${OPENCV_DIR_HINT})

# include
set(DSST_include_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(DSST_include
        "${DSST_include_path}/dsst_tracker.hpp"
        "${DSST_include_path}/dsst_debug.hpp"
        "${DSST_include_path}/TrackTask.h"
)
include_directories(${DSST_include_path})

# src
set(DSST_src_path "${CMAKE_CURRENT_SOURCE_DIR}")
set(DSST_src
        "${DSST_src_path}/main.cpp"
        "${DSST_src_path}/TrackTask.cpp"
)

# add 3rdparty source
set(CF_TCLAP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tclap")
set(CF_CV_EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cv_ext")
set(CF_PIOTR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/piotr")
set(CF_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")
set(CF_PIOTR_SOURCES
        ${CF_PIOTR_DIR}/gradientMex.hpp
        ${CF_PIOTR_DIR}/src/gradientMex.cpp
        ${CF_PIOTR_DIR}/src/sse.hpp
        ${CF_PIOTR_DIR}/src/wrappers.hpp)
set(CF_LIB_COMMON_SOURCES
        ${CF_COMMON_DIR}/feature_channels.hpp
        ${CF_COMMON_DIR}/mat_consts.hpp
        ${CF_COMMON_DIR}/math_helper.hpp
        ${CF_COMMON_DIR}/math_helper.cpp
        ${CF_COMMON_DIR}/cf_tracker.hpp
        ${CF_COMMON_DIR}/tracker_debug.hpp
        ${CF_COMMON_DIR}/scale_estimator.hpp
        ${CF_COMMON_DIR}/cv_ext.hpp
        ${CF_CV_EXT_DIR}/shift.cpp
        ${CF_CV_EXT_DIR}/shift.hpp
        ${CF_CV_EXT_DIR}/math_spectrums.cpp
        ${CF_CV_EXT_DIR}/math_spectrums.hpp
        ${CF_CV_EXT_DIR}/psr.hpp
        ${CF_PIOTR_SOURCES})
include_directories(${CF_COMMON_DIR}
                    ${CF_TCLAP_DIR}
                    ${CF_CV_EXT_DIR}
                    ${CF_PIOTR_DIR}
                    ${CF_PIOTR_DIR}/src
)

add_executable(DSST ${DSST_src} ${DSST_include}
        ${CF_PIOTR_SOURCES} ${CF_LIB_COMMON_SOURCES})
target_link_libraries(DSST ${OpenCV_LIBS})

set_target_properties(DSST PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)