cmake_minimum_required(VERSION 2.8)
project(KCF)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(OpenCV REQUIRED)

set(KCF_include_path "${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories(${KCF_include_path})

set(KCF_src_path "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(KCF_src_path
        "${KCF_src_path}/fhog.cpp"
        "${KCF_src_path}/kcftracker.cpp"
        "${KCF_src_path}/main.cpp"
        "${KCF_src_path}/TrackTask.cpp"
        )
add_executable( KCF ${KCF_src_path})
target_link_libraries( KCF ${OpenCV_LIBS})

set_target_properties(KCF PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        )