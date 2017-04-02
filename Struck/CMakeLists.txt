cmake_minimum_required(VERSION 3.4.1)
project(Struck)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(OpenCV REQUIRED)
if(NOT DEFINED EIGEN3_INCLUDE_DIRS)
    find_package(Eigen3 REQUIRED)
endif()

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    ${EIGEN3_INCLUDE_DIRS}
)

file(GLOB_RECURSE SRC RELATIVE ${PROJECT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE HEADERS RELATIVE ${PROJECT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")

add_executable(Struck ${HEADERS} ${SRC})

target_link_libraries(Struck
    ${OpenCV_LIBS}
)
