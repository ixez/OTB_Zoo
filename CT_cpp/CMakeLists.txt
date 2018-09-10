cmake_minimum_required(VERSION 3.0)
project(CT)

set(dep_libs "")

# ---[ OpenCV
find_package(OpenCV REQUIRED)
set(dep_libs ${dep_libs} ${OpenCV_LIBS})

# ---[ Source files
set(src_dir ${CMAKE_CURRENT_SOURCE_DIR})
set(src
        ${src_dir}/CompressiveTracker.cpp
        ${src_dir}/TrackTask.cpp
        ${src_dir}/main.cpp
)

set(include_dir ${CMAKE_CURRENT_SOURCE_DIR})
set(include
        ${include_dir}/CompressiveTracker.h
        ${include_dir}/TrackTask.h
)

# --- [ Executable
add_executable(CT ${src} ${include})
target_include_directories(CT PRIVATE ${include_dir})
target_link_libraries(CT ${dep_libs})

set_target_properties(CT PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)