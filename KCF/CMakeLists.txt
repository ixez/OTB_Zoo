cmake_minimum_required(VERSION 3.0)
project(zKCF)

find_package(OpenCV REQUIRED)

set(src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(src
        ${src_dir}/fhog.cpp
        ${src_dir}/FFTTools.cpp
        ${src_dir}/KCF.cpp
        ${src_dir}/Run.cpp

        ${src_dir}/FkFactory.cpp

        ${src_dir}/Kernel/GaussianKernel.cpp
        ${src_dir}/Feature/HogFeature.cpp
        ${src_dir}/Feature/HogLabFeature.cpp
        ${src_dir}/Feature/RawFeature.cpp
)

set(include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(include
        ${include_dir}/ITracker.h
        ${include_dir}/KCF.h
        ${include_dir}/fhog.hpp
        ${include_dir}/labdata.hpp
        ${include_dir}/recttools.hpp
        ${include_dir}/FFTTools.hpp

        ${include_dir}/Def.h
        ${include_dir}/FkFactory.h

        ${include_dir}/Kernel/IKernel.h
        ${include_dir}/Kernel/GaussianKernel.h

        ${include_dir}/Feature/IFeature.h
        ${include_dir}/Feature/HogFeature.h
        ${include_dir}/Feature/HogLabFeature.h
        ${include_dir}/Feature/RawFeature.h
)

add_executable(zKCF ${src} ${include})
target_include_directories(zKCF PRIVATE ${include_dir})
target_link_libraries(zKCF ${OpenCV_LIBS})

set_target_properties(zKCF PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)



