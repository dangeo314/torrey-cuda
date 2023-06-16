cmake_minimum_required(VERSION 3.10)

project(lajolla LANGUAGES CXX C CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

if(MSVC)
  add_compile_options(/Wall)
else()
  add_compile_options(-Wall)
endif()

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

include_directories(${CMAKE_SOURCE_DIR}/src)

set(SRCS src/3rdparty/miniz.h
         src/3rdparty/miniz.c
         src/3rdparty/pugiconfig.hpp
         src/3rdparty/pugixml.hpp
         src/3rdparty/pugixml.cpp
         src/3rdparty/stb_image.h
         src/3rdparty/tinyexr.h
         src/3rdparty/tinyply.h
        src/bbox.cuh
        src/bvh.cuh
        src/camera.cuh
        src/compute_normals.cuh
        src/flexception.h
        src/hw1.cuh
        #src/hw1_scenes.cuh
        #src/hw2.cuh
        #src/hw3.cuh
        #src/hw4.cuh
        src/image.cuh
        src/intersection.cuh
        src/ray.cuh
        src/material.cuh
        src/matrix.cuh
        src/parallel.cuh
        src/parse_obj.cuh
        src/parse_ply.cuh
        src/parse_scene.cuh
        src/parse_serialized.cuh
        #src/pcg.cuh
        src/print_scene.cuh
        src/shape.cuh
        src/sphere.cuh
        src/texture.cuh
        #src/perlin.cuh
        src/torrey.cuh
        src/transform.cuh
        src/vector.cuh
        src/bvh.cu
        src/camera.cu
        src/compute_normals.cu
        src/hw1.cu
        src/hw2.cu
        #src/hw3.cu
        #src/hw4.cu
        src/image.cu
        src/main.cu
        src/parallel.cu
        src/parse_obj.cu
        src/parse_ply.cu
        src/parse_scene.cu
        src/parse_serialized.cu
        src/print_scene.cu
        src/scene.cu
        src/shape.cu
        src/texture.cu
        src/transform.cu
        )

add_library(torrey_lib STATIC ${SRCS})
add_executable(torrey src/main.cu)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(torrey Threads::Threads)

target_link_libraries(torrey torrey_lib)

target_link_libraries(torrey ${CUDA_RUNTIME_LIBRARY})
