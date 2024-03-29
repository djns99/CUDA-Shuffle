cmake_minimum_required(VERSION 3.15)

project(CUDAShuffle LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} --std=c++17)
include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/vendor/moderngpu/src)
include_directories(${CMAKE_SOURCE_DIR})

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86")

enable_testing()

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
add_subdirectory(vendor/google/benchmark)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(vendor/google/googletest/googletest)
add_subdirectory(test)
add_subdirectory(benchmarks)
