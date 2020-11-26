cmake_minimum_required(VERSION 3.10)
project(CUDAShuffle LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/vendor/nvidia/cub)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} --std=c++17 )
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} --std=c++11 )
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_60,code=sm_60")

add_executable(main main.cu)
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
enable_testing()

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
add_subdirectory(vendor/google/benchmark)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(vendor/google/googletest/googletest)

add_subdirectory(mergeshuffle)
add_subdirectory(test)
add_subdirectory(benchmarks)
