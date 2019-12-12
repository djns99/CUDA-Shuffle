cmake_minimum_required(VERSION 3.10)
project(SummerShuffle LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_SOURCE_DIR}/include)

set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} )

#add_executable(main main.cu)
#target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
enable_testing()

set(OLD_BUILD_TYPE ${CMAKE_BUILD_TYPE})
set(CMAKE_BUILD_TYPE Release)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
add_subdirectory(vendor/google/benchmark)
add_subdirectory(vendor/google/googletest/googletest)
set(CMAKE_BUILD_TYPE ${OLD_BUILD_TYPE})

add_subdirectory(test)
add_subdirectory(benchmarks)
