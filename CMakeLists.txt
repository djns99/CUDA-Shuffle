cmake_minimum_required(VERSION 3.8)
project(SummerShuffle LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} )

add_executable(main main.cu)
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)