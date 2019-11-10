cmake_minimum_required(VERSION 3.8)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(SummerShuffle LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
include_directories(/usr/local/cuda/include)

add_executable(main main.cu ${HEADER_FILES})
target_include_directories(main PRIVATE /usr/local/cuda/include)