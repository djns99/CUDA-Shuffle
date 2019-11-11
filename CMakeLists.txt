cmake_minimum_required(VERSION 3.8)
project(SummerShuffle LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(main main.cu)
