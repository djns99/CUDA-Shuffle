project(mergeshuffle CXX ASM)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP REQUIRED)

set_source_files_properties(merge.s PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
add_library( MergeShuffle STATIC src/merge_omp.cc src/random.cc)
target_link_libraries( MergeShuffle OpenMP::OpenMP_CXX )
target_compile_options( MergeShuffle PRIVATE -mavx )

add_library( ParallelRaoSandelius STATIC src/split_par.cc src/random.cc)
target_compile_options( ParallelRaoSandelius PRIVATE -mavx)
