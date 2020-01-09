project(mergeshuffle CXX ASM)

set_source_files_properties(merge.s PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
add_library( MergeShuffle STATIC src/merge_omp.cc src/merge.s )
target_link_libraries( MergeShuffle openmp )

add_library( ParallelRaoSandelius STATIC src/split_par.cc )
