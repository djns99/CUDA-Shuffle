function( setup_benchmark name )
    add_executable(${name} ${name}.cu)
    message("Include dir set to ${GBENCH_SOURCE_DIR}")
    target_link_libraries(${name} PRIVATE benchmark)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

endfunction()

setup_benchmark(ShuffleBench)