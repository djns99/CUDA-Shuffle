function(create_test test_name)
	add_executable(${test_name} ${test_name}.cc)
	target_link_libraries(${test_name} gtest_main)
	target_compile_options(${test_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
	add_test(NAME ${test_name} COMMAND ${test_name})
endfunction(create_test)

create_test(FunctionalTests)

