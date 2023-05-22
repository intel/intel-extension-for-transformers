FILE(GLOB src main.cpp)
add_unit_test(block_load_store kernel_func.hpp ${src})
