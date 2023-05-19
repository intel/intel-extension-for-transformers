FILE(GLOB src main.cpp)
add_unit_test(global_atomic kernel_func.hpp ${src})
