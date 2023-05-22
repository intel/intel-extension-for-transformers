FILE(GLOB src main.cpp)
add_unit_test(named_barrier kernel_func.hpp ${src})
