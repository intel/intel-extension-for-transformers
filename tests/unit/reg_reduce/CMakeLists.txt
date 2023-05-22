FILE(GLOB src main.cpp)
add_unit_test(reg_reduce kernel_func.hpp ${src})
