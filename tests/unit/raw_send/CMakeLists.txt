FILE(GLOB src main.cpp)
add_unit_test(raw_send kernel_func.hpp ${src})
