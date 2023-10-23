FILE(GLOB src main.cpp)
add_unit_test(reg_layout_conversion kernel_func.hpp ${src})
