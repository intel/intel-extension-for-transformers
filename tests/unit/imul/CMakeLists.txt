FILE(GLOB src main.cpp)
add_unit_test(imul kernel_func.hpp ${src})
