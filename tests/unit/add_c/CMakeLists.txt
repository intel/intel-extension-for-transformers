FILE(GLOB src main.cpp)
add_unit_test(add_c kernel_func.hpp "${src}")
