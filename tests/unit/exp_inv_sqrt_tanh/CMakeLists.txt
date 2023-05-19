FILE(GLOB src main.cpp)
add_unit_test(exp_inv_sqrt_tanh kernel_func.hpp ${src})
