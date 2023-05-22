FILE(GLOB src main.cpp)
add_unit_test(philox_rng kernel_func.hpp ${src})
