FILE(GLOB src main.cpp)
add_unit_test(tile_mma kernel_func.hpp ${src})
