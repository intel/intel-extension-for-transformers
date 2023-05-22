FILE(GLOB src main.cpp)
add_unit_test(tile_row_reduction kernel_func.hpp ${src})
