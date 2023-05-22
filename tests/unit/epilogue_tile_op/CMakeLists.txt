FILE(GLOB src main.cpp)
add_unit_test(epilogue_tile_op kernel_func.hpp ${src})
