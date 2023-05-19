FILE(GLOB src main.cpp)
add_unit_test(tile_load_store_local kernel_func.hpp ${src})
