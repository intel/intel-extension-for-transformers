FILE(GLOB src main_block.cpp)
add_unit_test(global_load_store_block kernel_func.hpp "${src}")

FILE(GLOB src main_scatter.cpp)
add_unit_test(global_load_store_scatter kernel_func.hpp "${src}")
