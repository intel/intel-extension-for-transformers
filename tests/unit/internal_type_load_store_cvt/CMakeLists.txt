FILE(GLOB src main.cpp)
add_unit_test(internal_type_load_store_cvt kernel_func.hpp ${src})
