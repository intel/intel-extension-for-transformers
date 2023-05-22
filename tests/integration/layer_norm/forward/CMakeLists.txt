get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" ProjectId ${ProjectId})
string(PREPEND ProjectId "layer_norm_")

FILE(GLOB src main.cpp)
add_integration_test(${ProjectId} ${src})
