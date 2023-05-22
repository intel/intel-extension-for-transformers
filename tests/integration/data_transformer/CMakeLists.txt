get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" ProjectId ${ProjectId})

FILE(GLOB src main.cpp)
add_integration_test(${ProjectId} "${src}")
