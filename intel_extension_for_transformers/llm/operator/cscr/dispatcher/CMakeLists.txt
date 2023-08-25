cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(jblas_dispatcher LANGUAGES C CXX)
set(CMAKE_CXX_FLAGS "-O3 -fPIC -Wno-narrowing")
file(GLOB SOURCES 
    ${PROJECT_SOURCE_DIR}/*.cpp
)
file(GLOB HEADERS 
    ${PROJECT_SOURCE_DIR}/*.hpp
)

add_subdirectory(../../../library/jblas jblas_out)

add_library(jblas_dispatcher STATIC ${HEADERS} ${SOURCES})

set_target_properties(jblas_dispatcher PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(jblas_dispatcher "${TORCH_LIBRARIES}" jblas::jblas)
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
