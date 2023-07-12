cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(weight_only_jblasop LANGUAGES C CXX)

find_package(Torch REQUIRED)

add_subdirectory(../../jblas jblas.out)

file(GLOB custom_jblasop_src ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# Define our library target
add_library(weight_only_jblasop SHARED ${custom_jblasop_src})
# Enable C++14
target_compile_features(weight_only_jblasop PRIVATE cxx_std_14)

# Link against LibTorch
target_link_libraries(weight_only_jblasop "${TORCH_LIBRARIES}" jblas::jblas)

set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
