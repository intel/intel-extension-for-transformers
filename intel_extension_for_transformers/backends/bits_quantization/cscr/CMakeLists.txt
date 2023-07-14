cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(weight_only_jblasop LANGUAGES C CXX)

set(CMAKE_CXX_FLAGS "-g -O0 -Wno-narrowing")

# execute_process(COMMAND python -c "import torch; print(torch.__path__[0])"
#                 OUTPUT_VARIABLE torch_path
#                 OUTPUT_STRIP_TRAILING_WHITESPACE)

find_package(Torch REQUIRED
# PATHS ${torch_path}
PATHS /home/zhe/miniconda3/envs/s4fp16gemm/lib/python3.9/site-packages/torch
NO_DEFAULT_PATH)


add_subdirectory(../../neural_engine/graph/jblas jblas.out)

file(GLOB custom_jblasop_src ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# Define our library target
add_library(weight_only_jblasop SHARED ${custom_jblasop_src})
# Enable C++14
target_compile_features(weight_only_jblasop PRIVATE cxx_std_14)

# Link against LibTorch
target_link_libraries(weight_only_jblasop "${TORCH_LIBRARIES}" jblas::jblas)

set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
