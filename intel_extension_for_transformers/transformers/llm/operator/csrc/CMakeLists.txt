##  Copyright (c) 2023 Intel Corporation
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(qbits LANGUAGES C CXX)


set(QBITS_TORCH_PATH "" CACHE STRING "Torch install path")

if(QBITS_TORCH_PATH)
    set(torch_path ${QBITS_TORCH_PATH})
    unset(TORCH_LIBRARY CACHE)  # force find_package torch
    unset(c10_LIBRARY CACHE)
    unset(TORCH_DIR CACHE)
else()
    execute_process(COMMAND python -c "import torch; print(torch.__path__[0])"
                    OUTPUT_VARIABLE torch_path
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

find_package(Torch REQUIRED
PATHS ${torch_path}
NO_DEFAULT_PATH)
find_package(PythonLibs 3 REQUIRED)

include(FindOpenMP)
add_subdirectory(dispatcher)
add_subdirectory(../../../runtime/third_party/pybind11 pybind11)

file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)
file(GLOB qbits_src ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# Link against LibTorch
pybind11_add_module(qbits ${qbits_src})
target_compile_features(qbits PRIVATE cxx_std_14)
target_link_directories(qbits PRIVATE ${torch_path}/lib)
target_link_libraries(qbits PRIVATE bestla_dispatcher torch_python)
