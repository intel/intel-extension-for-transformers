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
project(qbits_py LANGUAGES C CXX)


set(QBITS_TORCH_PATH "" CACHE STRING "Torch install path")
set(torch_info "")

function(get_torch_info python_command)
    set(import_torch "import torch:")
    string(REPLACE ":" ";" import_torch ${import_torch})
    string(CONCAT fin_command "${import_torch}" "${python_command}")
    execute_process(COMMAND python -c "${fin_command}"
                    OUTPUT_VARIABLE torch_info
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(torch_info "${torch_info}" PARENT_SCOPE)
endfunction()


if(QBITS_TORCH_PATH)
    set(torch_path ${QBITS_TORCH_PATH})
    unset(TORCH_LIBRARY CACHE)  # force find_package torch
    unset(c10_LIBRARY CACHE)
    unset(TORCH_DIR CACHE)
else()
    get_torch_info("print(torch.__path__[0])")
    set(torch_path "${torch_info}")
endif()

find_package(Torch REQUIRED
PATHS ${torch_path}
NO_DEFAULT_PATH)

if(NOT WIN32)
find_package(PythonLibs 3 REQUIRED)
endif()

include(FindOpenMP)
set(BTLA_ENABLE_OPENMP ON CACHE BOOL "BesTLA enable compiling OpenMP threading")
add_subdirectory(dispatcher)
add_subdirectory(../transformers/runtime/third_party/pybind11 pybind11)

file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)
file(GLOB qbits_src ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_compile_options(-flto=auto)

# Link against LibTorch
pybind11_add_module(qbits_py ${qbits_src})
get_torch_info("print(torch.__version__)")
set(torch_version "${torch_info}")
target_compile_features(qbits_py PRIVATE cxx_std_14)
set(TORCH_VERSION_MACRO COMPATIBLE_TORCH_VERSION="${torch_version}")
target_compile_definitions(qbits_py PUBLIC ${TORCH_VERSION_MACRO})
target_link_directories(qbits_py PRIVATE ${torch_path}/lib)
target_link_libraries(qbits_py PRIVATE bestla_dispatcher torch_python)
