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
set(CMAKE_C_COMPILER "icx")
set(CMAKE_CXX_COMPILER "icpx")

project(gbits LANGUAGES C CXX)

set(GBITS_TORCH_PATH "" CACHE STRING "Torch install path")
set(GBITS_IPEX_PATH "" CACHE STRING "IPEX install path")

if(GBITS_TORCH_PATH)
    set(torch_path ${GBITS_TORCH_PATH})
    unset(TORCH_LIBRARY CACHE)  # force find_package torch
    unset(c10_LIBRARY CACHE)
    unset(TORCH_DIR CACHE)
else()
    execute_process(COMMAND python -c "import torch; print(torch.__path__[0])"
                    OUTPUT_VARIABLE torch_path
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(GBITS_IPEX_PATH)
    set(ipex_path ${GBITS_IPEX_PATH})
else()
    execute_process(COMMAND python -c "import torch; import intel_extension_for_pytorch; print(intel_extension_for_pytorch.__path__[0])"
                    OUTPUT_VARIABLE ipex_path
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

find_package(Python COMPONENTS Interpreter Development)
find_package(Torch REQUIRED PATHS ${torch_path} NO_DEFAULT_PATH)
find_package(IPEX REQUIRED PATHS ${ipex_path} NO_DEFAULT_PATH)

add_compile_options(-fsycl -fPIC)
add_link_options(-fsycl -lpthread -lm -fPIC)

file(GLOB gbits_src gbits_linear_new.cpp)
file(GLOB xetla_linear xetla_linear.cpp)

add_subdirectory(../../../runtime/graph/application/third_party/pybind11 pybind11)
# Define our library target
#pybind11_add_module(${PROJECT_NAME} ${gbits_src})
add_library(xetla_linear STATIC ${xetla_linear})

# gbits DPCPP need 17
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# Xetla Enable C++20
set_property(TARGET xetla_linear PROPERTY CXX_STANDARD 20)

target_include_directories(xetla_linear PRIVATE
../../../library/xetla
../../../library/xetla/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_IPEX_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} PRIVATE xetla_linear)

target_include_directories(${PROJECT_NAME} PUBLIC "${Python_INCLUDE_DIRS}")
target_include_directories(${PROJECT_NAME} PUBLIC "${TORCH_IPEX_INCLUDE_DIRS}")
