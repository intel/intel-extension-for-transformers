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

project(xetla_kernel LANGUAGES C CXX)

add_compile_options(-fsycl -fPIC -fp-model=precise)
add_link_options(-fsycl -lpthread -lm -fPIC)

file(GLOB xetla_kernel_src xetla_kernel.cpp)

# Define our library target
add_library(xetla_kernel STATIC ${xetla_kernel_src})

# Xetla Enable C++20
set_property(TARGET xetla_kernel PROPERTY CXX_STANDARD 20)

target_include_directories(xetla_kernel PRIVATE
../../../library/xetla
../../../library/xetla/include
)
