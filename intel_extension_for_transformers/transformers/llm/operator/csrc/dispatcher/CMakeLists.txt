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
project(bestla_dispatcher LANGUAGES C CXX)
if(NOT WIN32)
set(CMAKE_CXX_FLAGS "-fPIC -fconcepts")
endif()
file(GLOB SOURCES 
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)
file(GLOB HEADERS 
    ${PROJECT_SOURCE_DIR}/include/*.hpp
)

include(FetchContent)
include(neural_speed.cmake)

add_library(bestla_dispatcher STATIC ${HEADERS} ${SOURCES})
if(WIN32)
# MSVC does not allow sth like -fconcepts
set_target_properties(bestla_dispatcher PROPERTIES C_STANDARD 20 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
set_target_properties(bestla_dispatcher PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
endif()

set_target_properties(bestla_dispatcher PROPERTIES POSITION_INDEPENDENTBTLA_CODE ON)
set_target_properties(bestla_dispatcher PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(bestla_dispatcher OpenMP::OpenMP_CXX OpenMP::OpenMP_C "${TORCH_LIBRARIES}" torch_python bestla::bestla)
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
