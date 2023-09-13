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
project(jblas_dispatcher LANGUAGES C CXX)
set(CMAKE_CXX_FLAGS "-fPIC -Wno-narrowing -fconcepts")
file(GLOB SOURCES 
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)
file(GLOB HEADERS 
    ${PROJECT_SOURCE_DIR}/include/*.hpp
)

add_subdirectory(../../../library/jblas jblas_out)

add_library(jblas_dispatcher STATIC ${HEADERS} ${SOURCES})

set_target_properties(jblas_dispatcher PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(jblas_dispatcher "${TORCH_LIBRARIES}" jblas::jblas)
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
