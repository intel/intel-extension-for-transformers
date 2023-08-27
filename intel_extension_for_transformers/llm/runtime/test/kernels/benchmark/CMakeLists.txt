# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.11)

project(sparselib_benchmark)

set(BENCHMARK_EXE "benchmark")

if(UNIX)
  # Enable warning
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-missing-field-initializers")
endif()

if(OpenMP_FOUND)
    #message(${CMAKE_CXX_FLAGS})
else()
    find_package(OpenMP REQUIRED)
    if(WIN32)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp:experimental")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()


file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_subdirectory(sparse_matmul)
add_subdirectory(eltwiseop)
add_subdirectory(layernorm_ba)
add_subdirectory(transpose_matmul)
add_subdirectory(softmax)
add_subdirectory(attention)
add_subdirectory(transpose_mha)
add_subdirectory(mha_dense)
add_subdirectory(dynamic_quant_matmul)
add_subdirectory(dynamic_quant)

add_executable(${BENCHMARK_EXE} ${SRCS})

target_include_directories(${BENCHMARK_EXE}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${BENCHMARK_EXE}
    PRIVATE
        ${CMAKE_THREAD_LIBS_INIT}
        kernellibs
)

file(COPY ci DESTINATION "${CMAKE_BINARY_DIR}/bin")
