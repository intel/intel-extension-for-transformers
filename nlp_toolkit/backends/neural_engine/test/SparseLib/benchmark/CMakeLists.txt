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

set (BENCHMARK_EXE "benchmark")


set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -fpic -Wall -fopenmp")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "-g ${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
endif()

# -lpthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)
if(Threads_FOUND)
    message("Threads found: ${CMAKE_THREAD_LIBS_INIT}")
else()
    message(STATUS "Cannot find Threads")
endif()

# -fopenmp
find_package(OpenMP REQUIRED)
message("OpenMP found: ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
message(${CMAKE_CXX_FLAGS})

add_subdirectory(${PROJECT_SOURCE_DIR}/../../../SparseLib/ ./hostlibs)

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_subdirectory(sparse_matmul)
add_subdirectory(eltwiseop)
add_subdirectory(layernorm_ba)
add_subdirectory(transpose_matmul)
add_subdirectory(softmax)

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

