##  Copyright (c) 2022 Intel Corporation
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

cmake_minimum_required(VERSION 3.11)
project(kernellibs)
set(HOST_LIBRARY_NAME "kernellibs")

option(NE_WITH_SPARSELIB_VTUNE "Enable VTune profiling in kernels" OFF)
option(NE_WITH_SPARSELIB_GPU "Enable GPU" OFF)
set(CMAKE_VTUNE_HOME)

set(EXTERNAL_INCLUDE_PATH "")
set(EXTERNAL_LIBRARY_PAYH "")

file(GLOB SOURCES
  ${PROJECT_SOURCE_DIR}/src/*.cpp
  ${PROJECT_SOURCE_DIR}/src/data_type/*.cpp
  ${PROJECT_SOURCE_DIR}/src/cpu/*.cpp
  ${PROJECT_SOURCE_DIR}/src/cpu/engine/*.cpp
  ${PROJECT_SOURCE_DIR}/src/cpu/memory_storege/*.cpp
  ${PROJECT_SOURCE_DIR}/src/cpu/jit_domain/*.cpp
  ${PROJECT_SOURCE_DIR}/src/cpu/kernels/*.cpp
)

file(GLOB_RECURSE HEADERS
  ${PROJECT_SOURCE_DIR}/*.hpp
)

file(GLOB GPU_SOURCES
  ${PROJECT_SOURCE_DIR}/src/gpu/*.cpp
  ${PROJECT_SOURCE_DIR}/src/gpu/engine/*.cpp
  ${PROJECT_SOURCE_DIR}/src/gpu/memory_storage/*.cpp
  ${PROJECT_SOURCE_DIR}/src/gpu/stream/*.cpp
  ${PROJECT_SOURCE_DIR}/src/gpu/kernels/*.cpp
  ${PROJECT_SOURCE_DIR}/src/gpu/kernels/opencl/*.cpp
  ${PROJECT_SOURCE_DIR}/src/gpu/kernels/opencl/gemm/*.cpp
)

if (NE_WITH_SPARSELIB_GPU)
  message( STATUS "GPU enbled" )
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-operator-names")
  find_package(OpenCL REQUIRED)
  set(SOURCES ${SOURCES} ${GPU_SOURCES})
  add_definitions(-DSPARSELIB_GPU)
endif()
if (NE_DYNAMIC_LINK)
    add_library(${HOST_LIBRARY_NAME} SHARED ${SOURCES} ${HEADERS})
    target_compile_definitions(${HOST_LIBRARY_NAME} PRIVATE SPARSE_KERNEL_BUILD)
    if (NE_WITH_TESTS OR NE_WITH_SPARSELIB_BENCHMARK)
      target_compile_definitions(${HOST_LIBRARY_NAME} PRIVATE SPARSE_TEST)
    endif()
else()
    add_library(${HOST_LIBRARY_NAME} STATIC ${SOURCES} ${HEADERS})
    if (WIN32)
        target_compile_definitions(${HOST_LIBRARY_NAME} PRIVATE SPARSE_API_)
    endif()
endif()

install(TARGETS ${HOST_LIBRARY_NAME})

target_compile_definitions(${HOST_LIBRARY_NAME} PRIVATE SPARSE_KERNEL_BUILD)
include(cmake/Common.cmake)
include(cmake/FindOpenMP.cmake)
include(cmake/Utility.cmake)
# include external cmake
if (NE_WITH_SPARSELIB_VTUNE)
    include(cmake/FindVTune.cmake)
    message( STATUS "VTune enbled" )
endif()

# TODO(Yi): enable/disable kernel(s) with VNNI intrin given compile option(s)
if(WIN32)
    if(NE_WITH_AVX2)
    target_compile_options(${HOST_LIBRARY_NAME} PRIVATE /arch:AVX2)
    else()
    target_compile_options(${HOST_LIBRARY_NAME} PRIVATE /arch:AVX512)
    endif()
else()
    if(NE_WITH_AVX2)
    target_compile_options(${HOST_LIBRARY_NAME} PRIVATE -mavx2)
    else()
    target_compile_options(${HOST_LIBRARY_NAME} PRIVATE -march=icelake-server)
    endif()
endif()

set(GIT_HASH "unknown")
get_git_hash(GIT_HASH)
message(STATUS "Git hash is ${GIT_HASH}")

set(GIT_BRANCH "unknown")
get_git_branch(GIT_BRANCH)
message(STATUS "Git branch is ${GIT_BRANCH}")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/git_version.h.in
  ${CMAKE_BINARY_DIR}/git_version.h
  @ONLY
  )

set_target_properties(${HOST_LIBRARY_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Header file include path
message( STATUS "EXTERNAL_INCLUDE_PAYH ${EXTERNAL_INCLUDE_PAYH}" )
target_include_directories(${HOST_LIBRARY_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
)

## link against the third party libraries
target_link_libraries(${HOST_LIBRARY_NAME}
    PUBLIC
        ${CMAKE_THREAD_LIBS_INIT}
        ${OpenCL_LIBRARY}
        # OpenCL::OpenCL
        glog
        xbyak
)
