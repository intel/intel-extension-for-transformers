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

option(SPARSE_LIB_USE_VTUNE OFF)
option(CMAKE_VTUNE_HOME "whether use vtune" OFF)

set(EXTERNAL_INCLUDE_PATH "")
set(EXTERNAL_LIBRARY_PAYH "")

# include external cmake
if (SPARSE_LIB_USE_VTUNE)
    include(cmake/FindVTune.cmake)
    message( STATUS "VTune enbled" )
endif()

file(GLOB SOURCES
  ${PROJECT_SOURCE_DIR}/src/*.cpp
  ${PROJECT_SOURCE_DIR}/src/kernels/*.cpp
  ${PROJECT_SOURCE_DIR}/src/jit_domain/*.cpp
)
file(GLOB HEADERS
  ${PROJECT_SOURCE_DIR}/include/*.hpp
  ${PROJECT_SOURCE_DIR}/include/kernels/*.hpp
  ${PROJECT_SOURCE_DIR}/include/jit_domain/*.hpp
)

if (NE_DYNAMIC_LINK)
    add_library(${HOST_LIBRARY_NAME} SHARED ${SOURCES} ${HEADERS})
else()
    add_library(${HOST_LIBRARY_NAME} STATIC ${SOURCES} ${HEADERS})
endif()

install(TARGETS ${HOST_LIBRARY_NAME})

target_compile_definitions(${HOST_LIBRARY_NAME} PRIVATE SPARSE_KERNEL_BUILD)
include(cmake/Common.cmake)
include(cmake/FindOpenMP.cmake)
include(cmake/Utility.cmake)

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
        ${PROJECT_SOURCE_DIR}/include
)

## link against the third party libraries
target_link_libraries(${HOST_LIBRARY_NAME}
    PUBLIC
        ${CMAKE_THREAD_LIBS_INIT}
        glog
        xbyak
)


