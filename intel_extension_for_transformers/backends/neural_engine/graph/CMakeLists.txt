#  Copyright (c) 2023 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

cmake_minimum_required(VERSION 3.12) # Don't bump this version for no reason
project("ne_graph" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

#
# Option list
#

# general
option(NE_STATIC                 "neural_engine: static link libraries"                          OFF)
option(NE_NATIVE                 "neural_engine: enable -march=native flag"                      OFF)
option(NE_LTO                    "neural_engine: enable link time optimization"                  OFF)
option(NE_BUILD_APPLICATIONS     "neural_engine: build applications"                             ON)

# GPU
option(NE_GPU                    "neural_engine: enable GPU inference"                           OFF)

# debug
option(NE_ALL_WARNINGS           "neural_engine: enable all compiler warnings"                   ON)
option(NE_ALL_WARNINGS_3RD_PARTY "neural_engine: enable all compiler warnings in 3rd party libs" OFF)
option(NE_GPROF                  "neural_engine: enable gprof"                                   OFF)

# sanitizers
option(NE_SANITIZE_THREAD        "neural_engine: enable thread sanitizer"                        OFF)
option(NE_SANITIZE_ADDRESS       "neural_engine: enable address sanitizer"                       OFF)
option(NE_SANITIZE_UNDEFINED     "neural_engine: enable undefined sanitizer"                     OFF)

# instruction set specific
option(NE_AVX                    "neural_engine: enable AVX"                                     ON)
option(NE_AVX2                   "neural_engine: enable AVX2"                                    ON)
option(NE_AVX512                 "neural_engine: enable AVX512"                                  ON)
option(NE_AVX512_VBMI            "neural_engine: enable AVX512-VBMI"                             ON)
option(NE_AVX512_VNNI            "neural_engine: enable AVX512-VNNI"                             ON)
option(NE_FMA                    "neural_engine: enable FMA"                                     ON)
option(NE_AMX                    "neural_engine: enable AMX"                                     OFF)

# in MSVC F16C is implied with AVX2/AVX512
if (NOT MSVC)
    option(NE_F16C               "neural_engine: enable F16C"                                    ON)
endif()

# 3rd party libs
option(NE_ONEDNN                 "neural_engine: use oneDNN"                                     ON)
option(NE_LIBXSMM                "neural_engine: use libxsmm"                                    OFF)
option(NE_XETLA                  "neural_engine: use XeTLA"                                      OFF)
if (NE_GPU)
  set(NE_XETLA ON)
endif()

option(NE_BUILD_TESTS            "neural_engine: build tests"    ${NE_STANDALONE})
option(NE_BUILD_EXAMPLES         "neural_engine: build examples" ${NE_STANDALONE})

option(NE_PROFILING              "neural_engine: use Profiling"                                  OFF)
if (NE_PROFILING)
    add_compile_definitions(NE_PERF)
endif()

if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

if (NE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

if (NOT MSVC)
    if (NE_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (NE_GPROF)
        add_compile_options(-pg)
    endif()
    if (NE_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

include(cmake/ISA.cmake)
include(cmake/Common.cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
)

if(NE_GPU)
  list(APPEND COMMON_HEADER_DIRS ${GPU_ROOT}/include)
  list(APPEND COMMON_LIB_DIRS ${GPU_ROOT})
endif()

include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)


add_subdirectory(core)
add_subdirectory(jblas)
add_subdirectory(vectors)
add_subdirectory(models)

if (NE_BUILD_APPLICATIONS)
    add_subdirectory(application)
endif()
