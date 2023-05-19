#*******************************************************************************
# Copyright (c) 2022-2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#*******************************************************************************/

# The following are set after configuration is done:
#  ONEMKL_FOUND            : set to true if oneMKL is found.
#  ONEMKL_INCLUDE_DIR      : path to oneMKL include dir.
#  ONEMKL_GPU_LIBS         : list of oneMKL libraries for GPU
#===============================================================================

if(ONEMKL_FOUND)
  return()
endif()

set(ONEMKL_FOUND OFF)
set(ONEMKL_INCLUDE_DIR)
set(ONEMKL_GPU_LIBS)

set(mkl_root_hint)
if(DEFINED ENV{MKL_DPCPP_ROOT})
  set(mkl_root_hint $ENV{MKL_DPCPP_ROOT})
elseif(DEFINED ENV{MKLROOT})
  set(mkl_root_hint $ENV{MKLROOT})
elseif(DEFINED ENV{MKL_ROOT})
  set(mkl_root_hint $ENV{MKL_ROOT})
elseif(MKL_ROOT)
  set(mkl_root_hint ${MKL_ROOT})
else()
  message(FATAL_ERROR "Please set oneMKL root path by MKL_DPCPP_ROOT, or MKLROOT, or MKL_ROOT.")
endif()

# Try to find Intel MKL DPCPP header
find_file(MKL_HEADER NAMES mkl.h PATHS ${mkl_root_hint}
    PATH_SUFFIXES include NO_DEFAULT_PATH)

if(NOT MKL_HEADER)
  message(FATAL_ERROR "Intel oneMKL not found. No oneMKL support ${MKL_HEADER} -- ${mkl_root_hint}")
endif()
get_filename_component(ONEMKL_INCLUDE_DIR "${MKL_HEADER}/.." ABSOLUTE)

set(LIB_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
set(LIB_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})

set(MKL_THREAD "${LIB_PREFIX}mkl_gnu_thread${LIB_SUFFIX}")
find_library(MKL_LIB_THREAD ${MKL_THREAD} HINTS ${mkl_root_hint}
    PATH_SUFFIXES lib lib/intel64 NO_DEFAULT_PATH)
if(NOT MKL_LIB_THREAD)
  message(FATAL_ERROR "oneMKL library ${MKL_THREAD} not found")
endif()

set(MKL_LP64 "${LIB_PREFIX}mkl_intel_lp64${LIB_SUFFIX}")
find_library(MKL_LIB_LP64 ${MKL_LP64} HINTS ${mkl_root_hint}
    PATH_SUFFIXES lib lib/intel64 NO_DEFAULT_PATH)
if(NOT MKL_LIB_LP64)
  message(FATAL_ERROR "oneMKL library ${MKL_LP64} not found")
endif()

set(MKL_CORE "${LIB_PREFIX}mkl_core${LIB_SUFFIX}")
find_library(MKL_LIB_CORE ${MKL_CORE} HINTS ${mkl_root_hint}
    PATH_SUFFIXES lib lib/intel64 NO_DEFAULT_PATH)
if(NOT MKL_LIB_CORE)
  message(FATAL_ERROR "oneMKL library ${MKL_CORE} not found")
endif()

set(MKL_SYCL "${LIB_PREFIX}mkl_sycl${LIB_SUFFIX}")
find_library(MKL_LIB_SYCL ${MKL_SYCL} HINTS ${mkl_root_hint}
    PATH_SUFFIXES lib lib/intel64 NO_DEFAULT_PATH)
if(NOT MKL_LIB_SYCL)
  message(FATAL_ERROR "oneMKL library ${MKL_SYCL} not found")
endif()

set(START_GROUP)
set(END_GROUP)

set(ONEMKL_GPU_LIBS ${START_GROUP} ${MKL_LIB_LP64} ${MKL_LIB_CORE} ${MKL_LIB_THREAD} ${MKL_LIB_SYCL} ${END_GROUP})

set(ONEMKL_FOUND ON)
message("-- Detecting MKL features - done")

