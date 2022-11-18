cmake_minimum_required(VERSION 3.16)

project(NeuralEngine)

set(NE_ROOT "${PROJECT_SOURCE_DIR}")
set(NE_THIRD_PARTY_DIR "${NE_ROOT}/third_party")
set(NE_CMAKE_DIR "${NE_ROOT}/cmake")
list(APPEND CMAKE_MODULE_PATH ${NE_CMAKE_DIR})

option(NE_WITH_SPARSELIB "Enable sparselib with sparse gemm ops" OFF)
option(NE_WITH_SPARSELIB_ONLY "Only for sparselib" OFF)
option(NE_WITH_SPARSELIB_BENCHMARK "Enable sparselib sparse benchmark" OFF)
option(NE_WITH_TESTS "Build GTests" OFF)
option(NE_WITH_ONEDNN_GRAPH "Build oneDNNGraph" ON)

if(NE_WITH_SPARSELIB_ONLY AND NOT NE_WITH_SPARSELIB)
  message(FATAL_ERROR "SPAESELIB is off")
endif()

if(UNIX)
    set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -ffast-math -lm -ftree-loop-vectorize -fpic -Wall -fopenmp -march=native -Wl,-rpath,'$ORIGIN'")
else()
    include(PresetOs)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

if(NOT NE_WITH_SPARSELIB_ONLY)
  set(GFLAGS_USE_TARGET_NAMESPACE TRUE)
  set(GFLAGS_BUILD_SHARED_LIBS OFF)
  set(GFLAGS_BUILD_TESTING OFF)
  add_subdirectory(${NE_THIRD_PARTY_DIR}/gflags)
  set(WITH_GFLAGS OFF CACHE BOOL "disable gflags for glog")
endif()

set(BUILD_TESTING OFF)
set(BUILD_SHARED_LIBS ON)
add_subdirectory(${NE_THIRD_PARTY_DIR}/glog)

if(NOT NE_WITH_SPARSELIB_ONLY)
  set(DNNL_LIBRARY_TYPE SHARED)
  set(DNNL_GRAPH_LIBRARY_TYPE SHARED)
  if(NE_WITH_ONEDNN_GRAPH)
    add_compile_definitions(WITH_ONEDNN_GRAPH)
    add_subdirectory(${NE_THIRD_PARTY_DIR}/oneDNNGraph)
  else()
    add_subdirectory(${NE_THIRD_PARTY_DIR}/oneDNN)
  endif()
  add_subdirectory(${NE_THIRD_PARTY_DIR}/pybind11)
endif()

if(NOT NE_WITH_SPARSELIB_ONLY)
  set(BUILD_SHARED_LIBS OFF)
  set(YAML_CPP_BUILD_TOOLS OFF)
  set(YAML_BUILD_SHARED_LIBS OFF)
  if(NOT YAML_CPP_BUILD_TOOLS)
    add_compile_definitions(YAML_CPP_STATIC_DEFINE)
  endif()
  add_subdirectory(${NE_THIRD_PARTY_DIR}/yaml-cpp)
endif()

if(NE_WITH_SPARSELIB)
  add_subdirectory(${NE_THIRD_PARTY_DIR}/xbyak)
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
find_package(OpenMP)
if(OpenMP_FOUND)
    message("OpenMP found: ${OpenMP_CXX_FLAGS}")
    if(WIN32)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp:experimental")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
else()
    message(FATAL_ERROR "Cannot find OpenMP")
endif()


set(BOOST_INC_DIRS
        ${NE_THIRD_PARTY_DIR}/boost/libs/assert/include
        ${NE_THIRD_PARTY_DIR}/boost/libs/core/include
        ${NE_THIRD_PARTY_DIR}/boost/libs/move/include
        ${NE_THIRD_PARTY_DIR}/boost/libs/config/include
        ${NE_THIRD_PARTY_DIR}/boost/libs/container/include
        ${NE_THIRD_PARTY_DIR}/boost/libs/intrusive/include
        ${NE_THIRD_PARTY_DIR}/boost/libs/static_assert/include
        ${NE_THIRD_PARTY_DIR}/boost/libs/type_traits/include
        ${NE_THIRD_PARTY_DIR}/boost/libs/interprocess/include
        ${NE_THIRD_PARTY_DIR}/boost/libs/unordered/include
        ${NE_THIRD_PARTY_DIR}/boost/libs/container_hash/include
        ${NE_THIRD_PARTY_DIR}/boost/libs/preprocessor/include
        ${NE_THIRD_PARTY_DIR}/boost/libs/throw_exception/include
        ${NE_THIRD_PARTY_DIR}/boost/libs/tuple/include
        ${NE_THIRD_PARTY_DIR}/boost/libs/predef/include
        ${NE_THIRD_PARTY_DIR}/boost/libs/mp11/include
)

if(WIN32)
  list(APPEND BOOST_INC_DIRS ${NE_THIRD_PARTY_DIR}/boost/libs/winapi/include)
endif()

if(NE_WITH_SPARSELIB)
  add_subdirectory(SparseLib)
  add_compile_definitions(WITH_SPARSELIB)
endif()


if(NOT NE_WITH_SPARSELIB_ONLY)
  add_subdirectory(executor)
endif()

if(NE_WITH_TESTS)
  enable_testing()
  add_subdirectory(${NE_THIRD_PARTY_DIR}/googletest)
  if(NOT NE_WITH_SPARSELIB_ONLY)
    add_subdirectory(test/gtest)
  else()
    add_subdirectory(test/gtest/SparseLib)
  endif()
endif()

if(NE_WITH_SPARSELIB_BENCHMARK)
  add_subdirectory(test/SparseLib/benchmark)
endif()
