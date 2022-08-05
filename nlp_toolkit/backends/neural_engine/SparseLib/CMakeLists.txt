cmake_minimum_required(VERSION 3.11)
project(kernellibs)
set(HOST_LIBRARY_NAME "kernellibs")

option(SPARSE_LIB_USE_AMX OFF)

set(CMAKE_CXX_FLAGS "-std=c++11 -fpic -Wall -fopenmp -march=native")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "-g ${CMAKE_CXX_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
endif()

if (SPARSE_LIB_USE_AMX)
    set(CMAKE_CXX_FLAGS "-DSPARSE_LIB_USE_AMX ${CMAKE_CXX_FLAGS}")
endif()

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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(${CMAKE_CXX_FLAGS})
else()
    message(STATUS "Cannot find OpenMP")
endif()

set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/third_party/")

set(WITH_GFLAGS OFF CACHE BOOL "disable gflags for glog")
set(WITH_GTEST OFF CACHE BOOL "disable gtest for glog")
add_subdirectory(${THIRD_PARTY_DIR}/glog)
add_subdirectory(${THIRD_PARTY_DIR}/xbyak)

add_subdirectory(src)

