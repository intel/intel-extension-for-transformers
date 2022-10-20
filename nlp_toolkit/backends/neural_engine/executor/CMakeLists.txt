project(neural_engine)
if(CMAKE_VERSION VERSION_LESS 3.12)
    message("${CMAKE_VERSION} is less then the minimum required for neural_engine build which is cmake 3.12, skip the neural_engine build")
elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.2.1)
    message("GCC version less then the minimum required for neural_engine build which is 7.2.1, skip the neural_engine build")
else()
cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -ffast-math -lm -ftree-loop-vectorize -fpic -Wall -fopenmp -march=native -Wl,-rpath,'$ORIGIN'")
set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/third_party")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Os -Wall -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Os -Wall -s")

find_library(LIBJEMALLOC jemalloc ${THIRD_PARTY_DIR}/jemalloc/lib)
if(${LIBJEMALLOC} STREQUAL LIBJEMALLOC-NOTFOUND)
  execute_process(COMMAND ./autogen.sh --disable-initial-exec-tls
                  WORKING_DIRECTORY ${THIRD_PARTY_DIR}/jemalloc)
  execute_process(COMMAND make -j
                WORKING_DIRECTORY ${THIRD_PARTY_DIR}/jemalloc)
endif()

#execute_process(COMMAND cat /proc/cpuinfo
#                COMMAND head -n 26
#                COMMAND grep -c avx512
#                OUTPUT_VARIABLE AVX_RESULT)
#if(NOT AVX_RESULT EQUAL "1")
#  message(FATAL_ERROR "AVX512 is necessary for neural_engine, please check your device")
#endif()

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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(${CMAKE_CXX_FLAGS})
else()
    message(STATUS "Cannot find OpenMP")
endif()

set(GFLAGS_USE_TARGET_NAMESPACE TRUE)
add_subdirectory(${THIRD_PARTY_DIR}/gflags)
set(WITH_GFLAGS OFF CACHE BOOL "disable gflags for glog")
add_subdirectory(${THIRD_PARTY_DIR}/oneDNNGraph)
add_subdirectory(${THIRD_PARTY_DIR}/pybind11)
add_subdirectory(${THIRD_PARTY_DIR}/yaml-cpp)
add_subdirectory(../SparseLib/ ./hostlibs)

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
list(REMOVE_ITEM SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/operators/transpose.cpp")
add_library(neural_engine SHARED ${SRC_FILES})

# Header file include path
target_include_directories(neural_engine
    PUBLIC
        ../
        ./include/sparse_operators
        ./include/llga_operators
        ./include/operators
        ./include
        ./third_party/boost/libs/assert/include
        ./third_party/boost/libs/core/include
        ./third_party/boost/libs/move/include
        ./third_party/boost/libs/config/include
        ./third_party/boost/libs/container/include
        ./third_party/boost/libs/intrusive/include
        ./third_party/boost/libs/static_assert/include
        ./third_party/boost/libs/type_traits/include
        ./third_party/boost/libs/interprocess/include
        ./third_party/boost/libs/unordered/include
        ./third_party/boost/libs/container_hash/include
        ./third_party/boost/libs/preprocessor/include
        ./third_party/boost/libs/throw_exception/include
        ./third_party/boost/libs/tuple/include
        ./third_party/boost/libs/predef/include
        ./third_party/boost/libs/mp11/include
        ./third_party/oneDNNGraph/include
)

# link against the third party libraries
target_link_libraries(neural_engine
    PUBLIC
        ${CMAKE_THREAD_LIBS_INIT}
        dnnl
        dnnl_graph
        yaml-cpp
        gflags
        glog
        rt
        kernellibs
)

## pybind11
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.9)
  cmake_policy(SET CMP0069 NEW)
endif()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
pybind11_add_module(neural_engine_py ${PROJECT_SOURCE_DIR}/python/bind_executor.cpp)
target_include_directories(neural_engine_py
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/python
)

# link against the third party libraries
target_link_libraries(neural_engine_py
    PRIVATE
        ${CMAKE_THREAD_LIBS_INIT}
        neural_engine
)

# build neural_engine_bin
add_executable(neural_engine_bin
    src/nlp_executor.cc
)

target_include_directories(neural_engine_bin
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(neural_engine_bin
    PRIVATE
        ${CMAKE_THREAD_LIBS_INIT}
        neural_engine
)
set_target_properties(neural_engine_bin
        PROPERTIES OUTPUT_NAME neural_engine)
endif()
