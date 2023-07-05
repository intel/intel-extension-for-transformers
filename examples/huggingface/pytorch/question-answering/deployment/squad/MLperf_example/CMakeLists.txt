project(inference_sut)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

if(NOT DEFINED NE_ROOT)
	if(NOT DEFINED ENV{NE_ROOT})
                set(NE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/intel_extension_for_transformers/backends/neural_engine)
	endif()
endif()
set(NE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/intel_extension_for_transformers/backends/neural_engine)
set(EXECUTOR_INCLUDE_DIRS ${NE_ROOT}/executor/include)
list(APPEND EXECUTOR_INCLUDE_DIRS ${NE_ROOT}/executor/include/llga_operators)
list(APPEND EXECUTOR_INCLUDE_DIRS ${NE_ROOT}/executor/include/operators)
set(KERNEL_INCLUDE_DIRS ${NE_ROOT}/kernels/include)
list(APPEND KERNEL_INCLUDE_DIRS ${NE_ROOT}/kernels)

if(NOT EXISTS ${NE_ROOT}/build/lib)
	message(FATAL_ERROR "please build neural engine first")
endif()

set(NE_LIBRARY_DIRS ${NE_ROOT}/build/lib)

set(LOCAL_INSTALL_DIR ${NE_ROOT}/third_party/local_install)
message("LOCAL_INSTALL_DIR: ${LOCAL_INSTALL_DIR}")

# glog
set(glog_DIR "${LOCAL_INSTALL_DIR}/glog/lib64/cmake/glog/")
find_package(glog REQUIRED)
if(glog_FOUND)
    message("glog found: ${glog_LIBRARIES}")
else()
    message(FATAL_ERROR "Cannot find glog")
endif()

# -lgflags
set(gflags_DIR "${LOCAL_INSTALL_DIR}/gflags/lib/cmake/gflags/")
find_package(gflags REQUIRED)
if(gflags_FOUND)
    message("gflags found: ${gflags_LIBRARIES}")
else()
    message(FATAL_ERROR "Cannot find gflags")
endif()

# -lyaml-cpp
set(yaml-cpp_DIR "${LOCAL_INSTALL_DIR}/yaml-cpp/share/cmake/yaml-cpp/")
find_package(yaml-cpp REQUIRED)
if(yaml-cpp_FOUND)
    message("yaml-cpp found: ${yaml-cpp_LIBRARIES}")
else()
    message(FATAL_ERROR "Cannot find yaml-cpp")
endif()

# TODO: Force IOMP, default
if(DEFINED ENV{CONDA_PREFIX})
  set(OpenMP_ROOT $ENV{CONDA_PREFIX})
else()
  set(OpenMP_ROOT /usr/local)
endif()

set(OpenMP_C_INCLUDE_DIR ${OpenMP_ROOT}/include)
set(OpenMP_CXX_INCLUDE_DIR ${OpenMP_ROOT}/include)

find_package(OpenMP REQUIRED)
# add_compile_options(-g -ggdb -Wall -fopenmp -isystem ${OpenMP_CXX_INCLUDE_DIRS})
# add_compile_options(-g -ggdb -Wall -O3 -fpic -fopenmp -isystem -lmkl_intel_lp64)
add_compile_options(-Wall -O3 -fpic -fopenmp -isystem -march=native -mfma)
add_compile_options(-Wall -isystem ${OpenMP_CXX_INCLUDE_DIRS})

add_executable(inference_sut 
  csrc/kmp_launcher.cpp
  csrc/hw_topology.cpp
  csrc/bert_qsl.cpp
  csrc/sut.cpp
  csrc/main.cpp  
)
set(BOOST_INC_DIRS
        ${NE_ROOT}/third_party/boost/libs/assert/include
        ${NE_ROOT}/third_party/boost/libs/core/include
        ${NE_ROOT}/third_party/boost/libs/move/include
        ${NE_ROOT}/third_party/boost/libs/config/include
        ${NE_ROOT}/third_party/boost/libs/container/include
        ${NE_ROOT}/third_party/boost/libs/intrusive/include
        ${NE_ROOT}/third_party/boost/libs/static_assert/include
        ${NE_ROOT}/third_party/boost/libs/type_traits/include
        ${NE_ROOT}/third_party/boost/libs/interprocess/include
        ${NE_ROOT}/third_party/boost/libs/unordered/include
        ${NE_ROOT}/third_party/boost/libs/container_hash/include
        ${NE_ROOT}/third_party/boost/libs/preprocessor/include
        ${NE_ROOT}/third_party/boost/libs/throw_exception/include
        ${NE_ROOT}/third_party/boost/libs/tuple/include
        ${NE_ROOT}/third_party/boost/libs/predef/include
        ${NE_ROOT}/third_party/boost/libs/mp11/include
)
file(GLOB THIRDPARTY_INCLUDE ${NE_ROOT}/third_party/*/include)

target_include_directories(inference_sut
    SYSTEM PUBLIC
        ${EXECUTOR_INCLUDE_DIRS}
        ${KERNEL_INCLUDE_DIRS}
        ${THIRDPARTY_INCLUDE}
        ${BOOST_INC_DIRS}
	${LOCAL_INSTALL_DIR}/oneDNNGraph/include
        ${NE_ROOT}/build/third_party/glog
        ${NE_ROOT}/third_party/glog/src
        ./csrc
        ./inference/loadgen
)
#find_package(OpenMP)
target_compile_options(inference_sut BEFORE PUBLIC ${OpenMP_CXX_OPTIONS})

set_property(TARGET inference_sut PROPERTY CXX_STANDARD 14)

message("Extra OpenMP C++ flags: ${OpenMP_CXX_FLAGS}")
message("Extra OpenMP C++ includes: ${OpenMP_CXX_INCLUDE_DIRS}")
message("Extra OpenMP C++ Libraries: ${OpenMP_CXX_LIBRARIES}")


if(UNIX)
  separate_arguments(OpenMP_CXX_OPTIONS UNIX_COMMAND "${OpenMP_CXX_FLAGS}")
else()
  separate_arguments(OpenMP_CXX_OPTIONS WINDOWS_COMMAND "${OpenMP_CXX_FLAGS}")
endif()

target_link_libraries(inference_sut
    PUBLIC 
        rt
        gflags
	glog::glog 
        yaml-cpp
        ${NE_LIBRARY_DIRS}/libdnnl_graph.so
        ${NE_LIBRARY_DIRS}/libglog.so
        ${NE_LIBRARY_DIRS}/libdnnl_graph.so
        ${NE_LIBRARY_DIRS}/libneural_engine.so
        ${NE_LIBRARY_DIRS}/libkernellibs.so
        ${PROJECT_SOURCE_DIR}/inference/loadgen/build/libmlperf_loadgen.a
        ${OpenMP_CXX_LIBRARIES} -lpthread -lm)
