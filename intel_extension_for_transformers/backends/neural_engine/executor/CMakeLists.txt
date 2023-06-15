project(neural_engine)

find_library(LIBJEMALLOC jemalloc ${NE_THIRD_PARTY_DIR}/jemalloc/lib)
if(${LIBJEMALLOC} STREQUAL LIBJEMALLOC-NOTFOUND)
  MESSAGE(STATUS "Build JEMalloc lib. Please install autoconf in advance: conda/apt install autoconf.")
  execute_process(COMMAND bash autogen.sh --disable-initial-exec-tls
                  WORKING_DIRECTORY ${NE_THIRD_PARTY_DIR}/jemalloc)
  execute_process(COMMAND make -j
                WORKING_DIRECTORY ${NE_THIRD_PARTY_DIR}/jemalloc)
else()
  MESSAGE(STATUS "JEMalloc lib path = ${NE_THIRD_PARTY_DIR}/jemalloc/lib.")
endif()

FILE(GLOB COMMON_SRCS "src/*.cpp")
FILE(GLOB COMMON_HEAS "include/*.h" "include/*.hpp")
FILE(GLOB OPERATORS "src/operators/*.cpp")
FILE(GLOB OPERATORS_HEAS "include/operators/*.h" "include/operators/*.hpp")
FILE(GLOB LLGA_OPERATORS "src/llga_operators/*.cpp")
FILE(GLOB LLGA_OPERATORS_HEAS "include/llga_operators/*.h" "include/llga_operators/*.hpp")
FILE(GLOB SPARSEOPS "src/sparse_operators/*.cpp")
FILE(GLOB SPARSEOPS_HEAS  "include/sparse_operators/*.h" "include/sparse_operators/*.hpp")
LIST(REMOVE_ITEM OPERATORS "${PROJECT_SOURCE_DIR}/src/operators/transpose.cpp")
#TODO STATIC support
set(SRC_SET
    ${COMMON_SRCS}
    ${OPERATORS}
    ${SPARSEOPS}

    ${COMMON_HEAS}
    ${OPERATORS_HEAS}
    ${SPARSEOPS_HEAS}
)
if(NE_WITH_ONEDNN_GRAPH)
  list(APPEND SRC_SET
    ${LLGA_OPERATORS}
    ${LLGA_OPERATORS_HEAS}
  )
endif()


add_library(neural_engine SHARED ${SRC_SET})
target_compile_definitions(neural_engine PRIVATE NEURALENGINE_SHARED_LIB_)

if(WIN32)
    if(NE_WITH_AVX2)
    target_compile_options(neural_engine PRIVATE /arch:AVX2)
    else()
    target_compile_options(neural_engine PRIVATE /arch:AVX512)
    endif()
else()
    target_compile_options(neural_engine PRIVATE -march=native)
endif()

target_compile_definitions(neural_engine PRIVATE NEURALENGINE_BUILD)

set(DEP_LIBS
     dnnl
     yaml-cpp
     glog
)
if(NE_WITH_ONEDNN_GRAPH)
list(APPEND DEP_LIBS dnnl_graph)
endif()
add_dependencies(neural_engine
  ${DEP_LIBS}
)

# Header file include path
target_include_directories(neural_engine
    PUBLIC
        ../
        ./include/sparse_operators
        ./include/llga_operators
        ./include/operators
        ./include
        ${BOOST_INC_DIRS}
)

# link against the third party libraries
set(LINK_LIBS
    ${DEP_LIBS}
)

if(NE_WITH_SPARSELIB)
  target_compile_definitions(neural_engine PRIVATE WITH_SPARSELIB)
  list(APPEND LINK_LIBS kernellibs)
endif()
if(WIN32)
else()
list(APPEND LINK_LIBS rt)
endif()

target_link_libraries(neural_engine
    PUBLIC
        ${CMAKE_THREAD_LIBS_INIT}
        ${LINK_LIBS}
)

## pybind11
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.9)
  cmake_policy(SET CMP0069 NEW)
endif()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
pybind11_add_module(neural_engine_py ${PROJECT_SOURCE_DIR}/python/bind_executor.cpp)

add_dependencies(neural_engine_py
    neural_engine
)

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
add_compile_definitions(neural_engine_bin PRIVATE NE_VERSION_STRING=${NE_VERSION_STRING})

add_dependencies(neural_engine_bin
    neural_engine
)

target_include_directories(neural_engine_bin
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${BOOST_INC_DIRS}
)

target_link_libraries(neural_engine_bin
    PRIVATE
        ${CMAKE_THREAD_LIBS_INIT}
        gflags
        neural_engine
)

set_target_properties(neural_engine_bin
        PROPERTIES OUTPUT_NAME neural_engine_bin) #corupt with neuralengine

set(TARGETS neural_engine neural_engine_bin neural_engine_py)

install(TARGETS neural_engine neural_engine_bin)
