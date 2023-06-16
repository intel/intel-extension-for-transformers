project(gtest_ut)


set(HOST_SRC_DIR "${NE_ROOT}/executor/")
set(MAIN_SRC_GTEST "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")


include_directories(${NE_ROOT}/executor/include)
#### PART1: build host_operators.so shared library ####
add_library(host_operators STATIC
    ${HOST_SRC_DIR}/src/common.cpp
    ${HOST_SRC_DIR}/src/i_malloc.cpp
    ${HOST_SRC_DIR}/src/operators/binary_add.cpp
    ${HOST_SRC_DIR}/src/operators/layer_norm.cpp
    ${HOST_SRC_DIR}/src/operators/softmax.cpp
    ${HOST_SRC_DIR}/src/operators/logsoftmax.cpp
    ${HOST_SRC_DIR}/src/operators/padding_sequence.cpp
    ${HOST_SRC_DIR}/src/operators/one_hot.cpp
    ${HOST_SRC_DIR}/src/operators/gather.cpp
    ${HOST_SRC_DIR}/src/operators/gather_elements.cpp
    ${HOST_SRC_DIR}/src/operators/strided_slice.cpp
    ${HOST_SRC_DIR}/src/operators/matmul.cpp
    ${HOST_SRC_DIR}/src/operators/inner_product.cpp
    ${HOST_SRC_DIR}/src/operators/quantize.cpp
    ${HOST_SRC_DIR}/src/operators/reorder.cpp
    ${HOST_SRC_DIR}/src/operators/reduce_mean.cpp
    ${HOST_SRC_DIR}/src/operators/gelu.cpp
    ${HOST_SRC_DIR}/src/operators/position_ids.cpp
    ${HOST_SRC_DIR}/src/operators/token_type_ids.cpp
    ${HOST_SRC_DIR}/src/operators/cossin.cpp
    ${HOST_SRC_DIR}/src/operators/constantofshape.cpp
    ${HOST_SRC_DIR}/src/operators/concat.cpp
    ${HOST_SRC_DIR}/src/operators/embeddingbag.cpp
    ${HOST_SRC_DIR}/src/operators/split.cpp
    ${HOST_SRC_DIR}/src/operators/latrange.cpp
    ${HOST_SRC_DIR}/src/operators/convolution.cpp
    ${HOST_SRC_DIR}/src/operators/group_norm.cpp
    ${HOST_SRC_DIR}/src/operators/range.cpp
    ${HOST_SRC_DIR}/src/operators/expand_indices.cpp
    ${HOST_SRC_DIR}/src/operators/erf.cpp
    ${HOST_SRC_DIR}/src/operators/exp.cpp
    ${HOST_SRC_DIR}/src/operators/unsqueeze.cpp
    ${HOST_SRC_DIR}/src/operators/squeeze.cpp
    ${HOST_SRC_DIR}/src/operators/pow.cpp
    ${HOST_SRC_DIR}/src/operators/shape.cpp
    ${HOST_SRC_DIR}/src/operators/slice.cpp
    ${HOST_SRC_DIR}/src/operators/dequantize.cpp
    ${HOST_SRC_DIR}/src/operators/binary_op.cpp
    ${HOST_SRC_DIR}/src/sparse_operators/sparse_inner_product.cpp
    ${HOST_SRC_DIR}/src/llga_operators/llga_kernel.cpp
    ${HOST_SRC_DIR}/src/llga_operators/llga_op_creator.cpp
    ${HOST_SRC_DIR}/src/operators/rmsnorm.cpp
    ${HOST_SRC_DIR}/src/weight_compression.cpp
    ${HOST_SRC_DIR}/src/activation_dag.cpp
    ${HOST_SRC_DIR}/src/memory_allocator.cpp
    ${HOST_SRC_DIR}/src/operators/multi_head_attention.cpp
)

if(WIN32)
    if(NE_WITH_AVX2)
    set(COMPILER_ARCH_OPTION /arch:AVX2)
    else()
    set(COMPILER_ARCH_OPTION /arch:AVX512)
    endif()
else()
    if(NE_WITH_AVX2)
    set(COMPILER_ARCH_OPTION -mavx2)
    else()
    set(COMPILER_ARCH_OPTION -march=skylake-avx512)
    endif()
endif()

target_compile_options(host_operators PRIVATE ${COMPILER_ARCH_OPTION})


# Header file include path
target_include_directories(host_operators
    PUBLIC
        ${HOST_SRC_DIR}/..
        ${HOST_SRC_DIR}/include/llga_operators
        ${HOST_SRC_DIR}/include/sparse_operators
        ${HOST_SRC_DIR}/include/operators
        ${HOST_SRC_DIR}/include
        ${BOOST_INC_DIRS}

)

# link against the third party libraries
target_link_libraries(host_operators
    PUBLIC
        ${CMAKE_THREAD_LIBS_INIT}
        dnnl
        dnnl_graph
        yaml-cpp
        glog
        xbyak::xbyak
)
if(NE_WITH_SPARSELIB)
target_link_libraries(host_operators
    PUBLIC
        ${CMAKE_THREAD_LIBS_INIT}
        kernellibs
)
endif()


#### PART2: build gtest cases ####
file(GLOB OPS_TEST_CASES_SRC "*.cpp")

list(REMOVE_ITEM OPS_TEST_CASES_SRC ${PROJECT_SOURCE_DIR}/main.cpp)
if(NOT NE_WITH_SPARSELIB)
list(REMOVE_ITEM OPS_TEST_CASES_SRC ${PROJECT_SOURCE_DIR}/test_sparse_inner_product.cpp)
endif()

function(register_gtest_func)
    list(POP_FRONT ARGV exe)
    list(POP_FRONT ARGV src)
    set(libs ${ARGV})
    add_executable(${exe} ${MAIN_SRC_GTEST} ${src})
    target_link_libraries(${exe}
        PRIVATE
            ${libs}
    )
    
    target_compile_options(${exe} PRIVATE ${COMPILER_ARCH_OPTION})
    set_target_properties(${exe}
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
      COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endfunction()


list(APPEND LINK_LIBS ${CMAKE_THREAD_LIBS_INIT} GTest::gtest host_operators)
if(UNIX)
list(APPEND LINK_LIBS rt)
endif()
foreach(TEST_FILE ${OPS_TEST_CASES_SRC})
    get_filename_component(exe_name ${TEST_FILE} NAME_WE)
    register_gtest_func(${exe_name} ${TEST_FILE} ${LINK_LIBS})
    add_test(NAME ${exe_name} COMMAND ${exe_name})
    set_tests_properties(${exe_name} PROPERTIES LABELS "engine_test")
endforeach()

if(NE_WITH_SPARSELIB)
add_subdirectory(kernels)
endif()
