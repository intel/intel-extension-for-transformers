cmake_minimum_required(VERSION 3.11)

project(sparselib_ut)

option(NE_WITH_SPARSELIB_VTUNE "Enable VTune profiling in kernels" OFF)
option(NE_WITH_SPARSELIB_GPU OFF)
set(CMAKE_VTUNE_HOME)

set(MAIN_SRC_GTEST "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

if(OpenMP_FOUND)
    #message(${CMAKE_CXX_FLAGS})
else()
    find_package(OpenMP REQUIRED)
    if(WIN32)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp:experimental")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

if(UNIX)
  # Enable warning
  if(NOT NE_WITH_SPARSELIB_GPU)
    ## see https://github.com/google/sanitizers/issues/611 for details)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-missing-field-initializers")
endif()

#### PART2: build gtest cases ####
file(GLOB KERNEL_TEST_CASES_SRC
    test_spmm_vnni_kernel.cpp
    test_spmm_avx512f_kernel.cpp
    test_eltwiseop_kernel.cpp
    test_layernorm_ba_kernel.cpp
    test_gather_kernel.cpp
    test_matmul_avx512f_p2031_p2013.cpp
    test_matmul_avx512f_8bit.cpp
    test_matmul_vnni_p2031_p2013.cpp
    test_matmul_vnni_noperm_p2031_p1302.cpp
    test_softmax_kernel.cpp
    test_layernormalized_spmm_kernel.cpp
    test_seq_cpy_2x8x8.cpp
    test_seq_cpy_48x4.cpp
    test_dynamic_quant_mha.cpp
    test_mha_dense_kernel.cpp
    test_slice_kernel.cpp
    test_dynamic_quant_matmul.cpp
    test_dynamic_quant.cpp
    test_mha_dense_bf16_kernel.cpp
)
if (NE_WITH_SPARSELIB_GPU)
list(APPEND KERNEL_TEST_CASES_SRC ./gpu/test_gpu_matmul.cpp)
endif()

if(NOT WIN32) # TODO(jiwei): disable attention test on windows due to lack of VNNI instruction set
list(APPEND KERNEL_TEST_CASES_SRC test_attention.cpp)
list(APPEND KERNEL_TEST_CASES_SRC test_transpose_mha.cpp)
endif()

if(UNIX)
list(APPEND KERNEL_TEST_CASES_SRC test_spmm_amx_bf16_x16_kernel.cpp)
list(APPEND KERNEL_TEST_CASES_SRC test_matmul_vnni_p2031_p2013_tr_src0.cpp)
list(APPEND KERNEL_TEST_CASES_SRC test_matmul_vnni_p2031_p2013_tr_src1.cpp)
list(APPEND KERNEL_TEST_CASES_SRC test_matmul_vnni_p2031_p2013_mm_tile.cpp)
list(APPEND KERNEL_TEST_CASES_SRC test_mha_dense_tr_k.cpp)
list(APPEND KERNEL_TEST_CASES_SRC test_mha_dense_tr_v.cpp)
list(APPEND KERNEL_TEST_CASES_SRC test_mha_dense_softmax.cpp)
list(APPEND KERNEL_TEST_CASES_SRC test_mha_dense_mm_qk.cpp)
list(APPEND KERNEL_TEST_CASES_SRC test_mha_dense_mm_av.cpp)
list(APPEND KERNEL_TEST_CASES_SRC test_trans_AB16a4b_16x.cpp)
list(APPEND KERNEL_TEST_CASES_SRC test_trans_BA16b4a_trq10n_x16.cpp)
endif()

function(register_gtest_func exe src)
    add_executable(${exe} ${MAIN_SRC_GTEST} ${src})
    target_link_libraries(${exe}
        PRIVATE
            ${CMAKE_THREAD_LIBS_INIT}
            GTest::gtest
            kernellibs)
    add_dependencies(${exe} kernellibs GTest::gtest)
    add_test(NAME ${exe_name} COMMAND ${exe_name})
    set_tests_properties(${exe_name} PROPERTIES LABELS "kernel_test")
    set_target_properties(${exe}
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
      COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endfunction()

foreach(TEST_FILE ${KERNEL_TEST_CASES_SRC})
    get_filename_component(exe_name ${TEST_FILE} NAME_WE)
    register_gtest_func(${exe_name} ${TEST_FILE})
endforeach()
