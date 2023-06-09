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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -Wno-error=deprecated-declarations")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-missing-field-initializers -Wno-error=deprecated-declarations")
endif()

include(FindISA.cmake)

#### PART2: build gtest cases ####
file(GLOB KERNEL_TEST_CASES_SRC
    test_spmm_vnni_kernel.cpp
    test_spmm_avx512f_kernel.cpp
    test_eltwiseop_kernel.cpp
    test_groupnorm_kernel.cpp
    test_layernorm_ba_kernel.cpp
    test_gather_kernel.cpp
    test_matmul_avx512f_p2031_p2013.cpp
    test_matmul_avx512f_8bit.cpp
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
    test_spmm_amx_bf16_x16_kernel.cpp
)
if (NE_WITH_SPARSELIB_GPU)
list(APPEND KERNEL_TEST_CASES_SRC ./gpu/test_gpu_matmul.cpp)
endif()

check_isa("avx512_vnni")
if(${avx512_vnni_found})
  list(APPEND KERNEL_TEST_CASES_SRC test_attention.cpp)
  list(APPEND KERNEL_TEST_CASES_SRC test_transpose_mha.cpp)
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
