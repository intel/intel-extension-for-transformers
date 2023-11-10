include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/tests)
include_directories(${CMAKE_SOURCE_DIR}/tests/unit)

function(add_unit_test target kernel_func_file test_host)
    set(TARGET ${target})

    # build test
    add_executable(${TARGET} ${test_host})

    target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/tests/unit/${TARGET})
    set_target_properties(${TARGET} PROPERTIES FOLDER tests/unit/)

    # Test time out, default 1 minutes
    set(UNIT_TIMEOUT 500)

    # make test
    add_test(NAME ${TARGET} COMMAND ${TARGET})
    set_tests_properties(${TARGET} PROPERTIES LABELS "unit" TIMEOUT ${UNIT_TIMEOUT})
endfunction()

add_subdirectory(global_load_store)
add_subdirectory(global_atomic)
add_subdirectory(block_load_store)
add_subdirectory(tile_load_store)
add_subdirectory(tile_load_store_local)
add_subdirectory(internal_type_load_store_cvt)
add_subdirectory(local_load_store)
add_subdirectory(raw_send)
add_subdirectory(buff_compare)
add_subdirectory(tile_mma)
add_subdirectory(named_barrier)
add_subdirectory(tile_row_reduction)
add_subdirectory(add_c)
add_subdirectory(imul)
add_subdirectory(philox_rng)
add_subdirectory(exp_inv_sqrt_tanh)
add_subdirectory(reg_layout_conversion)
add_subdirectory(reg_reduce)
add_subdirectory(math_general)
add_subdirectory(epilogue_tile_op)
add_subdirectory(bit_mask_manipulation)

