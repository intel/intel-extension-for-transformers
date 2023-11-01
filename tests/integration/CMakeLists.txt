include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/tests)


function(add_integration_test target "host_cpp")
    set(TARGET ${target})

    # build test
    add_executable(${TARGET} ${host_cpp})

    set_target_properties(${TARGET} PROPERTIES FOLDER tests/integration/)

    # Test time out, default 5 minutes
    set(INTEGRATION_TIMEOUT 300)
    # make test
    add_test(NAME ${TARGET} COMMAND ${TARGET})
    set_tests_properties(${TARGET} PROPERTIES LABELS "integration" TIMEOUT ${INTEGRATION_TIMEOUT})
endfunction()

# add_subdirectory(vector_add)
add_subdirectory(gemm)
add_subdirectory(row_reduction)
add_subdirectory(layer_norm)
add_subdirectory(data_transformer)
add_subdirectory(default_config)
add_subdirectory(sg_dropout_op)
add_subdirectory(limitation)
add_subdirectory(softmax)
