include_directories(${CMAKE_SOURCE_DIR}/tests/integration/gemm)

add_subdirectory(bf16)
add_subdirectory(fp16)
add_subdirectory(fp32)
add_subdirectory(int8_quantization)
add_subdirectory(int8)
add_subdirectory(tf32)
add_subdirectory(int4_dequantization)
