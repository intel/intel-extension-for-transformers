set(TARGET kernel_level_gemm)

#build test
add_executable(${TARGET} basic_gemm.cpp)
