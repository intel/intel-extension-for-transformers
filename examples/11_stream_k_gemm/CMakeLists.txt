set(TARGET stream_k_gemm)

set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} -fsycl)
set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} -fsycl-targets=spir64_gen)

# disable loop invariance optimization, this is for performance
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -vc-disable-indvars-opt")
# For large registers mode, enable 256 registers for kernels
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -doubleGRF")
# For registers usage verbose at AOT
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -Xfinalizer -printregusage")
# Enable bank conflict reduction.
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -Xfinalizer -enableBCR")
# Optimization to reduce the tokens used for DPAS instruction.
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -Xfinalizer -DPASTokenReduction")

set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} -Xs)
set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} "-device pvc -options '${XETLA_OFFLINE_OPTIONS} ' ")

#build test
add_executable(${TARGET} stream_k_gemm.cpp)
target_link_options(${TARGET} PRIVATE ${XETLA_KERNEL_FLAGS})
# Disable vector combine, to remove redundant loads and stores
#target_compile_options(${TARGET} PRIVATE -mllvm -disable-vector-combine -fsycl -fsycl-targets=spir64_gen)

