#pragma once
#include "jit_blas_epilogue.h"
#include "jit_blas_gemm.h"
#include "jit_blas_prologue.h"
#include "jit_blas_utils.h"
#include "kernel_ref.h"
#include "kernel_avx512f.h"
#include "kernel_jit.h"
#include <thread>

namespace jblas {
namespace wrapper {
namespace gemm {

} // namespace gemm

namespace gemm_default {
using DefaultParallel = jblas::utils::parallel::Parallel2DGemm;

} // namespace gemm_default
} // namespace wrapper

} // namespace jblas