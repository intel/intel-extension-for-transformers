#include "jblas_weights4block_f32_linear.hpp"

#include <torch/script.h>

#include "jblas/jit_blas_transformer.h"
#include "jblas/jit_blas_weight_compression.h"

torch::Tensor weights4block_f32_linear_launcher(torch::Tensor activation,
                                                torch::Tensor weight,
                                                torch::Tensor output, int64_t m,
                                                int64_t n, int64_t k,
                                                int64_t lda, int64_t ldo) {
  auto wtmp = jblas::prologue::weight_comp::gemm::CompressedPackedWeight::
      deserialBuffer(weight.data_ptr<void>(), 0);
  if (wtmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_8X48 ||
      wtmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK) {
    using GemmKernel = jblas::wrapper::gemm_default::weight_comp::avx512_vnni::
        GemmSKernelDynamicS4KBlock;
    static GemmKernel kernel;
    auto ret = kernel.compute({m, n, k, activation.data_ptr<float>(), lda, wtmp,
                               output.data_ptr<float>(), ldo});
  } else if (wtmp->mCoreType == jblas::gemm::GemmCoreType::AVX512F_8X48) {
    using GemmKernel =
        jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS4KBlock;
    float alpha = 1.f, beta = 0.f;
    static GemmKernel kernel;
    auto ret =
        kernel.compute({m, n, k, activation.data_ptr<float>(), lda, wtmp,
                        output.data_ptr<float>(), output.data_ptr<float>(), ldo,
                        ldo, alpha, beta});
  }
  delete wtmp;
}