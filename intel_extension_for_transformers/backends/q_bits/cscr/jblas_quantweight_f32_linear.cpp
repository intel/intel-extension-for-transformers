#include "jblas_quantweight_f32_linear.hpp"

#include <torch/script.h>

#include "jblas/jit_blas_transformer.h"
#include "jblas/jit_blas_weight_compression.h"

#define AVX512F_LINEAR_EXECUTE                                                 \
  static GemmKernel kernel;                                                    \
  auto ret = kernel.compute({m, n, k, activation.data_ptr<float>(), lda, wtmp, \
                             output.data_ptr<float>(), bias_ptr, ldo, 0,       \
                             alpha, beta});

using CompType = jblas::prologue::weight_comp::gemm::WeightCompType;

void quantweight_f32_linear_launcher(const torch::Tensor& activation,
                                     const torch::Tensor& weight,
                                     const torch::Tensor& bias,
                                     torch::Tensor& output, int64_t m,
                                     int64_t n, int64_t k, int64_t lda,
                                     int64_t ldo, bool need_bias) {
  float* bias_ptr = output.data_ptr<float>();
  float alpha = 1.f, beta = 0.f;
  if (need_bias) {
    beta = 1.f;
    bias_ptr = bias.data_ptr<float>();
  }
  auto wtmp = jblas::prologue::weight_comp::gemm::CompressedPackedWeight::
      deserialBuffer(weight.data_ptr<int8_t>(), 0);
  if (wtmp->mType == static_cast<int>(CompType::S4_Bf16) ||
      wtmp->mType == static_cast<int>(CompType::S4_F32)) {
    if (wtmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_8X48 ||
        wtmp->mCoreType == jblas::gemm::GemmCoreType::AVX512_VNNI_3X48_KBLOCK) {
      using GemmKernel = jblas::wrapper::gemm_default::weight_comp::
          avx512_vnni::GemmSKernelDynamicS4KBlock;
      static GemmKernel kernel;
      auto ret = kernel.compute({m, n, k, activation.data_ptr<float>(), lda,
                                 wtmp, output.data_ptr<float>(), ldo});
    } else if (wtmp->mCoreType == jblas::gemm::GemmCoreType::AVX512F_8X48) {
      using GemmKernel = jblas::wrapper::gemm_default::weight_comp::avx512f::
          GemmKernelS4KBlock;
      AVX512F_LINEAR_EXECUTE
    }
  } else if (wtmp->mType == static_cast<int>(CompType::S8_F32)) {
    using GemmKernel =
        jblas::wrapper::gemm_default::weight_comp::avx512f::GemmKernelS8KBlock;
    AVX512F_LINEAR_EXECUTE
  } else {
    TORCH_CHECK(false, "unsupported comptype.");
  }
  delete wtmp;
}