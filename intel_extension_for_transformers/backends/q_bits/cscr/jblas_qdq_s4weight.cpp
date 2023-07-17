#include "jblas_qdq_s4weight.hpp"

#include <torch/script.h>

#include "jblas/jit_blas_weight_compression.h"

void symqdq_s4weight_launcher(torch::Tensor& Fp32Wei, bool transpose,
                              int64_t block_size) {
  int k = Fp32Wei.sizes()[0];
  int n = Fp32Wei.sizes()[1];
  if (transpose) {
    int tmp = k;
    k = n;
    n = tmp;
  }
  jblas::prologue::weight_comp::gemm::
      WeightS4_KBlock<jblas::gemm::GemmCore_Row_NN_8x48_AVX512F>::qdqWeight(
          n, k, Fp32Wei.data_ptr<float>(), n, block_size, transpose);
}