#include "jblas_qdq_weight.hpp"

#include <torch/script.h>

#include "jblas/jit_blas_weight_compression.h"

void symqdq_weight_launcher(torch::Tensor& Fp32Wei, bool transpose, const std::string& quant_type, int64_t block_size) {
  TORCH_CHECK(quant_type == "s8" || quant_type == "s4_clip" || quant_type == "s4_fullrange" || quant_type == "nf4",
              "unsupported quant_type.");
  int k = Fp32Wei.sizes()[0];
  int n = Fp32Wei.sizes()[1];
  if (transpose) {
    int tmp = k;
    k = n;
    n = tmp;
  }
  if (quant_type == "s8" || quant_type == "s4_clip") {
    jblas::prologue::weight_comp::gemm::WeightS4_Clip_KBlock<jblas::gemm::GemmCore_Row_NN_8x48_AVX512F, JblasNoSIMD>
        qdq_ker;
    qdq_ker.qdqWeight(quant_type, n, k, Fp32Wei.data_ptr<float>(), n, block_size, transpose);
  }
  if (quant_type == "s4_fullrange") {
    jblas::prologue::weight_comp::gemm::WeightS4_FullRange_KBlock<jblas::gemm::GemmCore_Row_NN_8x48_AVX512F,
                                                                  JblasNoSIMD>
        qdq_ker;
    qdq_ker.qdqWeight(quant_type, n, k, Fp32Wei.data_ptr<float>(), n, block_size, transpose);
  }
  if (quant_type == "nf4") {
    jblas::prologue::weight_comp::gemm::WeightNf4_KBlock<jblas::gemm::GemmCore_Row_NN_8x48_AVX512F, JblasNoSIMD>
        qdq_ker;
    qdq_ker.qdqWeight(quant_type, n, k, Fp32Wei.data_ptr<float>(), n, block_size, transpose);
  }
}