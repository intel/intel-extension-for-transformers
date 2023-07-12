#include <torch/script.h>
#include <torch/torch.h>

#include "jblas_quantize.hpp"
#include "jblas_weights4block_f32_linear.hpp"

static torch::Tensor jblas_quantize(const torch::Tensor& Fp32Wei,
                                    int64_t nthread, int64_t bits,
                                    const std::string& alg, int64_t block_size,
                                    const std::string& scale_dtype,
                                    const std::string& gemm_isa) {
  return quant_launcher(Fp32Wei, nthread, bits, alg, block_size, scale_dtype,
                        gemm_isa);
}

static void jblas_weights4block_f32_linear(const torch::Tensor& activation,
                                           const torch::Tensor& weight,
                                           torch::Tensor& output, int64_t m,
                                           int64_t n, int64_t k, int64_t lda,
                                           int64_t ldo) {
  weights4block_f32_linear_launcher(activation, weight, output, m, n, k, lda,
                                    ldo);
}

TORCH_LIBRARY(weight_only_jblasop, m) {
  m.def("jblas_quantize", &jblas_quantize);
  m.def("jblas_weights4block_f32_linear", &jblas_weights4block_f32_linear);
}
