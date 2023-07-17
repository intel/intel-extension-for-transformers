#include <torch/script.h>
#include <torch/torch.h>

#include "jblas_qdq_s4weight.hpp"
#include "jblas_quantize.hpp"
#include "jblas_quantweight_f32_linear.hpp"

static torch::Tensor jblas_quantize(const torch::Tensor& Fp32Wei,
                                    bool transpose, int64_t bits,
                                    const std::string& alg, int64_t block_size,
                                    const std::string& compute_type) {
  return quant_launcher(Fp32Wei, transpose, bits, alg, block_size,
                        compute_type);
}

static void jblas_symqdq_s4weight(torch::Tensor& Fp32Wei, bool transpose,
                                  int64_t block_size) {
  symqdq_s4weight_launcher(Fp32Wei, transpose, block_size);
}

static void jblas_quantweight_f32_linear(const torch::Tensor& activation,
                                         const torch::Tensor& weight,
                                         torch::Tensor& output, int64_t m,
                                         int64_t n, int64_t k, int64_t lda,
                                         int64_t ldo) {
  quantweight_f32_linear_launcher(activation, weight, output, m, n, k, lda,
                                  ldo);
}

TORCH_LIBRARY(weight_only_jblasop, m) {
  m.def("jblas_quantize", &jblas_quantize);
  m.def("jblas_quantweight_f32_linear", &jblas_quantweight_f32_linear);
  m.def("jblas_symqdq_s4weight", &jblas_symqdq_s4weight);
}
