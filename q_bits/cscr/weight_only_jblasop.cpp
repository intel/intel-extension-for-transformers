#include <torch/script.h>
#include <torch/torch.h>

#include "jblas_qdq_weight.hpp"
#include "jblas_quantize.hpp"
#include "jblas_quantweight_f32_linear.hpp"

static torch::Tensor jblas_quantize(const torch::Tensor& Fp32Wei, bool transpose, const std::string& alg,
                                    int64_t block_size, const std::string& compute_type,
                                    const std::string& quant_type) {
  return quant_launcher(Fp32Wei, transpose, alg, block_size, compute_type, quant_type);
}

static void jblas_symqdq_weight(torch::Tensor& Fp32Wei, bool transpose, const std::string& quant_type,
                                int64_t block_size) {
  symqdq_weight_launcher(Fp32Wei, transpose, quant_type, block_size);
}

static void jblas_quantweight_f32_linear_with_bias(const torch::Tensor& activation, const torch::Tensor& weight,
                                                   const torch::Tensor& bias, torch::Tensor& output, int64_t m,
                                                   int64_t n, int64_t k, int64_t lda, int64_t ldo,
                                                   const std::string& compute_type, const std::string& quant_type) {
  quantweight_f32_linear_launcher(activation, weight, bias.data_ptr<float>(), output, compute_type, quant_type, m, n, k,
                                  lda, ldo, true);
}

static void jblas_quantweight_f32_linear_without_bias(const torch::Tensor& activation, const torch::Tensor& weight,
                                                      torch::Tensor& output, int64_t m, int64_t n, int64_t k,
                                                      int64_t lda, int64_t ldo, const std::string& compute_type,
                                                      const std::string& quant_type) {
  quantweight_f32_linear_launcher(activation, weight, output.data_ptr<float>(), output, compute_type, quant_type, m, n,
                                  k, lda, ldo, false);
}

TORCH_LIBRARY(weight_only_jblasop, m) {
  m.def("jblas_quantize", &jblas_quantize);
  m.def("jblas_quantweight_f32_linear_with_bias", &jblas_quantweight_f32_linear_with_bias);
  m.def("jblas_quantweight_f32_linear_without_bias", &jblas_quantweight_f32_linear_without_bias);
  m.def("jblas_symqdq_weight", &jblas_symqdq_weight);
}
