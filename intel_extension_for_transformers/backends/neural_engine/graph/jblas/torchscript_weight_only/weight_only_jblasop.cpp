#include <torch/script.h>
#include <torch/torch.h>

#include "jblas_quantize.hpp"

static torch::Tensor jblas_quantize(torch::Tensor Fp32Wei, int64_t nthread,
                                    int64_t bits, torch::string alg,
                                    int64_t block_size, std::string scale_dtype,
                                    std::string gemm_isa) {
  return quant_launcher(Fp32Wei, nthread, bits, alg, block_size, scale_dtype,
                        gemm_isa);
}
// static auto registry = torch::RegisterOperators("myop::skbmm", &skbmm);
TORCH_LIBRARY(weight_only_jblasop, m) {
  m.def("jblas_quantize", &jblas_quantize);
}
