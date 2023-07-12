#pragma once

#include <torch/script.h>

torch::Tensor quant_launcher(const torch::Tensor& Fp32Wei, int64_t nthread,
                             int64_t bits, torch::string alg,
                             int64_t block_size, std::string scale_dtype,
                             std::string gemm_isa);
