#pragma once

#include <torch/script.h>

torch::Tensor quant_launcher(const torch::Tensor& Fp32Wei, bool transpose,
                             int64_t bits, const std::string& alg,
                             int64_t block_size,
                             const std::string& compute_type);
