#pragma once

#include <torch/script.h>

void qdq_s4weight_launcher(torch::Tensor& Fp32Wei, bool transpose,
                           const std::string& alg, int64_t block_size,
                           const std::string& compute_type);
