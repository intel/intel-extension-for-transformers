#pragma once

#include <torch/script.h>

void symqdq_weight_launcher(torch::Tensor& Fp32Wei, bool transpose, const std::string& quant_type, int64_t block_size);
