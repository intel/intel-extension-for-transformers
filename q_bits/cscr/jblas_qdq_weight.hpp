#pragma once

#include <torch/script.h>

void symqdq_weight_launcher(torch::Tensor& Fp32Wei, bool transpose, int64_t bits, int64_t block_size);
