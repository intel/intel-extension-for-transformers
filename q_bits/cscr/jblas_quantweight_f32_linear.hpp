#pragma once

#include <torch/script.h>

void quantweight_f32_linear_launcher(const torch::Tensor& activation, const torch::Tensor& weight, float* bias,
                                     torch::Tensor& output, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldo,
                                     bool need_bias);