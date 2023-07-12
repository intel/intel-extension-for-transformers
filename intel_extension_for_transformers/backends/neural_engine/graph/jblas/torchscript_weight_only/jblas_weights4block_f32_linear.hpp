#pragma once

#include <torch/script.h>

torch::Tensor weights4block_f32_linear_launcher(torch::Tensor activation,
                                                torch::Tensor weight,
                                                torch::Tensor output, int64_t m,
                                                int64_t n, int64_t k,
                                                int64_t lda, int64_t ldo);