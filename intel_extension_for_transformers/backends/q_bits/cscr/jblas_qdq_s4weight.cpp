#include "jblas_qdq_s4weight.hpp"

#include <torch/script.h>

#include "jblas/jit_blas_weight_compression.h"

void qdq_s4weight_launcher(torch::Tensor& Fp32Wei, bool transpose,
                           const std::string& alg, int64_t block_size,
                           const std::string& compute_type) {}