#pragma once

#include <torch/torch.h>
#include "jblas/jit_blas_weight_compression.h"

enum QBITS_TASK {
  QBITS_QUANTIZE,
  QBITS_DEQUANTIZE,
  QBITS_LINEAR,
};

enum QBITS_DT {
  QBITS_FP32,
  QBITS_BF16,
  QBITS_FP16,
};

struct qbits_config_param {
  std::string compute_type;  // determin gemm core template
  std::string weight_type;   // determin compress-weight template
  QBITS_DT src_dt;           // determin activation related template
  QBITS_DT dst_dt;           // determin write_back template
};

struct qbits_runtime_ctx {
  torch::Tensor *activation, *weight, *bias, *output;
  bool transpose;
  int64_t blocksize, m, n, k, lda, ldo;
  float alpha, beta;
  jblas::prologue::PackedWeight* deseries_wei;
};

void task_dispatcher(qbits_config_param* p, qbits_runtime_ctx* ctx, QBITS_TASK task);