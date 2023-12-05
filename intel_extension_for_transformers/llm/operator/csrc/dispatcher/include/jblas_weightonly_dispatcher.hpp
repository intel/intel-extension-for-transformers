//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#pragma once
#include <ATen/core/TensorBody.h>
#include <torch/torch.h>
#include "jblas/jit_blas_storage.h"
#include "../include/dispatcher_utils.hpp"
#include <string.h>
#include <assert.h>
#include <iostream>
namespace woq {

enum WOQ_TASK {
  WOQ_QUANTIZE,
  WOQ_DEQUANTIZE,
  WOQ_LINEAR,
};

struct woq_config_param {
  std::string compute_type;           // determin gemm core template
  std::string weight_type;            // determin compress-weight template
  dispatcher_utils::QBITS_DT src_dt;  // determin activation related template
  dispatcher_utils::QBITS_DT dst_dt;  // determin write_back template
};

struct woq_runtime_ctx {
  torch::Tensor *activation, *weight, *bias, *output;
  bool transpose;
  int64_t blocksize, m, n, k, lda, ldo;
  float alpha, beta;
  jblas::storage::gemm::IWeightBase* deseries_wei;
};

void dispatch_woq_task(woq_config_param* p, woq_runtime_ctx* ctx, WOQ_TASK task);
void set_woq_workspace(torch::Tensor* workspace);
}  // namespace woq
