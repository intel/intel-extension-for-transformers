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
  std::string compute_type;  // determin gemm core template
  std::string weight_type;   // determin compress-weight template
  std::string scale_type;    // determin scale param
  bool asym;
  dispatcher_utils::QBITS_DT src_dt;  // determin activation related template
  dispatcher_utils::QBITS_DT dst_dt;  // determin write_back template
};

struct woq_packq_param {
  std::string compute_type;
  std::string weight_type;
  std::string scale_type;
  std::string alg;  // sym/asym
  int group_size;
  bool enable_act_shuffle;
};

struct woq_packq_ctx {
  torch::Tensor *qweight, *scale, *zp, *g_idx, *output;
  int n, k;
};

struct woq_runtime_ctx {
  torch::Tensor *activation, *weight, *bias, *output;
  bool transpose;
  int blocksize, m, n, k, lda, ldo;
  float alpha, beta;
  jblas::storage::gemm::IWeightBase* deseries_wei;
};

static std::map<std::string, JBLAS_DTYPE> wei2jblasdt_map{
    {"int4_clip", JBLAS_DTYPE::S4_CLIP}, {"int4_fullrange", JBLAS_DTYPE::S4_FULLRANGE},
    {"nf4", JBLAS_DTYPE::F4_NF4},        {"fp4_e2m1_bnb", JBLAS_DTYPE::F4_BNB},
    {"fp4_e2m1", JBLAS_DTYPE::F4_E2M1},  {"fp8_e4m3", JBLAS_DTYPE::F8_E4M3},
    {"fp8_e5m2", JBLAS_DTYPE::F8_E5M2}};
static std::map<std::string, JBLAS_DTYPE> scale2jblasdt_map{{"fp32", JBLAS_DTYPE::F32},
                                                            {"fp8_e8m0", JBLAS_DTYPE::F8_E8M0}};

void dispatch_woq_task(woq_config_param* p, woq_runtime_ctx* ctx, WOQ_TASK task);
void jblas_packq(woq_packq_param* p, woq_packq_ctx* ctx);
void set_woq_workspace(torch::Tensor* workspace);
}  // namespace woq
