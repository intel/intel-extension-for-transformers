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
#include "bestla/bestla_storage.h"
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

struct woq_param_base {
  std::string compute_type;  // determine gemm core template
  std::string weight_type;   // determine compressed-weight template
  std::string scale_type;    // determine scale param
  bool asym;
  int blocksize;
};

struct woq_config_param : public woq_param_base {
  dispatcher_utils::QBITS_DT src_dt;  // determine activation related template
  dispatcher_utils::QBITS_DT dst_dt;  // determine write_back template
};

struct woq_packq_param : public woq_param_base {
  bool enable_act_shuffle;
};

struct woq_packq_ctx {
  torch::Tensor *qweight, *scale, *zp, *g_idx, *output;
  int n, k;
};

struct woq_runtime_ctx {
  torch::Tensor *activation, *weight, *bias, *output;
  bool transpose;
  int m, n, k, lda, ldo;
  float alpha, beta;
  bestla::storage::gemm::IWeightBase* deseries_wei;
};

static std::map<std::string, BTLA_DTYPE> wei2bestladt_map{{"int8", BTLA_DTYPE::S8},
                                                          {"int4_clip", BTLA_DTYPE::S4_CLIP},
                                                          {"int4_fullrange", BTLA_DTYPE::S4_FULLRANGE},
                                                          {"nf4", BTLA_DTYPE::F4_NF4},
                                                          {"fp4_e2m1_bnb", BTLA_DTYPE::F4_BNB},
                                                          {"fp4_e2m1", BTLA_DTYPE::F4_E2M1},
                                                          {"fp8_e4m3", BTLA_DTYPE::F8_E4M3},
                                                          {"fp8_e5m2", BTLA_DTYPE::F8_E5M2}};
static std::map<std::string, BTLA_DTYPE> scale2bestladt_map{
    {"fp32", BTLA_DTYPE::F32}, {"bf16", BTLA_DTYPE::BF16}, {"fp8_e8m0", BTLA_DTYPE::F8_E8M0}};

void dispatch_woq_task(woq_config_param* p, woq_runtime_ctx* ctx, WOQ_TASK task);
void bestla_packq(woq_packq_param* p, woq_packq_ctx* ctx);
void set_woq_workspace(torch::Tensor* workspace);
}  // namespace woq
