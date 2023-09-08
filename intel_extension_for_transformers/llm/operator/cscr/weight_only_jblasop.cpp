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

#include <torch/script.h>
#include <torch/torch.h>

#include "dispatcher/jblas_task_dispatcher.hpp"

template <QBITS_DT SRC_DT, QBITS_DT DST_DT>
static void inline init_qbits_config_param(qbits_config_param* p,
                                           const std::string& compute_type,
                                           const std::string& weight_type) {
  p->compute_type = compute_type;
  p->weight_type = weight_type;
  p->src_dt = SRC_DT;
  p->dst_dt = DST_DT;
}

static torch::Tensor qbits_quantize(const torch::Tensor& fp32_weight,
                                    bool transpose, int64_t block_size,
                                    const std::string& compute_type,
                                    const std::string& weight_type) {
  torch::Tensor output;
  qbits_config_param p;
  init_qbits_config_param<QBITS_FP32, QBITS_FP32>(&p, compute_type,
                                                  weight_type);
  qbits_runtime_ctx ctx{nullptr,   const_cast<torch::Tensor*>(&fp32_weight),
                        nullptr,   &output,
                        transpose, block_size};
  task_dispatcher(&p, &ctx, QBITS_QUANTIZE);
  return output;
}

static void qbits_dequantize(const torch::Tensor& compressed_weight,
                             torch::Tensor& dequantize_weight, bool transpose,
                             const std::string& compute_type,
                             const std::string& weight_type) {
  qbits_config_param p;
  init_qbits_config_param<QBITS_FP32, QBITS_FP32>(&p, compute_type,
                                                  weight_type);
  qbits_runtime_ctx ctx{nullptr, const_cast<torch::Tensor*>(&compressed_weight),
                        nullptr, &dequantize_weight, transpose};
  task_dispatcher(&p, &ctx, QBITS_DEQUANTIZE);
}

static void qbits_f32in_f32out_linear_with_bias(
    const torch::Tensor& activation, const torch::Tensor& weight,
    const torch::Tensor& bias, torch::Tensor& output, int64_t lda, int64_t ldo,
    const std::string& compute_type, const std::string& weight_type) {
  qbits_config_param p;
  init_qbits_config_param<QBITS_FP32, QBITS_FP32>(&p, compute_type,
                                                  weight_type);
  qbits_runtime_ctx ctx{
      const_cast<torch::Tensor*>(&activation),
      const_cast<torch::Tensor*>(&weight),
      const_cast<torch::Tensor*>(&bias),
      &output,
  };
  ctx.lda = lda;
  ctx.ldo = ldo;
  ctx.m = activation.sizes()[0];
  ctx.k = activation.sizes()[1];
  ctx.n = bias.sizes()[0];
  ctx.alpha = 1.f;
  ctx.beta = 1.f;
  task_dispatcher(&p, &ctx, QBITS_LINEAR);
}

static void qbits_f32in_f32out_linear_without_bias(
    const torch::Tensor& activation, const torch::Tensor& weight,
    torch::Tensor& output, int64_t n, int64_t lda, int64_t ldo,
    const std::string& compute_type, const std::string& weight_type) {
  qbits_config_param p;
  init_qbits_config_param<QBITS_FP32, QBITS_FP32>(&p, compute_type,
                                                  weight_type);
  qbits_runtime_ctx ctx{
      const_cast<torch::Tensor*>(&activation),
      const_cast<torch::Tensor*>(&weight),
      &output,
      &output,
  };
  ctx.lda = lda;
  ctx.ldo = ldo;
  ctx.m = activation.sizes()[0];
  ctx.k = activation.sizes()[1];
  ctx.n = n;
  ctx.alpha = 1.f;
  ctx.beta = 0.f;
  task_dispatcher(&p, &ctx, QBITS_LINEAR);
}

TORCH_LIBRARY(weight_only_jblasop, m) {
  m.def("qbits_quantize", &qbits_quantize);
  m.def("qbits_f32in_f32out_linear_with_bias",
        &qbits_f32in_f32out_linear_with_bias);
  m.def("qbits_f32in_f32out_linear_without_bias",
        &qbits_f32in_f32out_linear_without_bias);
  m.def("qbits_dequantize", &qbits_dequantize);
}