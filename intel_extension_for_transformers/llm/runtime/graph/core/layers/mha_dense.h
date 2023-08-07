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
#ifndef NE_CORE_GRAPH_MHA_DENSE_H
#define NE_CORE_GRAPH_MHA_DENSE_H

#include "core/data_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct attn_shape_t {
  int batch_size, head_num, head_size, sl_q, sl_kv;
} attn_shape_t;
size_t jblas_attn_bf16_workspace_size(const attn_shape_t* params);

typedef struct attn_bf16_fwd_args_t {
  ne_bf16_t *Q, *K, *V, *dst;
  char* tmp;
  float QK_scale;
  bool is_causal;
  int batch_size, head_num, head_size, sl_q, sl_kv;
  int step_q_bs, step_q_head_num, step_q_sl;
  int step_k_bs, step_k_head_num, step_k_sl, step_k_head_size;
  int step_v_bs, step_v_head_num, step_v_sl;
  int step_dst_bs, step_dst_head_num, step_dst_sl;
} attn_bf16_fwd_args_t;
void jblas_attn_bf16_forward(const attn_bf16_fwd_args_t* params);

typedef struct attn_fp32_fp16_fp16_fp32_fwd_args_t {
  float* Q;
  ne_fp16_t* K;
  ne_fp16_t* V;
  float* dst;
  char* tmp;
  float QK_scale;
  bool is_causal;
  int batch_size, head_num, head_size, sl_q, sl_kv;
  int step_q_bs, step_q_head_num, step_q_sl;
  int step_k_bs, step_k_head_num, step_k_sl, step_k_head_size;
  int step_v_bs, step_v_head_num, step_v_sl;
  int step_dst_bs, step_dst_head_num, step_dst_sl;
} attn_fp32_fp16_fp16_fp32_fwd_args_t;

void jblas_attn_bf16_forward(const attn_bf16_fwd_args_t* params);

bool jblas_fusion_attn_fp32_fp16_fp16_fp32_support(const attn_shape_t* params);
void jblas_fusion_attn_fp32_fp16_fp16_fp32_forward(const attn_fp32_fp16_fp16_fp32_fwd_args_t* params);

size_t jblas_fusion_attn_bf16_workspace_size(const attn_shape_t* params);

void jblas_attn_fp32_fp16_fp16_fp32_forward(const attn_fp32_fp16_fp16_fp32_fwd_args_t* params);

#ifdef __cplusplus
}
#endif
#endif  // NE_CORE_GRAPH_MHA_DENSE_H
