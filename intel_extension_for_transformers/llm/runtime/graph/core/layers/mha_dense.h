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
#include "core/ne_layers.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct attn_shape_t {
  int batch_size, head_num, head_size, sl_q, sl_kv;
} attn_shape_t;
size_t jblas_fusion_attn_workspace_size(const attn_shape_t* params);

typedef struct kv_shape_t {
  uint32_t head_num, head_size, sl_kv_max;
} kv_shape_t;

typedef enum ATTN_FWD_LAYOUT {
  // plain layout
  ATTN_FWD_LAYOUT_PLAIN,

  // step of sl/hs only works on indices which is a multiple of 64/4 on corresponding dimensions
  ATTN_FWD_LAYOUT_NTILE48_ROWPACK4,

  // step of sl/hs only works on indices which is a multiple of 32/4 on corresponding dimensions
  ATTN_FWD_LAYOUT_NTILE48_ROWPACK2,
} ATTN_FWD_LAYOUT;

typedef struct kv_cache_info_t {
  size_t k_bytes, v_bytes;
  ATTN_FWD_LAYOUT k_layout, v_layout;
  int stride_k_head_num, stride_k_sl, stride_k_head_size;
  int stride_v_head_num, stride_v_sl, stride_v_head_size;
} kv_cache_info_t;

typedef struct attn_bf16_fwd_args_t {
  ne_bf16_t *Q, *K, *V, *dst;
  float Q_sc, K_sc, V_sc, dst_sc;
  char* tmp;
  float QK_scale;
  bool is_causal;
  int batch_size, head_num, head_size, sl_q, sl_kv;
  ATTN_FWD_LAYOUT Q_layout, K_layout, V_layout, dst_layout;
  int step_q_bs, step_q_head_num, step_q_sl;
  int step_k_bs, step_k_head_num, step_k_sl, step_k_head_size;
  int step_v_bs, step_v_head_num, step_v_sl, step_v_head_size;
  int step_dst_bs, step_dst_head_num, step_dst_sl;
} attn_bf16_fwd_args_t;
void jblas_fusion_attn_bf16_forward(const attn_bf16_fwd_args_t* params);

typedef struct attn_fp32_fp16_fp16_fp32_fwd_args_t {
  float* Q;
  ne_fp16_t* K;
  ne_fp16_t* V;
  float* dst;
  float Q_sc, K_sc, V_sc, dst_sc;
  char* tmp;
  float QK_scale;
  bool is_causal;
  int batch_size, head_num, head_size, sl_q, sl_kv;
  ATTN_FWD_LAYOUT Q_layout, K_layout, V_layout, dst_layout;
  int step_q_bs, step_q_head_num, step_q_sl;
  int step_k_bs, step_k_head_num, step_k_sl, step_k_head_size;
  int step_v_bs, step_v_head_num, step_v_sl, step_v_head_size;
  int step_dst_bs, step_dst_head_num, step_dst_sl;
} attn_fp32_fp16_fp16_fp32_fwd_args_t;

void jblas_fusion_attn_bf16_forward(const attn_bf16_fwd_args_t* params);

bool jblas_fusion_attn_fp32_fp16_fp16_fp32_support(const attn_shape_t* params);
void jblas_fusion_attn_fp32_fp16_fp16_fp32_forward(const attn_fp32_fp16_fp16_fp32_fwd_args_t* params);

typedef struct attn_fp16_fwd_args_t {
  ne_fp16_t *Q, *K, *V, *dst;
  float Q_sc, K_sc, V_sc, dst_sc;
  char* tmp;
  float QK_scale;
  bool is_causal;
  int batch_size, head_num, head_size, sl_q, sl_kv;
  ATTN_FWD_LAYOUT Q_layout, K_layout, V_layout, dst_layout;
  int step_q_bs, step_q_head_num, step_q_sl;
  int step_k_bs, step_k_head_num, step_k_sl, step_k_head_size;
  int step_v_bs, step_v_head_num, step_v_sl, step_v_head_size;
  int step_dst_bs, step_dst_head_num, step_dst_sl;
} attn_fp16_fwd_args_t;
bool jblas_fusion_attn_fp16_support(const attn_shape_t* params);
void jblas_fusion_attn_fp16_forward(const attn_fp16_fwd_args_t* params);

typedef struct attn_int8_fwd_args_t {
  int8_t *Q, *K, *V, *dst;
  float Q_sc, K_sc, V_sc, dst_sc;
  char* tmp;
  float QK_scale;
  bool is_causal;
  int batch_size, head_num, head_size, sl_q, sl_kv;
  ATTN_FWD_LAYOUT Q_layout, K_layout, V_layout, dst_layout;
  int step_q_bs, step_q_head_num, step_q_sl;
  int step_k_bs, step_k_head_num, step_k_sl, step_k_head_size;
  int step_v_bs, step_v_head_num, step_v_sl, step_v_head_size;
  int step_dst_bs, step_dst_head_num, step_dst_sl;
} attn_int8_fwd_args_t;
void jblas_fusion_attn_int8_forward(const attn_int8_fwd_args_t* params);

// check if jblas_reordered_attn is supported at runtime
bool jblas_reordered_attn_fp32_support(const attn_shape_t* params);

// kv cache sizes in bytes per layer per batch per beam
void jblas_reordered_attn_fp32_batch_kv_info(const kv_shape_t* params, kv_cache_info_t* out);

typedef struct jblas_fusion_attn_fp32_update_kv_args_t {
  float* src;
  char* cache;
  int batch_size, head_num, head_size, seq_off, seq_size, seq_max;
  int step_bs, step_head_num, step_seq, step_head_size;
} jblas_fusion_attn_fp32_update_kv_args_t;
// update k-cache and output the memory layout of it
void jblas_reordered_attn_fp32_update_k(const jblas_fusion_attn_fp32_update_kv_args_t* params);
// update v-cache and output the memory layout of it
void jblas_reordered_attn_fp32_update_v(const jblas_fusion_attn_fp32_update_kv_args_t* params);

typedef struct jblas_reordered_attn_fp32_fp32_fwd_args_t {
  float* Q;
  char* K;  // K/V should be of type and layout used in corrsponding jblas_reordered_attn_xxx_update_kv
  char* V;  // K/V should be of type and layout used in corrsponding jblas_reordered_attn_xxx_update_kv
  float* dst;
  float Q_sc, K_sc, V_sc, dst_sc;
  char* tmp;
  float QK_scale;
  bool is_causal;
  int batch_size, head_num, head_size, sl_q, sl_kv;
  ATTN_FWD_LAYOUT Q_layout, K_layout, V_layout, dst_layout;
  int step_q_bs, step_q_head_num, step_q_sl;
  int stride_k_bs, stride_k_head_num, stride_k_sl, stride_k_head_size;
  int stride_v_bs, stride_v_head_num, stride_v_sl, stride_v_head_size;
  int step_dst_bs, step_dst_head_num, step_dst_sl;
} jblas_reordered_attn_fp32_fp32_fwd_args_t;
void jblas_reordered_attn_fp32_forward(const jblas_reordered_attn_fp32_fp32_fwd_args_t* params);

#ifdef __cplusplus
}
#endif
#endif  // NE_CORE_GRAPH_MHA_DENSE_H
