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
#ifdef __cplusplus
extern "C" {
#endif

// available tensor operations:
enum ne_op {
  NE_OP_NONE = 0,

  NE_OP_DUP,
  NE_OP_ADD,
  NE_OP_ADD1,
  NE_OP_ACC,
  NE_OP_SUB,
  NE_OP_MUL,
  NE_OP_DIV,
  NE_OP_SQR,
  NE_OP_SQRT,
  NE_OP_LOG,
  NE_OP_SUM,
  NE_OP_SUM_ROWS,
  NE_OP_MEAN,
  NE_OP_REPEAT,
  NE_OP_ABS,
  NE_OP_SGN,
  NE_OP_NEG,
  NE_OP_STEP,
  NE_OP_RELU,
  NE_OP_GELU,
  NE_OP_SILU,
  NE_OP_SILU_BACK,
  NE_OP_NORM,  // normalize
  NE_OP_RMS_NORM,
  NE_OP_RMS_NORM_BACK,

  NE_OP_MUL_MAT,

  NE_OP_SCALE,
  NE_OP_SET,
  NE_OP_CPY,
  NE_OP_CONT,
  NE_OP_RESHAPE,
  NE_OP_VIEW,
  NE_OP_PERMUTE,
  NE_OP_TRANSPOSE,
  NE_OP_GET_ROWS,
  NE_OP_GET_ROWS_BACK,
  NE_OP_DIAG,
  NE_OP_DIAG_MASK_INF,
  NE_OP_DIAG_MASK_ZERO,
  NE_OP_SOFT_MAX,
  NE_OP_ROPE,
  NE_OP_ROPE_BACK,
  NE_OP_ALIBI,
  NE_OP_CLAMP,
  NE_OP_CONV_1D_1S,
  NE_OP_CONV_1D_2S,

  // LLM related
  NE_OP_MUL_QKV,
  NE_OP_MUL_FFN_SILU,
  NE_OP_FLASH_ATTN,
  NE_OP_FLASH_FF,

  NE_OP_MAP_UNARY,
  NE_OP_MAP_BINARY,

  NE_OP_COUNT,
};

#ifdef __cplusplus
}
#endif
