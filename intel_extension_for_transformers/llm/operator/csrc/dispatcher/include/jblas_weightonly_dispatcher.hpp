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
#include "jblas/jit_blas_weight_compression.h"
#include <string.h>
#include <assert.h>
#include <iostream>

inline bool check_amx() { return jblas::utils::parallel::CpuDevice::getInstance()->AMX_BF16(); }
inline bool check_avx512_vnni() { return jblas::utils::parallel::CpuDevice::getInstance()->AVX512_VNNI(); }
inline bool check_avx_vnni() { return jblas::utils::parallel::CpuDevice::getInstance()->AVX_VNNI(); };
inline bool check_avx512f() { return jblas::utils::parallel::CpuDevice::getInstance()->AVX512F(); }
inline bool check_avx2() { return jblas::utils::parallel::CpuDevice::getInstance()->AVX2(); }

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
  jblas::prologue::weight_comp::gemm_kblcok::WeightBase* deseries_wei;
};

void task_dispatcher(qbits_config_param* p, qbits_runtime_ctx* ctx, QBITS_TASK task);
void set_jblas_workspace(torch::Tensor* workspace);
