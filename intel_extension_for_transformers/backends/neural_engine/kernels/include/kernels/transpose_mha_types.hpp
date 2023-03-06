//  Copyright (c) 2022 Intel Corporation
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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_TRANSPOSE_MHA_TYPES_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_TRANSPOSE_MHA_TYPES_HPP_

#include <vector>

#include "param_types.hpp"

namespace jd {
namespace ssd {
namespace transpose_mha_io {
enum io {
  SRC_K,
  SRC_Q,
  MASK,
  SRC_V,
  DST,
  TMP2M,  // 2M per thread of extra engine managed memory
  SL_PAD,
  BATCH,
  HEAD_NUM,
  HEAD_SIZE,
  SEQ_LEN,
  SCALE_Q,
  SCALE_K,
  SCALE_V,
  SCALE_DST,
  ZP_DST,
  transpose_mha_io_MAX = ZP_DST
};
}  // namespace transpose_mha_io

struct transpose_copy_params {
  const void* srcptr;
  void* dstptr;
  int srcstride, dststride, k;
};

struct seq_vnni_copy_params {
  const void* srcptr;
  void* dstptr;
  int srcstride, dststride, k;
};

struct transpose_mha_step1_params {
  int8_t* matA;
  int8_t* matB;
  void* matC;  // store the exp(out1), data type can be bf16/f8
  const float* matD;
  void* expsum;  // store the sum(exp(out1)), data type can be fp32/f8
  void* cfg;
  int m, k, batchk, astep, cstep, sumstep, cbatchstep;
  float scaleAB;
};

struct transpose_mha_step2_params {
  void *srcptr, *dstptr;
  void* sumptr;
  int srcstride;
  int dststride, k;
};

struct transpose_mha_step3_params {
  const int8_t* matA;
  const uint8_t* matB;
  uint8_t* matC;
  const void* cfg;
  int k, astep, cstep;
  float scaleAB, scaleC;
  int zeropointC;
};

}  // namespace ssd
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_TRANSPOSE_MHA_TYPES_HPP_
