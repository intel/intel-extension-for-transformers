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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_TRANSPOSE_ATTENTION_TYPES_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_TRANSPOSE_ATTENTION_TYPES_HPP_

#include <vector>
#include "param_types.hpp"

namespace jd {
namespace ssd {
struct transpose_copy_params {
  void *srcptr, *dstptr;
  int srcstride, dststride, k;
};

struct seq_vnni_copy_params {
  void *srcptr, *dstptr;
  int srcstride, dststride, k;
};


struct transpose_attention_step1_params {
  int8_t* matA;
  int8_t* matB;
  void* matC;  // store the exp(out1), data type can be bf16/f8
  float* matD;
  void* expsum;  // store the sum(exp(out1)), data type can be fp32/f8
  void* cfg;
  int m, k, batchk, astep, cstep, sumstep, cbatchstep;
  float scaleAB;
};

struct transpose_attention_step2_params {
  void *srcptr, *dstptr;
  void* sumptr;
  int srcstride;
  int dststride, k;
};

struct transpose_attention_step3_params {
  int8_t* matA;
  uint8_t* matB;
  uint8_t* matC;
  void* cfg;
  int k, astep, cstep;
  float scaleAB, scaleC;
  int zeropointC;
};

}  // namespace ssd
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_TRANSPOSE_ATTENTION_TYPES_HPP_
