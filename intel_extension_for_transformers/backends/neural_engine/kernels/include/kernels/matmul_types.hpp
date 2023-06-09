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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_MATMUL_TYPES_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_MATMUL_TYPES_HPP_

#include <cstdint>
#include <vector>

#include "param_types.hpp"

namespace jd {
namespace ssd {

namespace matmul_io {
enum io {
  SRC0,
  SRC1,
  DST0,
  SRC2,
  SCALE0,
  ZP0,
  APPEND_SUM,
  matmul_io_MAX = APPEND_SUM,
};
}  // namespace matmul_io

namespace matmul_input {
enum input {
  SRC0,
  SRC1,
  SRC2,
  SCALE0,
  ZP0,
  APPEND_SUM,
  matmul_io_MAX = APPEND_SUM,
};
}  // namespace matmul_input

namespace matmul_output {
enum output {
  DST0,
};
}  // namespace matmul_output

struct matmul_param_t {
  dim_t M;
  dim_t N;
  dim_t K;
  dim_t batch;                    // leading dim is `batch` times of its num_cols
  float alpha = 1.f, beta = 1.f;  // alpha * (src0 * src1) + beta * src_binary_add = dst
  dim_t m_tile = 8;
  dim_t n_tile = 2;
};
struct matmul_fp8_param_t {
  dim_t M;
  dim_t N;
  dim_t K;
  float alpha = 1.f, beta = 0.f;  // alpha * (src0 * src1) + beta * src_binary_add = dst
  bfloat16_t* weight_bf16 = nullptr;
  union {
    int8_t* weight_int8;
    float8_e4m3_t* weight_f8_e4m3;
    float8_e5m2_t* weight_f8_e5m2;
    uint8_t* weight_8bit;
  };
  data_type weight_type = data_type::undef;
  bool has_scale0 = false;
  bool has_append_sum = false;
  std::vector<postop_attr> postop_attrs;
  int thread_num = 0;
};

struct matmul_data_t {
  const float* src0;
  const float* src1;
  float* dst;
  const float* src2;
};

struct matmul_u8_data_t {
  const uint8_t* src0;
  const int8_t* src1;
  uint8_t* dst;
  const float* scale;
  const float* zp;
};
struct matmul_fp8_data_t {
  bfloat16_t* matA;
  uint8_t* matB;
  bfloat16_t *matC, *matD, *matE;
  float* scale;
  int k, n, astep, bstep, cstep, dstep;
  int kpos;
  float alpha, beta;
};

}  // namespace ssd
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_MATMUL_TYPES_HPP_
