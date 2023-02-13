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
#include "../utils.hpp"
namespace jd {
namespace ssd {
/**
 * @brief tensors index configuration of this kernel.
 * TODO(Yi): potential confliction with indices of other op types
 */
static constexpr int SRC0 = 0;
static constexpr int SRC1 = 1;
static constexpr int DST0 = 2;
static constexpr int SRC2 = 3;  // for binary add
static constexpr int SCALE0 = 4;
static constexpr int ZP0 = 5;

struct matmul_param_t {
  dim_t M;
  dim_t N;
  dim_t K;
  dim_t batch;                    // leading dim is `batch` times of its num_cols
  float alpha = 1.f, beta = 1.f;  // alpha * (src0 * src1) + beta * src_binary_add = dst
  dim_t m_tile = 8;
  dim_t n_tile = 2;
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

}  // namespace ssd
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_MATMUL_TYPES_HPP_
