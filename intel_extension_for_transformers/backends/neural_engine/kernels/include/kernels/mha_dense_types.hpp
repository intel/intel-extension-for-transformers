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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_MHA_DENSE_TYPES_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_MHA_DENSE_TYPES_HPP_

#include <vector>

#include "amx_utils.hpp"
#include "param_types.hpp"

namespace jd {
namespace mha_dense_io {
enum io {
  SRC_Q,
  SRC_K,
  SRC_V,
  MASK,
  DST,
  WORKSPACE,
  BINARY_ADD,

  ATT_SCALE,  // scale the QxK; the `1/sqrt(seqlen)`

  Q_SCALE,
  Q_ZP,
  K_SCALE,
  K_ZP,
  V_SCALE,
  V_ZP,
  SRC_DST_SCALE,  // input scale for dst tensor
  SRC_DST_ZP,     // input zp for dst tensor
  DST_SCALE,      // output scale for dst tensor
  DST_ZP,         // output zp for dst tensor

  BATCH_SIZE,
  HEAD_NUM,
  HEAD_SIZE,
  M,  // "seq_len" for Q & DST
  N,  // "seq_len" for K & V

  mha_dense_io_MAX = N,
};
}  // namespace mha_dense_io

namespace ssd {
struct mha_dense_param_t {
  data_type dst_dt_;
  format_type kv_ft_;
  int src_bs_, src_sl_m_, src_sl_n_, head_num_, head_size_;
  int base_;
};
}  // namespace ssd
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_MHA_DENSE_TYPES_HPP_
