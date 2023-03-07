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
  mha_dense_io_MAX = BINARY_ADD,
};
}  // namespace mha_dense_io

namespace ssd {
struct mha_dense_param_t {
  data_type dst_dt_;
  int src_bs_, src_seq_len_, head_num_, head_size_;
  float QK_rescale_, softmax_rescale_, QKV_rescale_, QKV_dstzp_;
  float Q_scale_, K_scale_, V_scale_, DST_scale_, QK_output_scale_;
  int DST_zp_;
  int base_;
};
}  // namespace ssd
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_MHA_DENSE_TYPES_HPP_
