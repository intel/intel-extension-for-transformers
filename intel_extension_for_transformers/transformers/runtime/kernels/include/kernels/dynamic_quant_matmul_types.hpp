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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_DYNAMIC_QUANT_MATMUL_TYPES_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_DYNAMIC_QUANT_MATMUL_TYPES_HPP_

#include <vector>
#include "param_types.hpp"
#include "kernels/amx_utils.hpp"

namespace jd {
namespace ssd {
struct dynamic_quant_matmul_param_t {
  bool add_bias;
  int m, n, k;
  int pad_n;
  int align_build_block_num;  // num of Tmm which store the dst in row-majot-align-building-block loop.
  int align_m_loop;
  int align_n_loop;
  int tail_m = 0;
  int tail_n_loop = 0;
  int write_mask = 0;
  int tile_k;
  bool append_sum = false;
  data_type dst_dt;
  tileconfig_t m_align_cfg;
  tileconfig_t m_tail_cfg;
  std::vector<postop_attr> postop_attrs;
};

struct dynamic_quant_matmul_data_t {
  void* activation;
  void* reordered_weight;
  void* dst;
  void* scale_a;
  void* scale_w;
  void* scale_dst;
  void* bias;
  void* tmp_buf;
};

}  // namespace ssd
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_DYNAMIC_QUANT_MATMUL_TYPES_HPP_
