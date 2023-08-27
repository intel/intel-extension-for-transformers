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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_SOFTMAX_TYPES_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_SOFTMAX_TYPES_HPP_

#include <vector>
#include "param_types.hpp"
#include "data_type/bf16.hpp"

namespace jd {
namespace ssd {

enum spec_softmax_type { lut };

struct softmax_param_t {
  spec_softmax_type sepc_type;
  data_type input_dt;
  data_type output_dt;
  size_t scalar_num;
  size_t vec_align_len;
  size_t vec_tail_len;
  size_t vec_num_per_thr;
  size_t vec_num_tail_thr;
  std::vector<postop_attr> postop_attrs;
  std::vector<postop_attr> get_lut_exp_attrs;
};

struct softmax_data_t {
  void* src;
  void* dst;
  void* tmp;
  size_t process_vec_num;
  bfloat16_t one;
};

}  // namespace ssd
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_SOFTMAX_TYPES_HPP_
