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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_LAYERNORM_BA_TYPES_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_LAYERNORM_BA_TYPES_HPP_

#include <vector>
#include "param_types.hpp"

namespace jd {
namespace ssd {
enum spec_translnorm_type { normal, direct };

struct layernorm_ba_param_t {
  data_type input_dt;
  data_type output_dt;
  data_type output2_dt;
  spec_translnorm_type spec_type;
  int row_num;
  int col_num;
  int process_col;  // for control loop.
  int direct_process_row;
  int process_batch_per_ker;
  int ker_per_batch;          // for normal translnorm, each batch will be processed by same num of kers.
  int batch_num;              // for direct lnorm, pass the batch_num to the ker domain
  size_t thread_elt_offset;   // for load data.
  bool split_output = false;  // for write fp32 & u8 result to dst at the same time
  std::vector<postop_attr> postop_attrs;
  std::vector<binaryop_attr> binaryop_attrs;
};

struct layernorm_ba_data_t {
  void* src;
  void* dst;
  const float* alpha;
  const float* beta;
  const float* mean;
  const float* var;
  void* dst2;
  union {
    float n;
    int process_row;  // for direct_translnorm
  };
  const float one = 1;
  const float eps = 1e-5;
};

}  // namespace ssd
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_LAYERNORM_BA_TYPES_HPP_
