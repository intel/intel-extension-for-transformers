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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_MEAN_VAR_REDUCE_TYPES_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_MEAN_VAR_REDUCE_TYPES_HPP_

#include <vector>
#include "param_types.hpp"

namespace jd {
namespace ssd {

struct mean_var_reduce_param_t {
  dim_t element_num;
  dim_t M;
  dim_t N;
  dim_t BM;
  dim_t BN;
};

struct mean_var_reduce_data_t {
  float* mean_in;
  float* var_in;
  float* mean_out;
  float* var_out;
};

}  // namespace ssd
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_SOFTMAX_TYPES_HPP_
