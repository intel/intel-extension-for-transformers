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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_ELTWISEOP_TYPES_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_ELTWISEOP_TYPES_HPP_

#include "param_types.hpp"
#include <map>
#include <set>
#include <vector>

namespace jd {
namespace ssd {
struct eltwiseop_param_t {
  size_t element_num;
  data_type in_dt;
  data_type out_dt;
  std::vector<postop_attr> postop_attrs;
  size_t element_num_each_th = 0;
  size_t remain_element = 0;
};

struct eltwiseop_data_t {
  void *src;
  void *dst;
  size_t element_num;
};

} // namespace ssd
} // namespace jd
#endif // ENGINE_SPARSELIB_INCLUDE_KERNELS_ELTWISEOP_TYPES_HPP_
