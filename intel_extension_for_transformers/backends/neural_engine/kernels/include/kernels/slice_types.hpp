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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_SLICE_TYPES_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_SLICE_TYPES_HPP_

#include <vector>
#include "param_types.hpp"

namespace jd {
namespace ssd {
struct slice_param_t {
  int begin, step, axis;
  int outer_size = 1, src_axis_size = 1, dst_axis_size = 1, inner_size = 1, copy_size;
  int dt_size;
};

struct slice_data_t {
  void* src;
  void* dst;

 public:
  slice_data_t(void* a, void* b) : src(a), dst(b) {}
};

}  // namespace ssd
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_SLICE_TYPES_HPP_
