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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_POSTOP_TYPES_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_POSTOP_TYPES_HPP_

namespace jd {
namespace ssd {
enum class post_op_scheme : uint8_t { exp, gelu };
enum class data_type : uint8_t {
  bf16,
  fp32,
};
struct postop_param_t {
  post_op_scheme scheme;
  size_t element_num;
  data_type dt;
};

struct postop_data_t {
  void* src;
  void* dst;
  size_t element_num;
};

}  // namespace ssd
}  // namespace jd
#endif