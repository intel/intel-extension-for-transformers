//  Copyright (c) 2021 Intel Corporation
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

#include "cpu_engine.hpp"
namespace jd {
const std::vector<impl_list_item_t> cpu_engine::empty_list = {};

// C API forward declaration.
#define DECLARE_IMPL_LIST(kind) \
  const std::vector<impl_list_item_t>* get_##kind##_impl_list(const operator_desc& op_desc);

DECLARE_IMPL_LIST(sparse_matmul);
DECLARE_IMPL_LIST(eltwiseop);
DECLARE_IMPL_LIST(layernorm_ba);
DECLARE_IMPL_LIST(transpose_matmul);
DECLARE_IMPL_LIST(layernormalized_spmm);
DECLARE_IMPL_LIST(softmax);
DECLARE_IMPL_LIST(gather);
DECLARE_IMPL_LIST(attention);
DECLARE_IMPL_LIST(transpose_mha);

#undef DECLARE_IMPL_LIST

const std::vector<impl_list_item_t>* cpu_engine::get_implementation_list(const operator_desc& op_desc) const {
  // Call C API.
#define CASE(kind)        \
  case kernel_kind::kind: \
    return get_##kind##_impl_list(op_desc);

  switch (op_desc.kernel_kind()) {
    CASE(sparse_matmul);
    CASE(eltwiseop);
    CASE(layernorm_ba);
    CASE(layernormalized_spmm);
    CASE(gather);
    CASE(transpose_matmul);
    CASE(softmax);
    CASE(attention);
    CASE(transpose_mha);
    default:
      return &cpu_engine::empty_list;
  }

#undef CASE
}
}  // namespace jd
