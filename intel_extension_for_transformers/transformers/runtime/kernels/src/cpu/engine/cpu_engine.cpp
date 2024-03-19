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

#include "src/cpu/engine/cpu_engine.hpp"
#include "src/cpu/memory_storege/cpu_memory_storage.hpp"
#include "src/singleton.hpp"
#include "kernel_cache.hpp"

namespace jd {
const std::vector<impl_list_item_t> cpu_engine_t::empty_list = {};

// C API forward declaration.
#define DECLARE_IMPL_LIST(kind) \
  const std::vector<impl_list_item_t>* get_##kind##_impl_list(const operator_desc& op_desc);

DECLARE_IMPL_LIST(sparse_matmul);
DECLARE_IMPL_LIST(eltwiseop);
DECLARE_IMPL_LIST(groupnorm);
DECLARE_IMPL_LIST(layernorm_ba);
DECLARE_IMPL_LIST(transpose_matmul);
DECLARE_IMPL_LIST(dynamic_quant_matmul);
DECLARE_IMPL_LIST(layernormalized_spmm);
DECLARE_IMPL_LIST(softmax);
DECLARE_IMPL_LIST(gather);
DECLARE_IMPL_LIST(attention);
DECLARE_IMPL_LIST(transpose_mha);
DECLARE_IMPL_LIST(mha_dense);
DECLARE_IMPL_LIST(slice);
DECLARE_IMPL_LIST(dynamic_quant);

#undef DECLARE_IMPL_LIST

const std::vector<impl_list_item_t>* cpu_engine_t::get_implementation_list(const operator_desc& op_desc) const {
  // Call C API.
#define CASE(kind)        \
  case kernel_kind::kind: \
    return get_##kind##_impl_list(op_desc);

  switch (op_desc.kernel_kind()) {
    CASE(sparse_matmul);
    CASE(eltwiseop);
    CASE(groupnorm);
    CASE(layernorm_ba);
    CASE(layernormalized_spmm);
    CASE(gather);
    CASE(transpose_matmul);
    CASE(softmax);
    CASE(attention);
    CASE(transpose_mha);
    CASE(mha_dense);
    CASE(slice);
    CASE(dynamic_quant_matmul);
    CASE(dynamic_quant);
    default:
      return &cpu_engine_t::empty_list;
  }

#undef CASE
}
bool cpu_engine_t::create_memory_storage(memory_storage_t** storage) const {
  *storage = new cpu_memory_storage_t(this);
  return true;
}

bool cpu_engine_t::create_kernel(const operator_desc& op_desc, std::shared_ptr<kernel_t>& kernel,
                                 const stream_t*) const {
  auto impl_list_ = get_implementation_list(op_desc);
  if (impl_list_ == nullptr) {
    return false;
  }
  // Step 1: Get the first && success object in impl_list_.
  std::shared_ptr<const kernel_desc_t> result_kd;
  std::shared_ptr<const kernel_desc_t> candidate_kd;
  auto& impl_list = (*impl_list_);
  for (auto& impl : impl_list) {
    auto status = impl(candidate_kd, op_desc);  // kd->create() + kd->init()
    if (status) {
      result_kd = candidate_kd;
      break;
    }
  }
  // step 2 create kernel
  kernel_cache* global_primitive_cache = Singleton<kernel_cache>::GetInstance();
  const auto& callback = std::bind(&kernel_desc_t::create_primitive, result_kd, std::placeholders::_1,
                                   result_kd);  // k_t->create() + k_t->init()
  std::shared_ptr<const kernel_t> value =
      global_primitive_cache->find_or_construct(result_kd->get_operator_desc(), callback);
  if (value == nullptr) {
    return false;
  }
  kernel = std::const_pointer_cast<kernel_t>(value);
  return true;
}
}  // namespace jd
