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

#include "impl_list_item.hpp"
#include "param_types.hpp"
#include "softmax.hpp"
#include "softmax_ref.hpp"
#include "src/cpu/engine/cpu_engine.hpp"
#include <map>
#include <tuple>

namespace jd {
static const std::map<kernel_prop, std::vector<impl_list_item_t>>
    softmax_impl_list_map = {
        {kernel_prop::forward_inference,
         {CPU_INSTANCE(softmax_k_t), CPU_INSTANCE(softmax_ref_k_t),
          NULL_INSTANCE()}},
};

const std::vector<impl_list_item_t> *
get_softmax_impl_list(const operator_desc &op_desc) {
  const auto impl_list_it = softmax_impl_list_map.find(op_desc.kernel_prop());
  return (impl_list_it != softmax_impl_list_map.end())
             ? &(impl_list_it->second)
             : &cpu_engine_t::empty_list;
}
} // namespace jd
