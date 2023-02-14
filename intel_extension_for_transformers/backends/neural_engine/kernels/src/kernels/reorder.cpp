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

#include "kernels/reorder.hpp"

namespace jd {

bool reorder_kd_t::init() {
  if (!isa_available(avx512_core)) return false;
  return true;
}

bool reorder_k_t::init() {
  jit_kers_ = new jit_reorder_t(derived_kd()->params());
  if (!jit_kers_->create_kernel()) return false;
  return true;
}

bool reorder_k_t::execute(const std::vector<const void*>& rt_data) const {
  // auto& params = derived_kd()->params();
  // const auto& jit_impl = jit_kers_;
  int a = rt_data.size();
  a++;

  return true;
}

}  // namespace jd
