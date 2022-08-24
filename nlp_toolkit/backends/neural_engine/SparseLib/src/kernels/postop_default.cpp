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

#include "kernels/postop_default.hpp"
namespace jd {

bool postop_default_kd_t::init() {
  auto op_desc = op_desc_.attrs();
  auto op_type = op_desc["post_op"];
  if (op_type == "exp") {
    params_.scheme = ssd::post_op_scheme::exp;
  } else if (op_type == "gelu") {
    params_.scheme = ssd::post_op_scheme::gelu;
  } else {
    return false;
  }
  return true;
}

bool postop_default_k_t::init() {
  jit_postop_default_t* ker = nullptr;
  auto status = postop_kernel_create(&ker, derived_kd()->params());
  if (!status) {
    return false;
  }
  jit_kers_ = ker;
  return true;
}

bool postop_default_k_t::postop_kernel_create(jit_postop_default_t** ker_pp, const ssd::postop_param_t& param) {
  *ker_pp = new jit_postop_default_t(param);
  if (*ker_pp == nullptr) {
    return false;
  }
  return (*ker_pp)->create_kernel();
}

bool postop_default_k_t::execute(const std::vector<const void*>& rt_data) const {
  int nthr = kd()->operator_desc().impl_nthr();
  int offset = 0;
  int total_element = derived_kd()->params().element_num;
  switch (derived_kd()->params().dt) {
    case ssd::data_type::fp32:
      offset = 4;
      break;
    case ssd::data_type::bf16:
      offset = 2;
      break;
    default:
      break;
  }
  int element_num_each_th = total_element / nthr;
  int remain_element = total_element - (nthr - 1) * element_num_each_th;
  const auto& jit_impl = jit_kers_;
  std::vector<ssd::postop_data_t> td(nthr);
#pragma omp parallel for
  for (int idx = 0; idx < nthr; idx++) {
    td[idx].src = const_cast<void*>(rt_data[0]) + idx * offset * element_num_each_th;
    td[idx].dst = const_cast<void*>(rt_data[1]) + idx * offset * element_num_each_th;
    if (idx != nthr - 1) {
      td[idx].element_num = element_num_each_th;
    } else {
      td[idx].element_num = remain_element;
    }
    (*jit_impl)(&(td[idx]));
  }

  return true;
}

}  // namespace jd

