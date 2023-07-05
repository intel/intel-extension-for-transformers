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

#include "dynamic_quant.hpp"

namespace jd {
using io = exposed_enum::dynamic_quant::io;

bool dynamic_quant_kd_t::init() {
  if (!isa_available(avx512_core)) return false;
  auto ts_desc = op_desc_.tensor_descs();
  auto src_desc = ts_desc[io::SRC];
  auto dst_desc = ts_desc[io::MAT_DST];
  params_.input_dt = src_desc.dtype();
  params_.output_dt = dst_desc.dtype();
  params_.quantized_dim_elt_num = src_desc.shape().back();
  params_.ld_src = params_.quantized_dim_elt_num;
  params_.ld_dst = params_.quantized_dim_elt_num;
  return true;
}

bool dynamic_quant_k_t::init() {
  auto param = derived_kd()->params();
  int offset = 0;
  enable_thr = omp_get_max_threads();
  const auto& src_desc = kd_->get_operator_desc().tensor_descs()[io::SRC];
  size_t channel_num =
      std::accumulate(src_desc.shape().begin(), src_desc.shape().end() - 1, size_t(1), std::multiplies<size_t>());
  int remain_channel = channel_num % enable_thr;
  int channel_per_thr = channel_num / enable_thr;
  for (int i = 0; i < enable_thr; i++) {
    auto process_channel = i < remain_channel ? channel_per_thr + 1 : channel_per_thr;
    jit_kers_.push_back(new jit_dynamic_quant_t(param, process_channel));
    process_channel_list.push_back(process_channel);
    offset_list.push_back(offset);
    offset += process_channel * param.quantized_dim_elt_num;
  }

  for (auto&& i : jit_kers_) {
    if (!i->create_kernel()) return false;
  }

  return true;
}

bool dynamic_quant_k_t::execute(const std::vector<const void*>& rt_data) const {
  auto param = derived_kd()->params();
#pragma omp parallel for
  for (int i = 0; i < enable_thr; i++) {
    auto offset = offset_list[i];
    dynamic_quant_data_t data;
    auto ker = jit_kers_[i];
    data.src = reinterpret_cast<char*>(const_cast<void*>(rt_data[io::SRC])) + offset * get_data_size(param.input_dt);
    data.mat_dst = reinterpret_cast<char*>(const_cast<void*>(rt_data[io::MAT_DST])) + offset * sizeof(int8_t);
    data.scale = reinterpret_cast<char*>(const_cast<void*>(rt_data[io::SCALE_DST])) +
                 offset / param.quantized_dim_elt_num * sizeof(float);
    (*ker)(&data);
  }

  return true;
}

}  // namespace jd
