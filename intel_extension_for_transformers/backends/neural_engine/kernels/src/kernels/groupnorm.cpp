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
#include "kernels/groupnorm.hpp"

namespace jd {

#define KERNEL_INIT_CHECK(f)                                                   \
  if (!(f)) {                                                                  \
    SPARSE_LOG(ERROR) << "Transpose attention kernel requires `" << #f << "`"; \
    return false;                                                              \
  }
using idx = exposed_enum::groupnorm::io;

bool groupnorm_kd_t::init() {
  if (!isa_available(avx512_core)) return false;
  auto op_attrs = op_desc_.attrs();
  KERNEL_INIT_CHECK(op_attrs.count("groups") != 0);
  param_.groups = str_to_num<int>(op_attrs["groups"]);
  auto src_desc = op_desc_.tensor_descs()[0];
  auto src_shape = shape();
  param_.channels = src_shape[1];
  KERNEL_INIT_CHECK(src_shape.size() == 4);  // src shape must be NCHW.
  param_.dt = src_desc.dtype();
  param_.HW = std::accumulate(src_shape.begin() + 2, src_shape.end(), 1, std::multiplies<int>());
  KERNEL_INIT_CHECK(param_.HW % (64 / get_data_size(param_.dt)) == 0);  // can't process un-aligned case now.
  return true;
}

bool groupnorm_k_t::init() {
  auto param = derived_kd()->param();
  HW_ = param.HW;
  channels_ = param.channels;
  groups_ = param.groups;
  channels_per_group_ = channels_ / groups_;
  dt_bytewidth_ = get_data_size(param.dt);
  batchs_ = derived_kd()->shape()[0];
  jit_sum_ker_ = new jit_channelwise_sum_t(param);
  jit_norm_ker_ = new jit_channelwise_norm_t(param);
  return jit_sum_ker_->create_kernel() && jit_norm_ker_->create_kernel();
}

size_t groupnorm_k_t::get_workspace_size() const { return 2 * channels_ * sizeof(float); }

bool groupnorm_k_t::execute([[maybe_unused]] const std::vector<const void*>& rt_data) const {
  float* sum_x = static_cast<float*>(const_cast<void*>(rt_data[idx::WORKSPACE]));
  float* sum_powx = static_cast<float*>(const_cast<void*>(rt_data[idx::WORKSPACE])) + channels_;
  for (int batch = 0; batch < batchs_; batch++) {
#pragma omp parallel for
    for (int channel = 0; channel < channels_; channel++) {
      channelwise_sum_data_t data;
      auto offset = (batch * channels_ + channel) * HW_ * dt_bytewidth_;
      data.src = static_cast<char*>(const_cast<void*>(rt_data[idx::SRC])) + offset;
      data.sum_x_ptr = sum_x + channel;
      data.sum_powx_ptr = sum_powx + channel;
      (*jit_sum_ker_)(&data);
    }
#pragma omp parallel for
    for (int group = 0; group < groups_; group++) {
      auto base_idx = group * channels_per_group_;
#pragma omp simd
      for (int i = 1; i < channels_per_group_; i++) {
        sum_x[base_idx] += sum_x[base_idx + i];
        sum_powx[base_idx] += sum_powx[base_idx + i];
      }
    }
#pragma omp parallel for
    for (int channel = 0; channel < channels_; channel++) {
      channelwise_norm_data_t data;
      auto group = channel / channels_per_group_;
      auto offset = (batch * channels_ + channel) * HW_ * dt_bytewidth_;
      data.src = static_cast<char*>(const_cast<void*>(rt_data[idx::SRC])) + offset;
      data.dst = static_cast<char*>(const_cast<void*>(rt_data[idx::DST])) + offset;
      data.gamma = static_cast<float*>(const_cast<void*>(rt_data[idx::GAMMA])) + channel;
      data.beta = static_cast<float*>(const_cast<void*>(rt_data[idx::BETA])) + channel;
      data.group_sum_x_ptr = sum_x + group * channels_per_group_;
      data.group_sum_powx_ptr = sum_powx + group * channels_per_group_;
      (*jit_norm_ker_)(&data);
    }
  }
  return true;
}

}  // namespace jd
