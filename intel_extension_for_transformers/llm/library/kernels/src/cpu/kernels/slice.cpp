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

#include "slice.hpp"

#include <functional>

#define KERNEL_INIT_CHECK(f)                                         \
  if (!(f)) {                                                        \
    SPARSE_LOG(ERROR) << "MHA dense kernel requires `" << #f << "`"; \
    return false;                                                    \
  }

namespace jd {

bool slice_kd_t::init() {
  const auto& attrs = op_desc_.attrs();
  KERNEL_INIT_CHECK(attrs.find("step") != attrs.end());
  KERNEL_INIT_CHECK(attrs.find("begin") != attrs.end());
  KERNEL_INIT_CHECK(attrs.find("axis") != attrs.end());
  KERNEL_INIT_CHECK(attrs.size() == 3)
  step_ = str_to_num<int>(attrs.at("step"));
  begin_ = str_to_num<int>(attrs.at("begin"));
  axis_ = str_to_num<int>(attrs.at("axis"));

  SPARSE_LOG_IF(FATAL, step_ > 3) << "step only support 1,2 now";

  return true;
}

slice_k_t::slice_k_t(const std::shared_ptr<const kd_t>& kd)
    : kernel_t(kd),
      ts_descs(derived_kd()->get_operator_desc().tensor_descs()),
      src_shape(ts_descs[0].shape()),
      dst_shape(ts_descs[1].shape()),
      axis(derived_kd()->axis()),
      begin(derived_kd()->begin()),
      step(derived_kd()->step()),
      dt_size(get_data_size(ts_descs[0].dtype())),
      outer_size(std::accumulate(src_shape.cbegin(), src_shape.cbegin() + axis, 1, std::multiplies<int>())),
      src_axis_size(src_shape[axis]),
      dst_axis_size(dst_shape[axis]),
      inner_size(std::accumulate(src_shape.cbegin() + axis + 1, src_shape.cend(), 1, std::multiplies<int>())) {
  const auto src_axis_size = src_shape[axis];
  SPARSE_LOG_IF(FATAL, begin + (dst_axis_size - 1) * step + 1 > src_axis_size)
      << "slice out of range. Please check begin, step and length(the axis of dst tensor)";
}

bool slice_k_t::init() {
  const auto copy_size = (step == 1)        ? dst_axis_size * inner_size * dt_size
                         : (inner_size > 1) ? inner_size * dt_size
                                            : dst_axis_size * dt_size * step;

  jit_kern_.reset(new jit_slice_t({
      /* .use_avx512 = */ isa_available(avx512_core),
      /* .step = */ step,
      /* .src_axis_size = */ src_axis_size,
      /* .inner_size = */ inner_size,
      /* .copy_size = */ copy_size,
      /* .dt_size = */ dt_size,
  }));
  if (!jit_kern_->create_kernel()) return false;
  return true;
}

bool slice_k_t::execute(const std::vector<const void*>& rt_data) const {
  const auto src = reinterpret_cast<const char*>(rt_data[0]);
  const auto dst = reinterpret_cast<char*>(const_cast<void*>(rt_data[1]));
  if (inner_size > 1 && step > 1) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < outer_size; i++)
      for (int j = 0; j < dst_axis_size; j++) {
        jit_slice_t::rt_data_t data_param{
            src + (i * src_axis_size + begin + j * step) * inner_size * dt_size,
            dst + (i * dst_axis_size + j) * inner_size * dt_size,
        };
        (*jit_kern_)(&data_param);
      }
  } else {
#pragma omp parallel for
    for (int i = 0; i < outer_size; i++) {
      jit_slice_t::rt_data_t data_param{
          src + (i * src_axis_size + begin) * inner_size * dt_size,
          dst + i * dst_axis_size * inner_size * dt_size,
      };
      (*jit_kern_)(&data_param);
    }
  }
  return true;
}

}  // namespace jd
