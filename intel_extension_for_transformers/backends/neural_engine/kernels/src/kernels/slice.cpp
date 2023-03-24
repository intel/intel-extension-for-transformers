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

#include "kernels/slice.hpp"

namespace jd {

bool slice_kd_t::init() {
  if (!isa_available(avx512_core)) return false;
  auto& tensor_desc = op_desc_.tensor_descs();
  auto op_attrs = op_desc_.attrs();
  param_.begin = str_to_num<int>(op_attrs["begin"]);
  param_.step = str_to_num<int>(op_attrs["step"]);
  size_t axis = str_to_num<int>(op_attrs["axis"]);
  SPARSE_LOG_IF(FATAL, param_.step > 3) << "step only support 1,2 now";
  auto& src_shape = tensor_desc[0].shape();
  param_.src_axis_size = src_shape[axis];
  for (size_t i = 0; i < src_shape.size(); i++)
    if (i < axis)
      param_.outer_size *= src_shape[i];
    else if (i == axis)
      param_.dst_axis_size = shape()[axis];
    else
      param_.inner_size *= src_shape[i];
  SPARSE_LOG_IF(FATAL, param_.begin + (param_.dst_axis_size - 1) * param_.step >= param_.src_axis_size)
      << "slice out of range. Please check begin, step and length(the axis of dst tensor)";
  param_.dt_size = get_data_size(tensor_desc[0].dtype());

  if (param_.step == 1)
    param_.copy_size = param_.dst_axis_size * param_.inner_size * param_.dt_size;
  else if (param_.inner_size > 1)
    param_.copy_size = param_.inner_size * param_.dt_size;
  else
    param_.copy_size = (param_.dst_axis_size - 1) * param_.dt_size * param_.step;
  return true;
}

bool slice_k_t::init() {
  auto& params = derived_kd()->params();
  jit_kers_ = new jit_slice_t(params);
  if (!jit_kers_->create_kernel()) return false;
  return true;
}

bool slice_k_t::execute(const std::vector<const void*>& rt_data) const {
  auto& params = derived_kd()->params();
  const auto& jit_impl = jit_kers_;

  if (params.inner_size > 1 && params.step > 1) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < params.outer_size; i++)
      for (int j = 0; j < params.dst_axis_size; j++) {
        ssd::slice_data_t data_param =
            ssd::slice_data_t(const_cast<char*>(reinterpret_cast<const char*>(rt_data[0]) +
                                                (i * params.src_axis_size + params.begin + j * params.step) *
                                                    params.inner_size * params.dt_size),
                              const_cast<char*>(reinterpret_cast<const char*>(rt_data[1]) +
                                                (i * params.dst_axis_size + j) * params.inner_size * params.dt_size));
        (*jit_impl)(&data_param);
      }
  } else {
#pragma omp parallel for
    for (int i = 0; i < params.outer_size; i++) {
      ssd::slice_data_t data_param = ssd::slice_data_t(
          const_cast<char*>(reinterpret_cast<const char*>(rt_data[0]) +
                            (i * params.src_axis_size + params.begin) * params.inner_size * params.dt_size),
          const_cast<char*>(reinterpret_cast<const char*>(rt_data[1]) +
                            i * params.dst_axis_size * params.inner_size * params.dt_size));
      (*jit_impl)(&data_param);
    }
  }
  return true;
}

}  // namespace jd
