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

#include "kernels/gather.hpp"

namespace jd {

bool gather_kd_t::init() {
  auto& tensor_desc = op_desc_.tensor_descs();
  auto& src_shape = tensor_desc[0].shape();
  auto& idx_shape = tensor_desc[1].shape();
  std::vector<int64_t> dst_shape;
  auto op_attrs = op_desc_.attrs();
  int src_axis = str_to_num<dim_t>(op_attrs["src_axis"]);
  param_.src_axis_size = src_shape[src_axis];
  int idx_axis = str_to_num<dim_t>(op_attrs["idx_axis"]);
  param_.dst_axis_size = idx_shape[idx_axis];
  param_.outer_size = 1;

  SPARSE_LOG_IF(FATAL, idx_axis != idx_shape.size() - 1) << "idx_axis should be the last dim of idx";
  if (idx_axis != 0) {
    SPARSE_LOG_IF(FATAL, src_axis != idx_axis) << "idx_axis should equal to src_idx when idx_axis!=0";
    for (int i = 0; i < src_axis; i++) {
      SPARSE_LOG_IF(FATAL, src_shape[i] < idx_shape[i]) << "src shape less than idx on dim:" << i;
      param_.outer_size *= idx_shape[i];
      dst_shape.push_back(idx_shape[i]);
    }
  }
  dst_shape.push_back(param_.dst_axis_size);
  param_.inner_size = 1;
  for (int i = src_axis + 1; i < src_shape.size(); i++) {
    param_.inner_size *= src_shape[i];
    dst_shape.push_back(src_shape[i]);
  }
  int dst_size = 1;
  for (auto& i : shape()) dst_size *= i;
  SPARSE_LOG_IF(FATAL, dst_size != param_.outer_size * param_.dst_axis_size * param_.inner_size)
      << "cannot reshape to dst shape";
  param_.dt = tensor_desc[0].dtype();
  param_.dt_size = get_data_size(tensor_desc[0].dtype());
  param_.loops = param_.inner_size / (512 / 8 / param_.dt_size);
  param_.remain = param_.inner_size % (512 / 8 / param_.dt_size);
  param_.mask = (1LL << param_.remain) - 1;
  param_.extend_mask = (1LL << (param_.remain * param_.dt_size)) - 1;
  param_.binaryop_attrs = op_desc_.get_binaryop_list();
  for (int i = 0; i < param_.binaryop_attrs.size(); i++) {
    SPARSE_LOG_IF(FATAL, tensor_desc[3 + i].size() % param_.inner_size != 0) << "cannot boardcast in append op:" << i;
    param_.binary_ts_sizes.push_back(tensor_desc[3 + i].size());
  }
  return true;
}

bool gather_k_t::init() {
  jit_kers_ = new jit_gather_t(derived_kd()->params());
  if (!jit_kers_->create_kernel()) return false;
  return true;
}

bool gather_k_t::execute(const std::vector<const void*>& rt_data) const {
  auto& params = derived_kd()->params();
  const auto& jit_impl = jit_kers_;
#pragma omp parallel for collapse(2)
  for (int i = 0; i < params.outer_size; i++) {
    for (int j = 0; j < params.dst_axis_size; j++) {
      ssd::gather_data_t data_param = {
          const_cast<char*>((const char*)rt_data[0] + i * params.src_axis_size * params.inner_size * params.dt_size),
          const_cast<char*>((const char*)rt_data[1] + j * 4),  // idx is int32
          const_cast<char*>((const char*)rt_data[2] + (i * params.dst_axis_size + j) * params.inner_size * params.dt_size)};
      for (int k = 0; k < params.binaryop_attrs.size(); k++)
        data_param.binaryop_addrs[k] =
            (char*)const_cast<void*>(rt_data[3 + k]) +
            (((i * params.dst_axis_size + j) * params.inner_size) % params.binary_ts_sizes[i]) * params.dt_size;
      (*jit_impl)(&data_param);
    }
  }

  return true;
}

}  // namespace jd
