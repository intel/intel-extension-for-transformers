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

#include "gather.hpp"

#include "src/cpu/cpu_isa.hpp"

namespace jd {

bool gather_kd_t::init() { return true; }

gather_k_t::gather_k_t(const std::shared_ptr<const kd_t>& kd)
    : kernel_t(kd),
      ts_descs(derived_kd()->get_operator_desc().tensor_descs()),
      src_axis(str_to_num<size_t>(derived_kd()->get_operator_desc().attrs().at("src_axis"))),
      idx_axis(str_to_num<size_t>(derived_kd()->get_operator_desc().attrs().at("idx_axis"))),
      dt_size(get_data_size(ts_descs[io::SRC].dtype())),
      src_axis_size(ts_descs[io::SRC].shape()[src_axis]),
      dst_axis_size(ts_descs[io::IDX].shape()[idx_axis]),
      src_size(ts_descs[io::SRC].size()),
      idx_size(ts_descs[io::IDX].size()),
      binary_ops(derived_kd()->get_operator_desc().get_binaryop_list()),
      binary_op_sizes(),
      has_avx512(isa_available(avx512_core)) {
  for (size_t i = 0; i < binary_ops.size(); i++) binary_op_sizes.push_back(ts_descs[io::BINARY0 + i].size());
}

bool gather_k_t::init() {
  const auto& src_shape = ts_descs[io::SRC].shape();
  const auto& idx_shape = ts_descs[io::IDX].shape();

  outer_size = 1;
  if (src_axis != 0 && idx_axis != 0) {
    SPARSE_LOG_IF(FATAL, src_axis != idx_axis) << "src_axis should equal to idx_axis when both of them are not zero";
    for (size_t i = 0; i < src_axis; i++) {
      SPARSE_LOG_IF(FATAL, src_shape[i] < idx_shape[i]) << "src shape less than idx on dim:" << i;
      outer_size *= idx_shape[i];
    }
  } else if (src_axis != 0) {
    for (size_t i = 0; i < src_axis; i++) outer_size *= src_shape[i];
  } else if (idx_axis != 0) {
    for (size_t i = 0; i < idx_axis; i++) outer_size *= idx_shape[i];
  }

  inner_size = 1;
  for (size_t i = src_axis + 1; i < src_shape.size(); i++) inner_size *= src_shape[i];

  SPARSE_LOG_IF(FATAL, ts_descs[2].size() != outer_size * dst_axis_size * inner_size) << "cannot reshape to dst shape";

  jit_kern_.reset(new jit_gather_t({
      /* .use_avx512 = */ has_avx512,
      /* .dt = */ ts_descs[io::SRC].dtype(),
      /* .dt_size = */ dt_size,
      /* .src_axis_size = */ src_axis_size,
      /* .dst_axis_size = */ dst_axis_size,
      /* .src_size = */ src_size,
      /* .idx_size = */ idx_size,
      /* .outer_size = */ outer_size,
      /* .inner_size = */ inner_size,
      /* .binary_ops = */ binary_ops,
  }));
  if (!jit_kern_->create_kernel()) return false;
  return true;
}

bool gather_k_t::execute(const std::vector<const void*>& rt_data) const {
  const auto src = reinterpret_cast<const char*>(rt_data[io::SRC]);
  const auto idx = reinterpret_cast<const int32_t*>(rt_data[io::IDX]);
  const auto dst = reinterpret_cast<char*>(const_cast<void*>(rt_data[io::DST]));
#pragma omp parallel for collapse(2)
  for (int i = 0; i < outer_size; i++) {
    for (int j = 0; j < dst_axis_size; j++) {
      jit_gather_t::rt_data_t data{src + ((i * src_axis_size * inner_size) % src_size) * dt_size,
                                   idx + (i * dst_axis_size) % idx_size + j,
                                   dst + (i * dst_axis_size + j) * inner_size * dt_size};
      for (size_t k = 0; k < binary_ops.size(); k++) {
        const auto binary = reinterpret_cast<const char*>(rt_data[io::BINARY0 + k]);
        data.binaryop_addrs[k] = binary + (((i * dst_axis_size + j) * inner_size) % binary_op_sizes[k]) * dt_size;
      }
      (*jit_kern_)(&data);
    }
  }

  return true;
}

}  // namespace jd
