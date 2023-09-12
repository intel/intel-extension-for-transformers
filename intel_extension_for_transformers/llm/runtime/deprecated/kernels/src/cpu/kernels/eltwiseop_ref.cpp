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

#include "eltwiseop_ref.hpp"

namespace jd {

bool eltwiseop_ref_k_t::execute(const std::vector<const void*>& rt_data) const {
  auto op_desc = derived_kd()->get_operator_desc();
  auto src_tensor = op_desc.tensor_descs()[0];
  auto dst_tensor = op_desc.tensor_descs()[1];
  int size = src_tensor.size();
  auto src_dt = src_tensor.dtype();
  auto dst_dt = dst_tensor.dtype();

  const void* src = rt_data[0];
  void* dst = const_cast<void*>(rt_data[1]);
  std::vector<float> src_fp32(size, 0);
  if (src_dt == data_type::s8) {
    cast_to_float_array<int8_t>(src, &src_fp32, size);
  } else if (src_dt == data_type::u8) {
    cast_to_float_array<uint8_t>(src, &src_fp32, size);
  } else if (src_dt == data_type::bf16) {
    cast_to_float_array<bfloat16_t>(src, &src_fp32, size);
  } else if (src_dt == data_type::s32) {
    cast_to_float_array<int>(src, &src_fp32, size);
  } else if (src_dt == data_type::fp32) {
    cast_to_float_array<float>(src, &src_fp32, size);
  }
  auto attr = op_desc.apply_postops_list();
  for (int i = 0; i < size; i++) {
    src_fp32[i] = apply_postop_list(src_fp32[i], attr);
  }
  if (dst_dt == data_type::s8) {
    cast_from_float_array<int8_t>(src_fp32, dst, size);
  } else if (dst_dt == data_type::u8) {
    cast_from_float_array<uint8_t>(src_fp32, dst, size);
  } else if (dst_dt == data_type::bf16) {
    cast_from_float_array<bfloat16_t>(src_fp32, dst, size);
  } else if (dst_dt == data_type::s32) {
    cast_from_float_array<int>(src_fp32, dst, size);
  } else if (dst_dt == data_type::fp32) {
    cast_from_float_array<float>(src_fp32, dst, size);
  }
  return true;
}

}  // namespace jd
