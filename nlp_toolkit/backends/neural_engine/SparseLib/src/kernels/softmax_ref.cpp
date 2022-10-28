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

#include "kernels/softmax_ref.hpp"

namespace jd {

bool softmax_ref_kd_t::init() {
  auto op_attrs = op_desc_.attrs();
  if (op_attrs["spec_type"] == "lut") {
    // assert int8 dt as input.
    auto tensor_desc = op_desc_.tensor_descs();
    if (tensor_desc.size() != 2)
      SPARSE_LOG(ERROR) << "softmax lut kernel need 2 tensor descriptor:src & dst." << std::endl;
    auto input_dt = tensor_desc[0].dtype();
    auto output_dt = tensor_desc[1].dtype();
    SPARSE_LOG_IF(
        FATAL,
        output_dt != data_type::bf16);  // TODO(zhe1wang): support more dt,current impl is for experiment only.
    if (get_data_size(input_dt) != 1)
      SPARSE_LOG(ERROR) << "softmax lut kernel only support int8 dtype as input currently." << std::endl;
  } else {
    SPARSE_LOG(ERROR) << "do not supported specialization softmax type" << std::endl;
  }
  return true;
}

bool softmax_ref_k_t::execute(const std::vector<void*>& rt_data) const {
  auto op_desc = derived_kd()->operator_desc();
  auto src_s8 = reinterpret_cast<int8_t*>(rt_data[0]);
  auto src_u8 = reinterpret_cast<uint8_t*>(rt_data[0]);
  auto dst = reinterpret_cast<bfloat16_t*>(rt_data[1]);
  auto postop_lists = op_desc.apply_postops_list();
  auto src_tensor = op_desc.tensor_descs()[0];
  auto src_dt = src_tensor.dtype();
  auto tensor_shape = src_tensor.shape();
  int row = src_tensor.reduce_rows();
  int col = tensor_shape.back();
  std::vector<float> float_dst_data(row * col, 0);
  for (int i = 0; i < row; i++) {
    // step1. find max
    float max = static_cast<float>(src_u8[0]);
    for (int j = 0; j < col; j++) {
      int src_idx = i * col + j;
      if (src_dt == jd::data_type::s8) {
        max = static_cast<float>(src_s8[src_idx]) > max ? static_cast<float>(src_s8[src_idx]) : max;
      } else {
        max = static_cast<float>(src_u8[src_idx]) > max ? static_cast<float>(src_u8[src_idx]) : max;
      }
    }
    // get e^M
    float one_div_exp_M = 1.0 / get_exp(apply_postop_list(max, postop_lists));
    // step2. compute sum of exp
    float exp_sum = 0;
    for (int j = 0; j < col; j++) {
      float value = 0;
      if (src_dt == jd::data_type::s8) {
        value = apply_postop_list(static_cast<float>(src_s8[i * col + j] - max), postop_lists);
      } else {
        value = apply_postop_list(static_cast<float>(src_u8[i * col + j] - max), postop_lists);
      }
      value = get_exp(value) * one_div_exp_M;
      float_dst_data[i * col + j] = (value);
      exp_sum += value;
    }

    float scale = 1 / exp_sum;

    // step3. compute softmax
    for (int j = 0; j < col; j++) dst[i * col + j] = make_bf16(float_dst_data[i * col + j] * scale);
  }
  return true;
}

}  // namespace jd
