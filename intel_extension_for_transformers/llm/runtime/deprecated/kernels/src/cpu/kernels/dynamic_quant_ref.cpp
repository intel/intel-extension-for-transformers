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

#include "dynamic_quant_ref.hpp"

namespace jd {

using io = exposed_enum::dynamic_quant::io;

void get_dynamic_quant_scale(float* mat, float* scale, int channel_num, int quantized_dim_elt_num) {
#pragma omp parallel for
  for (int channel = 0; channel < channel_num; channel++) {
    float max = 0.f;
    for (int j = 0; j < quantized_dim_elt_num; j++)
      max = max < abs(mat[channel * quantized_dim_elt_num + j]) ? abs(mat[channel * quantized_dim_elt_num + j]) : max;
    scale[channel] = max / 127.f;
  }
}

void s8_quant_mat(int8_t* dst_mat, const std::vector<float>& src_mat, float* scale, int channel_num,
                  int quantized_dim_elt_num) {
#pragma omp parallel for
  for (int channel = 0; channel < channel_num; channel++) {
    for (int j = 0; j < quantized_dim_elt_num; j++) {
      int ans = nearbyint(src_mat[channel * quantized_dim_elt_num + j] / scale[channel]);
      ans = ans > 127 ? 127 : ans;
      ans = ans < -128 ? -128 : ans;
      dst_mat[channel * quantized_dim_elt_num + j] = ans;
    }
  }
}

bool dynamic_quant_ref_k_t::execute(const std::vector<const void*>& rt_data) const {
  auto op_desc = derived_kd()->get_operator_desc();
  auto ts_desc = op_desc.tensor_descs();
  auto src_desc = ts_desc[0];
  int quantized_dim_elt_num = src_desc.shape().back();
  int channel_num =
      std::accumulate(src_desc.shape().begin(), src_desc.shape().end() - 1, size_t{1}, std::multiplies<size_t>());
  std::vector<float> fp32_src(src_desc.size(), 0);
  if (src_desc.dtype() == data_type::fp32) {
    cast_to_float_array<float>(rt_data[io::SRC], &fp32_src, src_desc.size());
  } else {
    cast_to_float_array<bfloat16_t>(rt_data[io::SRC], &fp32_src, src_desc.size());
  }
  auto mat_dst = reinterpret_cast<int8_t*>(const_cast<void*>(rt_data[io::MAT_DST]));
  auto scale_dst = reinterpret_cast<float*>(const_cast<void*>(rt_data[io::SCALE_DST]));
  get_dynamic_quant_scale(fp32_src.data(), scale_dst, channel_num, quantized_dim_elt_num);
  s8_quant_mat(mat_dst, fp32_src, scale_dst, channel_num, quantized_dim_elt_num);
  return true;
}

}  // namespace jd
