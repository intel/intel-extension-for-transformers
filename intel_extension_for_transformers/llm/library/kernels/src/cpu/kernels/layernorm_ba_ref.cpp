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

#include "layernorm_ba_ref.hpp"

namespace jd {
bool layernorm_ba_ref_k_t::execute(const std::vector<const void*>& rt_data) const {
  auto op_desc = derived_kd()->get_operator_desc();
  auto op_attr = op_desc.attrs();
  auto tensor_desc = op_desc.tensor_descs();
  int batch = tensor_desc[0].shape().size() == 2 ? 1 : tensor_desc[0].shape()[0];
  int row = tensor_desc[0].shape().size() == 2 ? tensor_desc[0].shape()[0] : tensor_desc[0].shape()[1];
  int col = tensor_desc[0].shape().back();
  auto src_dt = tensor_desc[0].dtype();
  auto dst_dt = tensor_desc[1].dtype();

  float* src = reinterpret_cast<float*>(const_cast<void*>(rt_data[0]));
  float* alpha = reinterpret_cast<float*>(const_cast<void*>(rt_data[2]));
  float* beta = reinterpret_cast<float*>(const_cast<void*>(rt_data[3]));

  void* dst_data = const_cast<void*>(rt_data[1]);
  auto dst_fp32 = static_cast<float*>(dst_data);
  auto dst_u8 = static_cast<uint8_t*>(dst_data);
  auto dst_s8 = static_cast<int8_t*>(dst_data);

  auto store_data = [&](int dst_idx, float value) {
    if (dst_dt == data_type::fp32) {
      dst_fp32[dst_idx] = static_cast<float>(value);
    } else if (dst_dt == data_type::s8) {
      dst_s8[dst_idx] = static_cast<int8_t>(value);
    } else if (dst_dt == data_type::u8) {
      dst_u8[dst_idx] = static_cast<uint8_t>(value);
    }
  };

  auto normal_translnorm = [&]() {
    LOG_IF(FATAL, src_dt != data_type::fp32);
    for (int k = 0; k < batch; k++) {
      for (int i = 0; i < col; i++) {
        // calculate mean.
        float mean = 0;
        for (int j = 0; j < row; j++) mean += src[k * col * row + j * col + i];
        mean /= row;
        // calculate var
        float var = 0;
        for (int j = 0; j < row; j++)
          var += (src[k * col * row + j * col + i] - mean) * (src[k * col * row + j * col + i] - mean);
        var /= row;
        var += 1e-5;
        var = sqrt(var);
        var = 1 / var;
        // calculate layernorm.
        auto binary_op_list = op_desc.get_binaryop_list();
        for (int j = 0; j < row; j++) {
          int dst_idx = k * row * col + j * col + i;
          float value = (src[dst_idx] - mean) * var;
          value = alpha[j] * value + beta[j];
          value = apply_postop_list(value, op_desc.apply_postops_list());
          if (!binary_op_list.empty()) {
            value =
                get_quantize(value, binary_op_list[0].zp[j], 1 / binary_op_list[0].scale[j], binary_op_list[0].op_dt);
          }
          store_data(dst_idx, value);
        }
      }
    }
  };

  auto direct_translnorm = [&]() {
    LOG_IF(FATAL, src_dt != data_type::fp32 && src_dt != data_type::s32);
    float* mean_data = reinterpret_cast<float*>(const_cast<void*>(rt_data[4]));
    float* var_data = reinterpret_cast<float*>(const_cast<void*>(rt_data[5]));
    for (int i = 0; i < batch; i++) {
      for (int j = 0; j < row; j++) {
        for (int k = 0; k < col; k++) {
          int dst_idx = i * row * col + j * col + k;
          float value = src[dst_idx];
          float var = var_data[i * col + k];
          var += 1e-5;
          var = sqrt(var);
          float scale = alpha[j] / var;
          // check whether enable split-output, if true then stroe twice, one time is fp32, the other is dt(quant)
          value = (value - mean_data[i * col + k]) * scale + beta[j];
          if (op_attr["split_output"] == "true") {
            dst_dt = data_type::fp32;
            store_data(dst_idx, value);
            dst_dt = op_desc.apply_postops_list().back().dt;
            dst_u8 = static_cast<uint8_t*>(const_cast<void*>(rt_data[6]));
            dst_s8 = static_cast<int8_t*>(const_cast<void*>(rt_data[6]));
            value = apply_postop_list(value, op_desc.apply_postops_list());
            store_data(dst_idx, value);
          } else {
            value = apply_postop_list(value, op_desc.apply_postops_list());
            store_data(dst_idx, value);
          }
        }
      }
    }
  };

  if (op_attr.count("spec_type") == 0) {
    op_attr["spec_type"] = "normal";
    SPARSE_LOG(INFO) << "layernorm_ba spec_type set to normal by default.";
  } else if (op_attr["spec_type"] == "normal") {
    normal_translnorm();
  } else if (op_attr["spec_type"] == "direct") {
    direct_translnorm();
  } else {
    LOG(FATAL) << "unsupported translnorm spec type.";
  }

  return true;
}

}  // namespace jd
