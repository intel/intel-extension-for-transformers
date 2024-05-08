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

#include "groupnorm_ref.hpp"
#include "kernels/exposed_enum.hpp"
#include "src/utils.hpp"
namespace jd {
using idx = exposed_enum::groupnorm::io;
bool groupnorm_ref_k_t::execute(const std::vector<const void*>& rt_data) const {
  auto shape = derived_kd()->shape();
  auto postop_list = derived_kd()->get_operator_desc().apply_postops_list();
  auto src_desc = derived_kd()->get_operator_desc().tensor_descs()[0];
  auto op_attrs = derived_kd()->get_operator_desc().attrs();
  int64_t batch_size = shape[0];
  auto channels = shape[1];
  auto groups = str_to_num<int>(op_attrs["groups"]);
  auto eps = str_to_num<float>(op_attrs["eps"]);
  bool affine = true;
  int64_t map_size = std::accumulate(shape.begin() + 2, shape.end(), 1, std::multiplies<int>());
  const int64_t channels_per_group = channels / groups;
  std::vector<float> fp32_src(src_desc.size(), 0);
  std::vector<float> fp32_dst(src_desc.size(), 0);
  if (src_desc.dtype() == data_type::fp32) {
    cast_to_float_array<float>(rt_data[0], &fp32_src, src_desc.size());
  } else {
    cast_to_float_array<bfloat16_t>(rt_data[0], &fp32_src, src_desc.size());
  }
  float* src_data = fp32_src.data();
  float* dst_data = fp32_dst.data();
  float* gamma_data = static_cast<float*>(const_cast<void*>(rt_data[idx::GAMMA]));
  float* beta_data = static_cast<float*>(const_cast<void*>(rt_data[idx::BETA]));

#pragma omp parallel for
  for (int64_t n = 0; n < batch_size; n++) {
    const float* src_single_data = src_data + n * channels * map_size;
    float* dst_single_data = dst_data + n * channels * map_size;
#pragma omp simd
    for (int64_t g = 0; g < groups; g++) {
      const float* src_group_data = src_single_data + g * channels_per_group * map_size;
      float* dst_group_data = dst_single_data + g * channels_per_group * map_size;
      // mean and var
      float sum = 0.f;
      for (int64_t q = 0; q < channels_per_group; q++) {
        const float* ptr = src_group_data + q * map_size;
        for (int64_t i = 0; i < map_size; i++) {
          sum += ptr[i];
        }
      }
      float mean = sum / (channels_per_group * map_size);

      float sqsum = 0.f;
      for (int64_t q = 0; q < channels_per_group; q++) {
        const float* ptr = src_group_data + q * map_size;
        for (int64_t i = 0; i < map_size; i++) {
          float tmp = ptr[i] - mean;
          sqsum += tmp * tmp;
        }
      }
      float var = sqsum / (channels_per_group * map_size);

      for (int64_t q = 0; q < channels_per_group; q++) {
        float a;
        float b;
        if (affine) {
          float gamma = gamma_data[g * channels_per_group + q];
          float beta = beta_data[g * channels_per_group + q];

          a = static_cast<float>(gamma / sqrt(var + eps));
          b = -mean * a + beta;
        } else {
          a = static_cast<float>(1.f / (sqrt(var + eps)));
          b = -mean * a;
        }

        const float* ptr = src_group_data + q * map_size;
        float* dst_ptr = dst_group_data + q * map_size;
        for (int64_t i = 0; i < map_size; i++) {
          dst_ptr[i] = apply_postop_list(ptr[i] * a + b, postop_list);
        }
      }
    }
  }

  if (src_desc.dtype() == data_type::fp32) {
    cast_from_float_array<float>(fp32_dst, const_cast<void*>(rt_data[idx::DST]), src_desc.size());
  } else {
    cast_from_float_array<bfloat16_t>(fp32_dst, const_cast<void*>(rt_data[idx::DST]), src_desc.size());
  }

  return true;
}

}  // namespace jd
