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

#include "gather_ref.hpp"

namespace jd {

template <typename T>
void binary_add(void* dst, void* append_src) {
  auto dst_T = reinterpret_cast<T*>(dst);
  auto src_T = reinterpret_cast<T*>(append_src);
  *dst_T = *dst_T + *src_T;
}

bool gather_ref_k_t::execute(const std::vector<const void*>& rt_data) const {
  auto op_desc = derived_kd()->get_operator_desc();
  auto ts_descs = op_desc.tensor_descs();
  auto input_dt = ts_descs[0].dtype();
  auto src0_shape = ts_descs[0].shape();
  auto src1_shape = ts_descs[1].shape();
  auto dst_shape = ts_descs[2].shape();
  auto src0_data = reinterpret_cast<const char*>(rt_data[0]);
  auto src1_data = (const int32_t*)rt_data[1];
  auto dst_data = reinterpret_cast<char*>(const_cast<void*>(rt_data[2]));

#pragma omp parallel for
  for (int i = 0; i < src1_shape[0]; ++i) {
    int indices_val = src1_data[i];
// copy slices
#pragma omp simd
    for (int j = 0; j < src0_shape[1]; ++j) {
      memcpy(dst_data + (i * src0_shape[1] + j) * get_data_size(input_dt),
             reinterpret_cast<const char*>(src0_data) + (indices_val * src0_shape[1] + j) * get_data_size(input_dt),
             get_data_size(input_dt));
    }
  }
#pragma omp parallel for
  for (int i = 0; i < dst_shape[0]; ++i) {
#pragma omp simd
    for (int j = 0; j < dst_shape[1]; ++j) {
      // TODO(Yucheng/Zhe): refactor here when postop-injector avaliable.
      for (size_t k = 3; k < ts_descs.size(); k++) {
        int broad_cast_i = i;
        if (ts_descs[k].shape()[0] == 1) broad_cast_i = 0;
        if (input_dt == data_type::s8) {
          binary_add<int8_t>(dst_data + (i * dst_shape[1] + j) * get_data_size(input_dt),
                             reinterpret_cast<char*>(const_cast<void*>(rt_data[k])) +
                                 (broad_cast_i * dst_shape[1] + j) * get_data_size(input_dt));
        } else if (input_dt == data_type::u8) {
          binary_add<uint8_t>(dst_data + (i * dst_shape[1] + j) * get_data_size(input_dt),
                              reinterpret_cast<char*>(const_cast<void*>(rt_data[k])) +
                                  (broad_cast_i * dst_shape[1] + j) * get_data_size(input_dt));
        } else if (input_dt == data_type::fp32) {
          binary_add<float>(dst_data + (i * dst_shape[1] + j) * get_data_size(input_dt),
                            reinterpret_cast<char*>(const_cast<void*>(rt_data[k])) +
                                (broad_cast_i * dst_shape[1] + j) * get_data_size(input_dt));
        }
      }
    }
  }

  return true;
}

}  // namespace jd
