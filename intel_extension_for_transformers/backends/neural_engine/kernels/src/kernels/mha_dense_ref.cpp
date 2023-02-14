//  Copyright (c) 2021 Intel Corporation
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

#include "kernels/mha_dense_ref.hpp"
#include <memory>

#define KERNEL_INIT_CHECK(f)                                         \
  if (!(f)) {                                                        \
    SPARSE_LOG(ERROR) << "MHA dense kernel requires `" << #f << "`"; \
    return false;                                                    \
  }

namespace jd {

inline std::vector<std::vector<dim_t>> get_tensor_shapes(const std::vector<tensor_desc>& descs) {
  std::vector<std::vector<dim_t>> shapes(descs.size());
  std::transform(descs.begin(), descs.end(), shapes.begin(), [&](tensor_desc d) { return d.shape(); });
  return shapes;
}

// Part1: class mha_dense_ref_kd_t

bool mha_dense_ref_kd_t::init() {
  auto& descs = op_desc_.tensor_descs();
  auto& op_attrs = op_desc_.attrs();
  merged_QKV_ = op_attrs.find("merged_QKV") != op_attrs.end() && op_attrs.at("merged_QKV") == "True";
  approx_exp_ = op_attrs.find("approx_exp") != op_attrs.end() && op_attrs.at("approx_exp") == "True";

  dst_dt_ = descs[mha_dense_io::DST].dtype();
  KERNEL_INIT_CHECK(
      is_any_of({data_type::u8, data_type::s8, data_type::fp32}, [&](const data_type t) { return dst_dt_ == t; }));
  KERNEL_INIT_CHECK(descs[mha_dense_io::SRC_Q].shape() == descs[mha_dense_io::SRC_K].shape());
  KERNEL_INIT_CHECK(descs[mha_dense_io::SRC_Q].shape() == descs[mha_dense_io::SRC_V].shape());
  KERNEL_INIT_CHECK(descs[mha_dense_io::SRC_Q].shape() == descs[mha_dense_io::DST].shape());
  KERNEL_INIT_CHECK(descs[mha_dense_io::MASK].shape() == std::vector<jd::dim_t>{bs()});

  return true;
}

inline float exp_2nd(float x) {
  constexpr float log2e = 1.442695f;
  constexpr float c0 = 0.240226507f;
  constexpr float c1 = 0.452920674f;
  constexpr float c2 = 0.713483036f;

  const float x1 = x * log2e + 0.5f;
  const float z = std::floor(x1);
  const float f = x1 - z;

  return (c0 * f * f + c1 * f + c2) * std::pow(2, z);
}

// Part2: class mha_dense_ref_k_t
mha_dense_ref_k_t::mha_dense_ref_k_t(const std::shared_ptr<const kernel_desc_t>& kd)
    : kernel_t(kd),
      approx_exp(derived_kd()->approx_exp()),
      dst_dt_(derived_kd()->dst_dt()),
      bs_(derived_kd()->bs()),
      seq_len_(derived_kd()->seq_len()),
      head_num_(derived_kd()->head_num()),
      head_size_(derived_kd()->head_size()),
      ld_src_(derived_kd()->merged_QKV() ? head_size_ * head_num_ * 3 : head_size_ * head_num_),
      ld_dst_(head_size_ * head_num_) {}

bool mha_dense_ref_k_t::init() { return true; }

template <float (*func_exp)(float)>
inline float post_softmax(float x) {
  return std::roundf(x);
}
template <>
inline float post_softmax<&std::exp>(float x) {
  return x;
}

template <float (*func_exp)(float)>
bool mha_dense_ref_k_t::execute_(const std::vector<const void*>& rt_data) const {
  // configure alias
  const mha_dense_ref_kd_t& ref_kd = *derived_kd();
  auto& op_desc = ref_kd.get_operator_desc();
  auto& attrs = op_desc.attrs();

  const float QK_rescale_ = str_to_num<float>(attrs.at("QK_rescale"));
  const float softmax_rescale_ = str_to_num<float>(attrs.at("softmax_rescale"));
  const float QKV_rescale_ = str_to_num<float>(attrs.at("QKV_rescale"));
  const float QKV_dstzp_ = str_to_num<float>(attrs.at("QKV_dstzp"));

  // stride for dims index afte perm. e.g. first elements is always for bs0
  std::vector<dim_t> left_perm_stride, right_perm_stride, dst_perm_stride;

#pragma omp parallel for collapse(2)
  for (int ibs = 0; ibs < bs_; ibs++) {
    for (int ihn = 0; ihn < head_num_; ihn++) {
      const auto mask = reinterpret_cast<const int32_t*>(rt_data[mha_dense_io::MASK])[ibs];

      const int src_offset = ibs * seq_len_ * ld_src_ + ihn * head_size_;
      const auto q_s8 = reinterpret_cast<const int8_t*>(rt_data[mha_dense_io::SRC_Q]) + src_offset;
      const auto k_s8 = reinterpret_cast<const int8_t*>(rt_data[mha_dense_io::SRC_K]) + src_offset;
      const auto v_s8 = reinterpret_cast<const int8_t*>(rt_data[mha_dense_io::SRC_V]) + src_offset;

      const int dst_offset = ibs * seq_len_ * ld_dst_ + ihn * head_size_;
      const auto dst_data = const_cast<void*>(rt_data[mha_dense_io::DST]);
      const auto dst_u8 = reinterpret_cast<uint8_t*>(dst_data) + dst_offset;
      const auto dst_s8 = reinterpret_cast<int8_t*>(dst_data) + dst_offset;
      const auto dst_f32 = reinterpret_cast<float*>(dst_data) + dst_offset;

      for (int i = 0; i < seq_len_; ++i) {
        const auto tmp_row = std::unique_ptr<float[]>(new float[mask]);

        // Q x K
        float tmp_max = -INFINITY;
        for (int j = 0; j < mask; ++j) {
          float value = 0;
          for (int k = 0; k < head_size_; ++k) {
            value += q_s8[i * ld_src_ + k] * k_s8[j * ld_src_ + k];
          }
          tmp_row[j] = value;
          tmp_max = std::max(tmp_max, value);
        }

        // softmax
        tmp_max *= QK_rescale_;
        float tmp_sum = 0;
        for (int j = 0; j < mask; ++j) {
          tmp_row[j] = func_exp(tmp_row[j] * QK_rescale_ - tmp_max);
          tmp_sum += tmp_row[j];
        }
        for (int j = 0; j < mask; ++j) {
          tmp_row[j] = post_softmax<func_exp>(tmp_row[j] * softmax_rescale_ / tmp_sum);
        }

        // A x V
        for (int j = 0; j < head_size_; ++j) {
          float value = 0;
          for (int k = 0; k < mask; ++k) {
            value += tmp_row[k] * v_s8[k * ld_src_ + j];
          }
          value = value * QKV_rescale_ + QKV_dstzp_;
          const auto idx = i * ld_src_ + j;
          switch (dst_dt_) {
            case data_type::u8:
              dst_u8[idx] = value <= 0 ? 0 : value >= UINT8_MAX ? UINT8_MAX : static_cast<uint8_t>(std::roundf(value));
              break;
            case data_type::s8:
              dst_s8[idx] = value <= INT8_MIN    ? INT8_MIN
                            : value >= UINT8_MAX ? UINT8_MAX
                                                 : static_cast<uint8_t>(std::roundf(value));
              break;
            case data_type::fp32:
              dst_f32[idx] = value;
              break;
            default:
              SPARSE_LOG(FATAL) << "Unexpected dst type!";
              break;
          }
        }
      }
    }
  }
  return true;
}

bool mha_dense_ref_k_t::execute(const std::vector<const void*>& rt_data) const {
  return approx_exp ? execute_<&exp_2nd>(rt_data) : execute_<&std::exp>(rt_data);
}
}  // namespace jd
