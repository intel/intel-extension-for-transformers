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

#include <algorithm>
#include <memory>

#define KERNEL_INIT_CHECK(f)                                         \
  if (!(f)) {                                                        \
    SPARSE_LOG(ERROR) << "MHA dense kernel requires `" << #f << "`"; \
    return false;                                                    \
  }

namespace jd {
using dt = data_type;

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

  dst_dt_ = descs[io::DST].dtype();
  KERNEL_INIT_CHECK(is_any_of({dt::u8, dt::s8, dt::fp32, dt::bf16}, [dst = dst_dt_](const dt t) { return dst == t; }))

  KERNEL_INIT_CHECK((descs[io::SRC_Q].shape() == std::vector<dim_t>{bs(), sl_m(), head_num(), head_size()}))
  KERNEL_INIT_CHECK((descs[io::SRC_K].shape() == std::vector<dim_t>{bs(), sl_n(), head_num(), head_size()}))
  KERNEL_INIT_CHECK((descs[io::SRC_V].shape() == std::vector<dim_t>{bs(), sl_n(), head_num(), head_size()}))
  KERNEL_INIT_CHECK((descs[io::DST].shape() == std::vector<dim_t>{bs(), sl_m(), head_num(), head_size()}))
  KERNEL_INIT_CHECK((descs[io::MASK].shape() == std::vector<dim_t>{bs()}))

  if (descs.size() > io::BINARY_ADD && descs[io::BINARY_ADD].dtype() != dt::undef) {
    const auto& badd_shape = descs[io::BINARY_ADD].shape();
    KERNEL_INIT_CHECK(descs[io::BINARY_ADD].dtype() == dt::fp32);
    KERNEL_INIT_CHECK(sl_n() == badd_shape[badd_shape.size() - 1]);
    KERNEL_INIT_CHECK(badd_shape.size() < 3 || sl_n() == badd_shape[badd_shape.size() - 2]);
    KERNEL_INIT_CHECK(badd_shape.size() < 3 || head_num() == badd_shape[badd_shape.size() - 3]);
    KERNEL_INIT_CHECK(badd_shape.size() < 4 || bs() == badd_shape[badd_shape.size() - 4]);
  }

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
      ts_descs_(kd->get_operator_desc().tensor_descs()),
      has_badd(ts_descs_.size() > io::BINARY_ADD && ts_descs_[io::BINARY_ADD].dtype() != dt::undef),
      approx_exp(derived_kd()->approx_exp()),
      dst_dt_(derived_kd()->dst_dt()),
      bs_(derived_kd()->bs()),
      sl_m_(derived_kd()->sl_m()),
      sl_n_(derived_kd()->sl_n()),
      head_num_(derived_kd()->head_num()),
      head_size_(derived_kd()->head_size()),
      ld_src_(derived_kd()->merged_QKV() ? head_size_ * head_num_ * 3 : head_size_ * head_num_),
      ld_dst_(head_size_ * head_num_),
      badd_stride{
          !has_badd || ts_descs_[io::BINARY_ADD].shape().size() < 4 ? 0 : head_num_ * sl_m_ * sl_n_,
          !has_badd || ts_descs_[io::BINARY_ADD].shape().size() < 3 ? 0 : sl_m_ * sl_n_,
          !has_badd || ts_descs_[io::BINARY_ADD].shape().size() < 2 ? 0 : sl_n_,
          !has_badd || ts_descs_[io::BINARY_ADD].shape().size() < 1 ? 0 : 1,
      } {}

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
  const auto& attrs = derived_kd()->get_operator_desc().attrs();

  const float QK_rescale_ = attrs.find("QK_rescale") != attrs.end() ? str_to_num<float>(attrs.at("QK_rescale")) : 1.f;
  const float softmax_rescale_ =
      attrs.find("softmax_rescale") != attrs.end() ? str_to_num<float>(attrs.at("softmax_rescale")) : 1.f;
  const float QKV_rescale_ =
      attrs.find("QKV_rescale") != attrs.end() ? str_to_num<float>(attrs.at("QKV_rescale")) : 1.f;
  const float QKV_dstzp_ = attrs.find("QKV_dstzp") != attrs.end() ? str_to_num<float>(attrs.at("QKV_dstzp")) : 0.f;
  const auto has_badd = ts_descs_[io::BINARY_ADD].dtype() != dt::undef;

#pragma omp parallel for collapse(2)
  for (int ibs = 0; ibs < bs_; ibs++) {
    for (int ihn = 0; ihn < head_num_; ihn++) {
      const auto mask = reinterpret_cast<const int32_t*>(rt_data[io::MASK])[ibs];

      const int src_offset_q = ibs * sl_m_ * ld_src_ + ihn * head_size_;
      const int src_offset_kv = ibs * sl_n_ * ld_src_ + ihn * head_size_;
      const auto q_s8 = reinterpret_cast<const int8_t*>(rt_data[io::SRC_Q]) + src_offset_q;
      const auto k_s8 = reinterpret_cast<const int8_t*>(rt_data[io::SRC_K]) + src_offset_kv;
      const auto v_s8 = reinterpret_cast<const int8_t*>(rt_data[io::SRC_V]) + src_offset_kv;

      const int dst_offset = ibs * sl_m_ * ld_dst_ + ihn * head_size_;
      const auto dst_data = const_cast<void*>(rt_data[io::DST]);
      const auto dst_u8 = reinterpret_cast<uint8_t*>(dst_data) + dst_offset;
      const auto dst_s8 = reinterpret_cast<int8_t*>(dst_data) + dst_offset;
      const auto dst_f32 = reinterpret_cast<float*>(dst_data) + dst_offset;
      const auto dst_bf16 = reinterpret_cast<bfloat16_t*>(dst_data) + dst_offset;

      for (int i = 0; i < sl_m_; ++i) {
        const int badd_offset = ibs * badd_stride[0] + ihn * badd_stride[1] + i * badd_stride[2];
        const auto badd_f32 =
            has_badd ? reinterpret_cast<const float*>(rt_data[io::BINARY_ADD]) + badd_offset : nullptr;
        const auto exp_row = std::unique_ptr<float[]>(new float[mask]);

        // Q x K
        float tmp_max = -INFINITY;
        for (int j = 0; j < mask; ++j) {
          int32_t value = 0;
#pragma omp simd
          for (int k = 0; k < head_size_; ++k) value += q_s8[i * ld_src_ + k] * k_s8[j * ld_src_ + k];
          exp_row[j] = QK_rescale_ * value + (has_badd ? badd_f32[j * badd_stride[3]] : 0);
          tmp_max = std::max(tmp_max, exp_row[j]);
        }

        // softmax
        float tmp_sum = 0;
        for (int j = 0; j < mask; ++j) {
          exp_row[j] = func_exp(exp_row[j] - tmp_max);
          tmp_sum += exp_row[j];
        }
#pragma omp simd
        for (int j = 0; j < mask; ++j) {
          // round to nearest when not accurate
          exp_row[j] = post_softmax<func_exp>(exp_row[j] * softmax_rescale_ / tmp_sum);
        }

        // A x V
        for (int j = 0; j < head_size_; ++j) {
          float value = 0;
#pragma omp simd
          for (int k = 0; k < mask; ++k) {
            value += exp_row[k] * v_s8[k * ld_src_ + j];
          }
          value = value * QKV_rescale_ + QKV_dstzp_;
          const auto idx = i * ld_src_ + j;
          switch (dst_dt_) {
            case dt::u8:
              dst_u8[idx] = value <= 0 ? 0 : value >= UINT8_MAX ? UINT8_MAX : static_cast<uint8_t>(std::roundf(value));
              break;
            case dt::s8:
              dst_s8[idx] = value <= INT8_MIN    ? INT8_MIN
                            : value >= UINT8_MAX ? UINT8_MAX
                                                 : static_cast<uint8_t>(std::roundf(value));
              break;
            case dt::fp32:
              dst_f32[idx] = value;
              break;
            case dt::bf16:
              dst_bf16[idx] = make_bf16(value);
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
