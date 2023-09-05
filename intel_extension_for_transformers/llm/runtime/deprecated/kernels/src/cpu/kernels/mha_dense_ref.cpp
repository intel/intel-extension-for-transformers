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

#include "mha_dense_ref.hpp"

#include <algorithm>
#include <memory>

#include "include/engine.hpp"
#include "include/engine_factory.hpp"
#include "mha_dense_ctx.hpp"
#include "src/utils.hpp"

#define KERNEL_INIT_CHECK(f)                                         \
  if (!(f)) {                                                        \
    SPARSE_LOG(ERROR) << "MHA dense kernel requires `" << #f << "`"; \
    return false;                                                    \
  }

namespace jd {

// Part1: class mha_dense_ref_kd_t

bool mha_dense_ref_kd_t::init() {
  const auto shapes = op_desc_.tensor_shapes();
  const auto dtypes = op_desc_.tensor_dtypes();
  const auto ftypes = op_desc_.tensor_ftypes();
  auto& op_attrs = op_desc_.attrs();
  merged_QKV_ = op_attrs.find("merged_QKV") != op_attrs.end() && op_attrs.at("merged_QKV") == "True";
  approx_exp_ = op_attrs.find("approx_exp") != op_attrs.end() && op_attrs.at("approx_exp") == "True";
  stable_softmax_ = op_attrs.find("stable_softmax") != op_attrs.end() && op_attrs.at("stable_softmax") == "True";

  dst_dt_ = dtypes[io::DST];
  KERNEL_INIT_CHECK(is_any_of({data_type::u8, data_type::s8, data_type::fp32, data_type::bf16},
                              [dst = dst_dt_](const data_type t) { return dst == t; }))

  // dynamic shape
  KERNEL_INIT_CHECK((bs() > 0 || shapes[io::BATCH_SIZE] == std::vector<dim_t>{1}));
  KERNEL_INIT_CHECK((head_num() > 0 || shapes[io::HEAD_NUM] == std::vector<dim_t>{1}));
  KERNEL_INIT_CHECK((head_size() > 0 || shapes[io::HEAD_SIZE] == std::vector<dim_t>{1}));
  KERNEL_INIT_CHECK((sl_m() > 0 || shapes[io::M] == std::vector<dim_t>{1}));
  KERNEL_INIT_CHECK((sl_n() > 0 || shapes[io::N] == std::vector<dim_t>{1}));

  KERNEL_INIT_CHECK((shapes[io::SRC_Q] == std::vector<dim_t>{bs(), sl_m(), head_num(), head_size()}))
  KERNEL_INIT_CHECK((shapes[io::SRC_K] == std::vector<dim_t>{bs(), sl_n(), head_num(), head_size()}))
  KERNEL_INIT_CHECK((shapes[io::SRC_V] == std::vector<dim_t>{bs(), sl_n(), head_num(), head_size()}))
  KERNEL_INIT_CHECK((shapes[io::DST] == std::vector<dim_t>{bs(), sl_m(), head_num(), head_size()}))

  KERNEL_INIT_CHECK(ftypes[io::MASK] == format_type::undef ||
                    (dtypes[io::MASK] == data_type::s32 && shapes[io::MASK] == std::vector<dim_t>{bs()}))

  // attention scale
  KERNEL_INIT_CHECK((dtypes[io::ATT_SCALE] == data_type::fp32))
  KERNEL_INIT_CHECK((ftypes[io::ATT_SCALE] == format_type::a))
  KERNEL_INIT_CHECK(
      (shapes[io::ATT_SCALE] == std::vector<dim_t>{bs()} || shapes[io::ATT_SCALE] == std::vector<dim_t>{1}))

  // can not use DST_SCALE as input and output at the same time
  KERNEL_INIT_CHECK(dtypes[io::DST_SCALE] == data_type::undef || dtypes[io::SRC_DST_SCALE] == data_type::undef)

  KERNEL_INIT_CHECK((shapes[io::Q_SCALE].size() == 0 ||  //
                     shapes[io::Q_SCALE] == std::vector<dim_t>{1} ||
                     shapes[io::Q_SCALE] == std::vector<dim_t>{bs(), sl_m()}))
  KERNEL_INIT_CHECK((shapes[io::K_SCALE].size() == 0 ||  //
                     shapes[io::K_SCALE] == std::vector<dim_t>{1} ||
                     shapes[io::K_SCALE] == std::vector<dim_t>{bs(), sl_n()}))
  KERNEL_INIT_CHECK((shapes[io::V_SCALE].size() == 0 ||  //
                     shapes[io::V_SCALE] == std::vector<dim_t>{1} ||
                     shapes[io::V_SCALE] == std::vector<dim_t>{bs(), sl_n()}))
  KERNEL_INIT_CHECK((shapes[io::SRC_DST_SCALE].size() == 0 || shapes[io::SRC_DST_SCALE] == std::vector<dim_t>{1} ||
                     shapes[io::SRC_DST_SCALE] == std::vector<dim_t>{bs(), sl_m()}))

  KERNEL_INIT_CHECK((shapes[io::DST_SCALE].size() == 0 ||  //
                     shapes[io::DST_SCALE] == std::vector<dim_t>{bs(), sl_m()}))

  // currently only support s8
  KERNEL_INIT_CHECK((shapes[io::Q_ZP].empty()))
  KERNEL_INIT_CHECK((shapes[io::K_ZP].empty()))
  KERNEL_INIT_CHECK((shapes[io::V_ZP].empty()))
  KERNEL_INIT_CHECK((shapes[io::DST_ZP].empty()))

  KERNEL_INIT_CHECK(ftypes[io::SRC_Q] == format_type::abcd)
  KERNEL_INIT_CHECK(ftypes[io::DST] == format_type::abcd)
  KERNEL_INIT_CHECK(ftypes[io::SRC_K] == format_type::abcd || ftypes[io::SRC_K] == format_type::acbd)
  KERNEL_INIT_CHECK(ftypes[io::SRC_V] == ftypes[io::SRC_K])

  if (shapes.size() > io::BINARY_ADD && dtypes[io::BINARY_ADD] != data_type::undef) {
    const auto badd_shape_bcst = pre_pad1(4, shapes[io::BINARY_ADD]);
    KERNEL_INIT_CHECK(dtypes[io::BINARY_ADD] == data_type::fp32)
    KERNEL_INIT_CHECK(badd_shape_bcst[0] == 1 || badd_shape_bcst[0] == bs())
    KERNEL_INIT_CHECK(badd_shape_bcst[1] == 1 || badd_shape_bcst[1] == head_num())
    KERNEL_INIT_CHECK(badd_shape_bcst[2] == 1 || badd_shape_bcst[2] == sl_m())
    KERNEL_INIT_CHECK(badd_shape_bcst[3] == 1 || badd_shape_bcst[3] == sl_n())
  }

  // dtype
  KERNEL_INIT_CHECK(is_all_of({dtypes[io::SRC_Q], dtypes[io::SRC_K], dtypes[io::SRC_V], dtypes[io::DST]},
                              [&](const data_type t) { return t != data_type::undef; }));

  return true;
}

inline float exp_2nd(float x) {
  static const float log2e = std::log2(std::exp(1.f));
  static const float ln2 = std::log(2.f);
  const float z = std::ceil(x * log2e);
  const float f = x - z * ln2;
  constexpr std::array<float, 3> exp_approx_f32_coeff{0.35815147f, 0.96963238f, 1.f};
  auto&& coeff = exp_approx_f32_coeff;
  return ldexpf(coeff[0] * f * f + coeff[1] * f + coeff[2], z);  // same as a * std::pow(2, z) but more precise
}

// Part2: class mha_dense_ref_k_t
mha_dense_ref_k_t::mha_dense_ref_k_t(const std::shared_ptr<const kernel_desc_t>& kd)
    : kernel_t(kd),
      ts_descs_(kd->get_operator_desc().tensor_descs()),
      has_badd(ts_descs_.size() > io::BINARY_ADD && ts_descs_[io::BINARY_ADD].dtype() != data_type::undef),
      approx_exp(derived_kd()->approx_exp()),
      stable_softmax(derived_kd()->stable_softmax()),
      dst_dt_(derived_kd()->dst_dt()),
      dst_v_(ts_descs_[io::SRC_V].dtype()),
      kv_ft_(derived_kd()->kv_ft()),
      bs_(derived_kd()->bs()),
      sl_m_(derived_kd()->sl_m()),
      sl_n_(derived_kd()->sl_n()),
      head_num_(derived_kd()->head_num()),
      head_size_(derived_kd()->head_size()),
      is_dynq10n_dst(ts_descs_.size() > io::DST_SCALE && ts_descs_[io::DST_SCALE].dtype() != data_type::undef),
      workspace_size_(!is_dynq10n_dst ? 0 : sizeof(float) * ts_descs_[io::DST].size()) {}

bool mha_dense_ref_k_t::init() { return true; }

template <float (*func_exp)(float)>
inline float post_softmax(float x) {
  return std::roundf(x);
}
template <>
inline float post_softmax<&std::exp>(float x) {
  return x;
}

static float quant_s8(const float* src, size_t n, int8_t* dst) {
  float absmax = 0.f;
#pragma omp simd
  for (size_t i = 0; i < n; ++i) absmax = std::max(absmax, std::abs(src[i]));
  const float scale = absmax / INT8_MAX;
  constexpr auto epsilon = 1e-9f;
  const float rcpscale = INT8_MAX / std::max(absmax, epsilon);
#pragma omp simd
  for (size_t i = 0; i < n; ++i) dst[i] = std::roundf(src[i] * rcpscale);
  return scale;
}

template <float (*func_exp)(float)>
bool mha_dense_ref_k_t::execute_(const exec_context_t& ctx) const {
  void *src_data[io_src::SIZE], *dst_data[io_dst::SIZE], *workspace;
  dim_t shape_data[io_shape::SIZE];
  for (auto i = 0; i < io_src::SIZE; ++i) ctx.input(i)->get_handle(&src_data[i]);
  for (auto i = 0; i < io_dst::SIZE; ++i) ctx.output(i)->get_handle(&dst_data[i]);
  for (auto i = 0; i < io_shape::SIZE; ++i) shape_data[i] = ctx.get_dynamic_shape()[i];
  ctx.workspace()->get_handle(&workspace);

  const auto bs = shape_data[io_shape::BATCH_SIZE] > 0 ? shape_data[io_shape::BATCH_SIZE] : bs_;
  const auto sl_m = shape_data[io_shape::M] > 0 ? shape_data[io_shape::M] : sl_m_;
  const auto sl_n = shape_data[io_shape::N] > 0 ? shape_data[io_shape::N] : sl_n_;
  const auto head_num = shape_data[io_shape::HEAD_NUM] > 0 ? shape_data[io_shape::HEAD_NUM] : head_num_;
  const auto head_size = shape_data[io_shape::HEAD_SIZE] > 0 ? shape_data[io_shape::HEAD_SIZE] : head_size_;

  const auto ld_q = (derived_kd()->merged_QKV() ? 3 : 1) * head_size * head_num;
  const auto ld_kv = kv_ft_ == format_type::abcd   ? ld_q
                     : kv_ft_ == format_type::acbd ? head_size_ * (derived_kd()->merged_QKV() ? 3 : 1)
                                                   : 0;
  const auto ld_dst = head_size_ * head_num_;
  auto badd_shape_static = pre_pad1(4, ts_descs_[io::BINARY_ADD].shape());
  badd_shape_static[0] = badd_shape_static[0] > 1 ? bs : badd_shape_static[0];
  badd_shape_static[1] = badd_shape_static[1] > 1 ? head_num : badd_shape_static[1];
  badd_shape_static[2] = badd_shape_static[2] > 1 ? sl_m : badd_shape_static[2];
  badd_shape_static[3] = badd_shape_static[3] > 1 ? sl_n : badd_shape_static[3];
  const auto badd_step = dim2step(badd_shape_static);

  // configure alias
  const auto& attrs = derived_kd()->get_operator_desc().attrs();
  const float softmax_rescale_ =  //
      attrs.find("softmax_rescale") == attrs.end() ? UINT8_MAX
      : attrs.at("softmax_rescale") == "dynamic"   ? 0.f
                                                   : str_to_num<float>(attrs.at("softmax_rescale"));

  const auto has_badd = ts_descs_[io::BINARY_ADD].dtype() != data_type::undef;
  // default scale and zp
  const float def_scale = 1.f;
  const int32_t def_zp = 0;

  const auto q_scale_num = ts_descs_[io::Q_SCALE].size();
  const auto k_scale_num = ts_descs_[io::K_SCALE].size();
  const auto v_scale_num = ts_descs_[io::V_SCALE].size();
  const auto dst_scale_num = ts_descs_[io::SRC_DST_SCALE].size();
  const auto dst_zp_num = ts_descs_[io::SRC_DST_ZP].size();
  const auto q_scale_f32 = reinterpret_cast<const float*>(src_data[io_src::Q_SCALE]);
  const auto k_scale_f32 = reinterpret_cast<const float*>(src_data[io_src::K_SCALE]);
  const auto v_scale_f32 = reinterpret_cast<const float*>(src_data[io_src::V_SCALE]);
  const auto dst_scale_f32 = reinterpret_cast<const float*>(src_data[io_src::SRC_DST_SCALE]);
  const auto dst_zp_s32 = reinterpret_cast<const int32_t*>(src_data[io_src::SRC_DST_ZP]);

#pragma omp parallel for collapse(3)
  for (int ibs = 0; ibs < bs; ibs++) {
    for (int ihn = 0; ihn < head_num; ihn++) {
      for (int i = 0; i < sl_m; ++i) {  // put at the top so it can be paralleled
        const auto curr_sl_n = ts_descs_[io::MASK].dtype() == data_type::undef
                                   ? sl_n
                                   : reinterpret_cast<const int32_t*>(src_data[io_src::MASK])[ibs];

        const int src_offset_q = ibs * sl_m * ld_q + ihn * head_size;
        const int src_offset_kv = kv_ft_ == format_type::abcd   ? ibs * sl_n * ld_kv + ihn * head_size
                                  : kv_ft_ == format_type::acbd ? (ibs * head_num + ihn) * sl_n * ld_kv
                                                                : 0;
        const auto q_s8 = reinterpret_cast<const int8_t*>(src_data[io_src::SRC_Q]) + src_offset_q;
        const auto k_s8 = reinterpret_cast<const int8_t*>(src_data[io_src::SRC_K]) + src_offset_kv;
        const auto v_s8 = reinterpret_cast<const int8_t*>(src_data[io_src::SRC_V]) + src_offset_kv;
        const auto q_bf16 = reinterpret_cast<const bfloat16_t*>(src_data[io_src::SRC_Q]) + src_offset_q;
        const auto k_bf16 = reinterpret_cast<const bfloat16_t*>(src_data[io_src::SRC_K]) + src_offset_kv;
        const auto v_bf16 = reinterpret_cast<const bfloat16_t*>(src_data[io_src::SRC_V]) + src_offset_kv;

        const int dst_offset = ibs * sl_m * ld_dst + ihn * head_size;
        const auto dst_u8 = reinterpret_cast<uint8_t*>(dst_data[io_dst::DST]) + dst_offset;
        const auto dst_s8 = reinterpret_cast<int8_t*>(dst_data[io_dst::DST]) + dst_offset;
        const auto dst_f32 = reinterpret_cast<float*>(dst_data[io_dst::DST]) + dst_offset;
        const auto dst_bf16 = reinterpret_cast<bfloat16_t*>(dst_data[io_dst::DST]) + dst_offset;
        const auto dst_tmp = reinterpret_cast<float*>(workspace) + dst_offset;

        const auto q_scale = q_scale_num == 0 ? &def_scale : q_scale_f32 + (q_scale_num == 1 ? 0 : ibs * sl_m);
        const auto k_scale = k_scale_num == 0 ? &def_scale : k_scale_f32 + (k_scale_num == 1 ? 0 : ibs * sl_n);
        const auto v_scale = v_scale_num == 0 ? &def_scale : v_scale_f32 + (v_scale_num == 1 ? 0 : ibs * sl_n);
        const auto dst_scale = dst_scale_num == 0 ? &def_scale : dst_scale_f32 + (dst_scale_num == 1 ? 0 : ibs * sl_m);
        const auto dst_zp = dst_zp_num == 0 ? &def_zp : dst_zp_s32 + (dst_zp_num == 1 ? 0 : ibs * sl_m);

        const auto att_scale_idx = ts_descs_[io::ATT_SCALE].size() <= 1 ? 0 : ibs;
        const float att_scale = reinterpret_cast<const float*>(src_data[io_src::ATT_SCALE])[att_scale_idx];

        const int badd_offset = ibs * badd_step[0] + ihn * badd_step[1] + i * badd_step[2];
        const auto badd_f32 =
            has_badd ? reinterpret_cast<const float*>(src_data[io_src::BINARY_ADD]) + badd_offset : nullptr;
        const auto exp_row = std::unique_ptr<float[]>(new float[curr_sl_n]);

        // Q x K
        float tmp_max = -INFINITY;
        for (int j = 0; j < curr_sl_n; ++j) {
          if (ts_descs_[io::SRC_Q].dtype() == data_type::s8 && ts_descs_[io::SRC_K].dtype() == data_type::s8) {
            int32_t value = 0;
#pragma omp simd
            for (int k = 0; k < head_size; ++k) value += q_s8[i * ld_q + k] * k_s8[j * ld_kv + k];
            exp_row[j] = value;
          } else if (ts_descs_[io::SRC_Q].dtype() == data_type::bf16 &&
                     ts_descs_[io::SRC_K].dtype() == data_type::bf16) {
            float value = 0;
#pragma omp simd
            for (int k = 0; k < head_size; ++k)
              value += static_cast<float>(q_bf16[i * ld_q + k]) * static_cast<float>(k_bf16[j * ld_kv + k]);
            exp_row[j] = value;
          } else {
            SPARSE_LOG(FATAL) << "Unexpected Q K type!";
          }
          exp_row[j] *= q_scale[q_scale_num > 1 ? i : 0] * k_scale[q_scale_num > 1 ? j : 0] * att_scale;
          if (has_badd) exp_row[j] += badd_f32[j * badd_step[3]];
          tmp_max = std::max(tmp_max, exp_row[j]);
        }

        // softmax
        float exp_row_sum = 0;
        float exp_row_max = 0;
        for (int j = 0; j < curr_sl_n; ++j) {
          exp_row[j] = func_exp(exp_row[j] - (stable_softmax ? tmp_max : 0));
          exp_row_sum += exp_row[j];
          exp_row_max = std::max(exp_row_max, exp_row[j]);
        }
        const auto exp_sum_rcp = 1.f / exp_row_sum;
        const auto softmax_rescale = softmax_rescale_ > 0 ? softmax_rescale_ : 255.f / exp_row_max * exp_row_sum;
#pragma omp simd
        for (int j = 0; j < curr_sl_n; ++j) {
          // round to nearest when not accurate
          auto&& a_val = exp_row[j] * exp_sum_rcp * softmax_rescale;
          exp_row[j] = ts_descs_[io::SRC_V].dtype() == data_type::s8 ? post_softmax<func_exp>(a_val) : a_val;
        }

        // A x V
        for (int j = 0; j < head_size; ++j) {
          std::unique_ptr<int8_t[]> v_requant_s8(nullptr);
          float curr_v_scale = 1.f;
          const bool use_requant_v = ts_descs_[io::SRC_V].dtype() == data_type::s8 && ts_descs_[io::V_SCALE].size() > 1;
          if (use_requant_v) {
            // re-quant v to simulate the AMX kernel
            const auto v_f32 = std::unique_ptr<float[]>(new float[curr_sl_n]);
#pragma omp simd
            for (int k = 0; k < curr_sl_n; ++k) v_f32[k] = v_scale[k] * v_s8[k * ld_kv + j];
            v_requant_s8.reset(new int8_t[curr_sl_n]);
            curr_v_scale = quant_s8(v_f32.get(), curr_sl_n, v_requant_s8.get());
          } else if (v_scale_num <= 1) {
            curr_v_scale = v_scale[0];
          }

          float value = 0.f;
#pragma omp simd
          for (int k = 0; k < curr_sl_n; ++k) {
            auto v_val =  //
                ts_descs_[io::SRC_V].dtype() == data_type::s8 ? (use_requant_v ? v_requant_s8[k] : v_s8[k * ld_kv + j])
                : ts_descs_[io::SRC_V].dtype() == data_type::bf16 ? static_cast<float>(v_bf16[k * ld_kv + j])
                                                                  : (SPARSE_LOG(FATAL) << "Unexpected V type", NAN);
            value += exp_row[k] * v_val;
          }
          value *= curr_v_scale / softmax_rescale;
          if (ts_descs_[io::SRC_DST_SCALE].ftype() != format_type::undef) value /= dst_scale[dst_scale_num > 1 ? i : 0];
          if (ts_descs_[io::SRC_DST_ZP].ftype() != format_type::undef) value += dst_zp[dst_zp_num > 1 ? i : 0];

          const auto idx = i * ld_dst + j;
          if (is_dynq10n_dst) {
            dst_tmp[idx] = value;
            continue;
          }
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
            case data_type::bf16:
              dst_bf16[idx] = value;
              break;
            default:
              SPARSE_LOG(FATAL) << "Unexpected dst type!";
              break;
          }
        }
      }
    }
  }
  if (!is_dynq10n_dst) return true;

#pragma omp parallel for collapse(2)
  // dynamic quant dst
  for (int ibs = 0; ibs < bs; ibs++) {
    for (int i = 0; i < sl_m; ++i) {
      const int dst_offset = (ibs * sl_m + i) * ld_dst;
      const auto dst_tmp = reinterpret_cast<const float*>(workspace) + dst_offset;
      const auto dst_s8 = reinterpret_cast<int8_t*>(const_cast<void*>(dst_data[io_dst::DST])) + dst_offset;
      auto&& dst_scale = reinterpret_cast<float*>(const_cast<void*>(dst_data[io_dst::DST_SCALE]))[ibs * sl_m + i];
      switch (dst_dt_) {
        case data_type::s8:
          dst_scale = quant_s8(dst_tmp, ld_dst, dst_s8);
          break;
        default:
          SPARSE_LOG(FATAL) << "Unexpected dst type!";
          break;
      }
    }
  }
  return true;
}

bool mha_dense_ref_k_t::execute(const exec_context_t& ctx) const {
  return approx_exp ? execute_<&exp_2nd>(ctx) : execute_<&std::exp>(ctx);
}

bool mha_dense_ref_k_t::execute(const std::vector<const void*>& rt_data) const {
  return execute(*get_mha_dense_ctx(rt_data));
}
}  // namespace jd
