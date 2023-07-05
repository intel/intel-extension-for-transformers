//  Copyright (c) 2023 Intel Corporation
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

#include "dynamic_quant_mha.hpp"

#include <algorithm>

namespace jd {

#define KERNEL_INIT_CHECK(f)                                            \
  if (!(f)) {                                                           \
    SPARSE_LOG(ERROR) << "Dynamic q10n kernel requires `" << #f << "`"; \
    return false;                                                       \
  }
bool dynamic_quant_mha_kd_t::init() {
  if (!isa_available(amx_int8)) return false;

  const auto& op_attrs = op_desc_.attrs();
  KERNEL_INIT_CHECK(op_attrs.find("approx_exp") != op_attrs.end() && op_attrs.at("approx_exp") == "True");
  KERNEL_INIT_CHECK(op_attrs.find("stable_softmax") != op_attrs.end() && op_attrs.at("stable_softmax") == "False");

  const auto shapes = op_desc_.tensor_shapes();
  const auto dtypes = op_desc_.tensor_dtypes();

  const auto batch_size = shapes[io::SRC_Q][0];
  const auto head_num = shapes[io::SRC_Q][2];
  const auto M = shapes[io::SRC_Q][1];
  const auto head_size = shapes[io::SRC_Q][3];
  const auto N = shapes[io::SRC_K][1];

  KERNEL_INIT_CHECK((batch_size > 0 || shapes[io::BATCH_SIZE] == std::vector<dim_t>{1}));
  KERNEL_INIT_CHECK((head_num > 0 || shapes[io::HEAD_NUM] == std::vector<dim_t>{1}));
  KERNEL_INIT_CHECK((M > 0 || shapes[io::HEAD_SIZE] == std::vector<dim_t>{1}));
  KERNEL_INIT_CHECK((head_size > 0 || shapes[io::M] == std::vector<dim_t>{1}));
  KERNEL_INIT_CHECK((N > 0 || shapes[io::N] == std::vector<dim_t>{1}));

  KERNEL_INIT_CHECK((shapes[io::SRC_Q] == std::vector<dim_t>{batch_size, M, head_num, head_size}));
  KERNEL_INIT_CHECK((shapes[io::SRC_K] == std::vector<dim_t>{batch_size, N, head_num, head_size}));
  KERNEL_INIT_CHECK((shapes[io::SRC_V] == std::vector<dim_t>{batch_size, N, head_num, head_size}));
  KERNEL_INIT_CHECK((shapes[io::DST] == std::vector<dim_t>{batch_size, M, head_num, head_size}));
  KERNEL_INIT_CHECK((shapes[io::BINARY_ADD].size() == 0 ||  //
                     shapes[io::BINARY_ADD] == std::vector<dim_t>{batch_size, 1, 1, N}));

  KERNEL_INIT_CHECK((shapes[io::ATT_SCALE].empty() || shapes[io::ATT_SCALE] == std::vector<dim_t>{1}));

  KERNEL_INIT_CHECK((shapes[io::Q_SCALE] == std::vector<dim_t>{batch_size, M}));
  KERNEL_INIT_CHECK((shapes[io::K_SCALE] == std::vector<dim_t>{batch_size, N}));
  KERNEL_INIT_CHECK((shapes[io::V_SCALE] == std::vector<dim_t>{batch_size, N}));
  KERNEL_INIT_CHECK((shapes[io::DST_SCALE] == std::vector<dim_t>{batch_size, M}));

  // currently only support s8
  KERNEL_INIT_CHECK((shapes[io::Q_ZP].empty()));
  KERNEL_INIT_CHECK((shapes[io::K_ZP].empty()));
  KERNEL_INIT_CHECK((shapes[io::V_ZP].empty()));
  KERNEL_INIT_CHECK((shapes[io::DST_ZP].empty()));
  KERNEL_INIT_CHECK((shapes[io::SRC_DST_SCALE].empty()));  // static prechannel dst scale
  KERNEL_INIT_CHECK((shapes[io::SRC_DST_ZP].empty()));     // static prechannel dst zp

  // dtype
  KERNEL_INIT_CHECK(is_all_of(
      {
          dtypes[io::SRC_Q],
          dtypes[io::SRC_K],
          dtypes[io::SRC_V],
          dtypes[io::DST],
      },
      [&](const data_type t) { return t == data_type::s8; }));
  KERNEL_INIT_CHECK(is_all_of(
      {
          dtypes[io::Q_SCALE],
          dtypes[io::K_SCALE],
          dtypes[io::V_SCALE],
          dtypes[io::DST_SCALE],
      },
      [&](const data_type t) { return t == data_type::fp32; }));
  KERNEL_INIT_CHECK(shapes[io::ATT_SCALE].size() == 0 || dtypes[io::ATT_SCALE] == data_type::fp32);
  KERNEL_INIT_CHECK(shapes[io::BINARY_ADD].size() == 0 || dtypes[io::BINARY_ADD] == data_type::fp32);

  return true;
}
#undef KERNEL_INIT_CHECK

dynamic_quant_mha_k_t::dynamic_quant_mha_k_t(const std::shared_ptr<const kernel_desc_t>& kd)
    : kernel_t(kd),
      t_shapes_(derived_kd()->get_operator_desc().tensor_shapes()),
      batch_size_(t_shapes_[io::SRC_Q][0]),
      head_num_(t_shapes_[io::SRC_Q][2]),
      M_(t_shapes_[io::SRC_Q][1]),
      head_size_(t_shapes_[io::SRC_Q][3]),
      N_(t_shapes_[io::SRC_K][1]),
      has_attscale(t_shapes_[io::ATT_SCALE].size() != 0),
      has_badd(t_shapes_[io::BINARY_ADD].size() != 0),
      amx_full_tile_param_(16, 16, 64, false, 4),
      amx_full_tile_cfg_(amx_full_tile_param_) {}

bool dynamic_quant_mha_k_t::init() {
  if (!ker_amx_cfg_.create_kernel()) return false;
  if (!ker_amx_rls_.create_kernel()) return false;

  ker_seq_cpy_k_.reset(new jit_trans_AB16a4b_16x({/*.pad_n = */ 64, false, 1}));
  if (!ker_seq_cpy_k_->create_kernel()) return false;
  ker_seq_cpy_v_.reset(new jit_trans_BA16b4a_trq10n_x16());
  if (!ker_seq_cpy_v_->create_kernel()) return false;
  ker_qxk_.reset(new jit_mmexp_amx_s8_ab_BA16b4a_u8_16x(
      {/*.has_bias = */ has_badd || N_ % 16 != 0, /*.K = */ head_size_, /*.pre_amx_cfg = */ &amx_full_tile_param_}));
  if (!ker_qxk_->create_kernel()) return false;
  ker_axv_.reset(new jit_scale_mm_amx_u8s8_ab_BA16b_16x({/*.pre_amx_cfg = */ &amx_full_tile_param_}));
  if (!ker_axv_->create_kernel()) return false;
  ker_quant_.reset(new jit_dynamic_quant_t(  //
      dynamic_quant_param_t{
          /* .input_dt = */ data_type::fp32,
          /* .output_dt = */ data_type::s8,
          /* .quantized_dim_elt_num = */ static_cast<size_t>(head_size_),
          /* .ld_src = */ static_cast<size_t>(pad_to(head_size_, 16)),
          /* .ld_dst = */ static_cast<size_t>(head_size_ * head_num_),
      },
      16, true));
  if (!ker_quant_->create_kernel()) return false;

  return true;
}

size_t dynamic_quant_mha_k_t::get_workspace_size() const {
  return sizeof(float) * batch_size_ * pad_to(N_, 64) +                                              // mask
         sizeof(int8_t) * batch_size_ * head_num_ * (pad_to(N_, 64) * pad_to(head_size_, 64)) * 2 +  // K & V
         sizeof(float) * batch_size_ * head_num_ * pad_to(head_size_, 64) +                          // v scale
         sizeof(float) * omp_get_max_threads() * 16 *
             (pad_to(N_, 64) + head_num_ * pad_to(head_size_, 16));  // exp &  dst
}

#ifdef __AVX512F__
bool dynamic_quant_mha_k_t::execute(const std::vector<const void*>& rt_data) const {
  const auto max_threads = omp_get_max_threads();
  const auto src_q = reinterpret_cast<const int8_t*>(rt_data[io::SRC_Q]);
  const auto src_k = reinterpret_cast<const int8_t*>(rt_data[io::SRC_K]);
  const auto mask = reinterpret_cast<const float*>(rt_data[io::BINARY_ADD]);
  const auto src_v = reinterpret_cast<const int8_t*>(rt_data[io::SRC_V]);
  const auto dst = reinterpret_cast<int8_t*>(const_cast<void*>(rt_data[io::DST]));
  const auto workspace = reinterpret_cast<char*>(const_cast<void*>(rt_data[io::WORKSPACE]));
  const auto q_scale = reinterpret_cast<const float*>(rt_data[io::Q_SCALE]);
  const auto k_scale = reinterpret_cast<const float*>(rt_data[io::K_SCALE]);
  const auto v_scale = reinterpret_cast<const float*>(rt_data[io::V_SCALE]);
  const auto dst_scale = reinterpret_cast<float*>(const_cast<void*>(rt_data[io::DST_SCALE]));

  const auto batch_size = batch_size_ > 0 ? batch_size_ : reinterpret_cast<const int32_t*>(rt_data[io::BATCH_SIZE])[0];
  const auto head_num = head_num_ > 0 ? head_num_ : reinterpret_cast<const int32_t*>(rt_data[io::HEAD_NUM])[0];
  const auto head_size = head_size_ > 0 ? head_size_ : reinterpret_cast<const int32_t*>(rt_data[io::HEAD_SIZE])[0];
  const auto M = M_ > 0 ? M_ : reinterpret_cast<const int32_t*>(rt_data[io::M])[0];
  const auto N = N_ > 0 ? N_ : reinterpret_cast<const int32_t*>(rt_data[io::N])[0];
  const auto att_scale = has_attscale ? reinterpret_cast<const float*>(rt_data[io::ATT_SCALE])[0] : 1.f;

  const auto head_size_pad16 = pad_to(head_size, 16);
  const auto head_size_pad64 = pad_to(head_size, 64);
  const auto N_pad4 = pad_to(N, 4);
  const auto N_pad16 = pad_to(N, 16);
  const auto N_pad64 = pad_to(N, 64);
  const auto size_trq10n_v_block = 16 * N_pad4;
  const auto size_pad0_v_block = N_pad64 * 16 - size_trq10n_v_block;

  const auto tmp_mask_size = batch_size * N_pad16;
  const auto head_tmp_k_size = head_size_pad64 * N_pad16;
  const auto head_tmp_v_size = N_pad64 * head_size_pad16;
  const auto head_tmp_v_scale_size = head_size_pad16;
  const auto tmp_thread_size =
      16 * N_pad64 + 16 * head_num * head_size_pad16;  // for softmax result & axv un-quant f32 result

  auto tmp_mask = reinterpret_cast<float*>(workspace);
  const auto tmp_k = reinterpret_cast<int8_t*>(tmp_mask + tmp_mask_size);
  const auto tmp_v = reinterpret_cast<int8_t*>(tmp_k + batch_size * head_num * head_tmp_k_size);
  const auto tmp_v_scale = reinterpret_cast<float*>(tmp_v + batch_size * head_num * head_tmp_v_size);
  const auto tmp_threads = reinterpret_cast<float*>(tmp_v_scale + batch_size * head_num * head_tmp_v_scale_size);
  // const auto tmp_end = reinterpret_cast<char*>(tmp_threads + max_threads * tmp_thread_size);
  // assert(workspace + 16 * 1024 * 1204 < tmp_end);

#pragma omp parallel for collapse(2)
  for (int ibs = 0; ibs < batch_size; ++ibs) {
    for (int ihn = 0; ihn < head_num; ++ihn) {
      // transpose K
      const auto curr_k = src_k + ibs * N * head_size * head_num + ihn * head_size;
      const auto curr_tmp_k = tmp_k + (ibs * head_num + ihn) * head_tmp_k_size;
      for (int j = 0; j < N; j += 16) {
        jit_trans_AB16a4b_16x::rt_data_t tr_k_data{
            /*.src = */ curr_k + j * (head_size * head_num),
            /*.dst = */ curr_tmp_k + j * head_size_pad64,
            /*.ld_src = */ head_size * head_num,
            /*.M = */ std::min(16, N - j),
            /*.N = */ head_size,
        };
        (*ker_seq_cpy_k_)(&tr_k_data);
      }

      // transpose V
      const auto curr_v = src_v + ibs * N * head_size * head_num + ihn * head_size;
      const auto curr_v_scale = v_scale + ibs * N;
      const auto curr_tmp_v = tmp_v + (ibs * head_num + ihn) * head_tmp_v_size;
      const auto curr_tmp_v_scale = tmp_v_scale + (ibs * head_num + ihn) * head_tmp_v_scale_size;
      for (int j = 0; j < head_size; j += 16) {
        const auto block_tmp_v = curr_tmp_v + j * N_pad64;
        jit_trans_BA16b4a_trq10n_x16::rt_data_t tr_v_data{
            /*.src = */ curr_v + j,
            /*.dst = */ block_tmp_v,
            /*.src_scale = */ curr_v_scale,
            /*.dst_scale = */ curr_tmp_v_scale + j,
            /*.ld_src = */ head_size * head_num,
            /*.M = */ N,
            /*.N = */ std::min(16, head_size - j),
        };
        (*ker_seq_cpy_v_)(&tr_v_data);
        // fill 0 for padding of AxV
        if (size_pad0_v_block) std::fill_n(block_tmp_v + size_trq10n_v_block, size_pad0_v_block, 0);
      }
    }
  }

  if (N != N_pad16) {
    for (int ibs = 0; ibs < batch_size; ++ibs) {
      const auto curr_tmp_mask = tmp_mask + ibs * N_pad16;
      if (has_badd) {
        std::copy_n(mask + ibs * N, N, curr_tmp_mask);
      } else {
        std::fill_n(curr_tmp_mask, N, 0.f);
      }
      std::fill_n(curr_tmp_mask + N, N_pad16 - N, -1000.f);
    }
  } else {
    tmp_mask = const_cast<float*>(mask);
  }
  bool amx_init[256];
  std::fill_n(amx_init, max_threads, false);
#pragma omp parallel for collapse(2)
  for (int ibs = 0; ibs < batch_size; ++ibs) {
    for (int i = 0; i < M; i += 16) {
      if (!amx_init[omp_get_thread_num()]) {
        ker_amx_cfg_(&amx_full_tile_cfg_);
        amx_init[omp_get_thread_num()] = true;
      }

      const auto curr_tmp_exp =
          tmp_threads + omp_get_thread_num() * tmp_thread_size;  // Of size `headsize x 16 x n_pad64`
      const auto curr_tmp_dst = curr_tmp_exp + 16 * N_pad64;
      alignas(64) float absmax_dst[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      for (int ihn = 0; ihn < head_num; ++ihn) {
        const auto curr_q = src_q + (ibs * M + i) * head_size * head_num + head_size * ihn;
        const auto curr_q_scale = q_scale + ibs * M + i;
        const auto curr_tmp_k = tmp_k + (ibs * head_num + ihn) * head_tmp_k_size;
        const auto curr_k_scale = k_scale + ibs * N;
        const auto curr_mask = tmp_mask + ibs * N_pad16;
        alignas(64) float exp_sum[16];
        alignas(64) float exp_max[16];
        {  //  MMQK + softmax + 0-255quant
          jit_mmexp_amx_s8_ab_BA16b4a_u8_16x::rt_data_t mm_qk_data{
              /*.src0 = */ curr_q,
              /*.src1 = */ curr_tmp_k,
              /*.scale_src0 = */ curr_q_scale,
              /*.scale_src1 = */ curr_k_scale,
              /*.src_bias = */ curr_mask,  // May not used if !has_badd && N %16 == 0
              /*.dst = */ curr_tmp_exp,
              /*.dst_sum = */ exp_sum,
              /*.dst_max = */ exp_max,
              /*.scale = */ att_scale,
              /*.K = */ head_size,
              /*.N = */ N,
              /*.ld_src0 = */ head_size * head_num,
              /*.ld_src1 = */ head_size_pad64 * 16,
          };
          (*ker_qxk_)(&mm_qk_data);
        }

        alignas(64) float act_prescale[16];
#pragma omp simd
        for (int ii = 0; ii < 16; ++ii) act_prescale[ii] = 255.f / exp_max[ii];
// use exp_sum as dst scale
#pragma omp simd
        for (int ii = 0; ii < 16; ++ii) exp_sum[ii] = 1.f / 255.f * exp_max[ii] / exp_sum[ii];

        const auto curr_tmp_v = tmp_v + (ibs * head_num + ihn) * head_tmp_v_size;
        const auto curr_tmp_v_scale = tmp_v_scale + (ibs * head_num + ihn) * head_tmp_v_scale_size;
        {  // MMAV + perchannel-q10n
          jit_scale_mm_amx_u8s8_ab_BA16b_16x::rt_data_t mm_av_data{
              /*.src0 = */ curr_tmp_exp,
              /*.prescale_src0 = */ act_prescale,
              /*.scale_src0 = */ exp_sum,
              /*.src1 = */ curr_tmp_v,
              /*.scale_src1 = */ curr_tmp_v_scale,
              /*.dst = */ curr_tmp_dst + 16 * head_size_pad16 * ihn,
              /*.absmax_dst = */ absmax_dst,
              /*.K = */ N,
              /*.N = */ head_size,
              /*.ld_src1 = */ N_pad64 * 16,
              /*.ld_dst = */ head_size_pad16,
          };
          (*ker_axv_)(&mm_av_data);
        }
      }
      {
        constexpr float epsilon = 1e-9f;
        alignas(64) float rcp_scale[16];
        const auto ld_dst = head_size * head_num;
        const auto curr_dst = dst + (ibs * M + i) * ld_dst;
        const auto curr_dst_scale = dst_scale + ibs * M + i;
#pragma omp simd
        for (int ii = 0; ii < 16; ++ii) {
          curr_dst_scale[ii] = absmax_dst[ii] / 127.f;
          rcp_scale[ii] = 127.f / (absmax_dst[ii] + epsilon);
        }
        for (int ihn = 0; ihn < head_num; ++ihn) {
          dynamic_quant_data_t quant_data{
              /* .src = */ curr_tmp_dst + ihn * 16 * head_size_pad16,
              /* .mat_dst = */ curr_dst + ihn * head_size,
              /* .scale = */ rcp_scale,
          };
          (*ker_quant_)(&quant_data);
        }
      }
    }
  }

  // #pragma omp parallel for  // release amx
  //   for (int i = 0; i < max_threads; ++i) ker_amx_rls_.tile_release();

  return true;
}
#else
bool dynamic_quant_mha_k_t::execute(const std::vector<const void*>&) const {
  SPARSE_LOG(ERROR) << "dynamic_quant_mha execute for AVX2 not implemented!";
  return false;
}
#endif

}  // namespace jd
