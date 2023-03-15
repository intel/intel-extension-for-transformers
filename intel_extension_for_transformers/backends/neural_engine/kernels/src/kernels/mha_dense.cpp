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

#include "kernels/mha_dense.hpp"

#include <algorithm>

#define KERNEL_INIT_CHECK(f)                                         \
  if (!(f)) {                                                        \
    SPARSE_LOG(ERROR) << "Attention kernel requires `" << #f << "`"; \
    return false;                                                    \
  }
namespace jd {

bool mha_dense_kd_t::init() {
  if (!isa_available(avx512_core_amx)) return false;
  auto op_attrs = op_desc_.attrs();
  param_.QK_rescale_ = str_to_num<float>(op_attrs["QK_rescale"]);
  param_.softmax_rescale_ = str_to_num<float>(op_attrs["softmax_rescale"]);
  param_.QKV_rescale_ = str_to_num<float>(op_attrs["QKV_rescale"]);
  param_.QKV_dstzp_ = str_to_num<float>(op_attrs["QKV_dstzp"]);
  param_.Q_scale_ = str_to_num<float>(op_attrs["Q_scale"]);
  param_.K_scale_ = str_to_num<float>(op_attrs["K_scale"]);
  param_.V_scale_ = str_to_num<float>(op_attrs["V_scale"]);
  param_.DST_scale_ = str_to_num<float>(op_attrs["DST_scale"]);
  param_.QK_output_scale_ = str_to_num<float>(op_attrs["QK_output_scale"]);
  if (op_attrs.find("merged_QKV") != op_attrs.end() && op_attrs["merged_QKV"] == "True")
    param_.base_ = 3;
  else
    param_.base_ = 1;

  auto& tensor_desc = op_desc_.tensor_descs();
  auto& src_shape = tensor_desc[io::SRC_Q].shape();
  auto& dst_shape = tensor_desc[io::DST].shape();
  KERNEL_INIT_CHECK(src_shape == tensor_desc[io::SRC_K].shape())
  KERNEL_INIT_CHECK(src_shape == tensor_desc[io::SRC_V].shape())
  KERNEL_INIT_CHECK(src_shape == dst_shape)

  param_.dst_dt_ = tensor_desc[io::DST].dtype();
  param_.src_bs_ = src_shape[0];
  param_.src_seq_len_ = src_shape[1];
  param_.head_num_ = src_shape[2];
  param_.head_size_ = src_shape[3];

  KERNEL_INIT_CHECK(param_.head_size_ == 32 || param_.head_size_ == 64 || param_.head_size_ % 64 == 0)

  const auto has_badd = has_binary_add();
  if (has_badd) {
    // only amx impl supports attention mask (binary add before softmax)
    KERNEL_INIT_CHECK(isa_available(avx512_core_amx))

    const auto& badd_shape = tensor_desc[io::BINARY_ADD].shape();
    KERNEL_INIT_CHECK(tensor_desc[io::BINARY_ADD].dtype() == data_type::fp32);
    KERNEL_INIT_CHECK(param_.src_seq_len_ == badd_shape[badd_shape.size() - 1]);
    KERNEL_INIT_CHECK(badd_shape.size() < 2 || param_.src_seq_len_ == badd_shape[badd_shape.size() - 2]);
    KERNEL_INIT_CHECK(badd_shape.size() < 3 || param_.head_num_ == badd_shape[badd_shape.size() - 3]);
    KERNEL_INIT_CHECK(badd_shape.size() < 4 || param_.src_bs_ == badd_shape[badd_shape.size() - 4]);
  }
  return true;
}

mha_dense_k_t::mha_dense_k_t(const std::shared_ptr<const kernel_desc_t>& kd)
    : kernel_t(kd),
      dst_dt_(derived_kd()->params().dst_dt_),
      src_bs_(derived_kd()->params().src_bs_),
      src_seq_len_(derived_kd()->params().src_seq_len_),
      head_num_(derived_kd()->params().head_num_),
      head_size_(derived_kd()->params().head_size_),
      ld_src_(head_size_ * head_num_ * derived_kd()->params().base_),
      ld_dst_(head_size_ * head_num_ * get_data_size(dst_dt_)),
      softmax_rescale_(fp32_to_fp16(derived_kd()->params().softmax_rescale_)),
      QK_rescale_(derived_kd()->params().QK_rescale_),
      QKV_rescale_(derived_kd()->params().QKV_rescale_),
      QKV_dstzp_(derived_kd()->params().QKV_dstzp_),
      Q_scale_(derived_kd()->params().Q_scale_),
      K_scale_(derived_kd()->params().K_scale_),
      V_scale_(derived_kd()->params().V_scale_),
      DST_scale_(derived_kd()->params().DST_scale_),
      QK_output_scale_(derived_kd()->params().QK_output_scale_),
      ts_descs_(derived_kd()->get_operator_desc().tensor_descs()),
      has_binary_add(derived_kd()->has_binary_add()),
      thread_workspace_size_(sizeof(int8_t) * head_size_ * pad_to(src_seq_len_, 16) +  // k
                             sizeof(int8_t) * head_size_ * pad_to(src_seq_len_, 64) +  // v
                             sizeof(int32_t) * 32 * pad_to(src_seq_len_, 16) +         // qk
                             sizeof(uint8_t) * 32 * pad_to(src_seq_len_, 64)),         // softmax
      amx_full_tile_param_(16, 16, 64, false, 4),
      amx_full_tile_cfg_(amx_full_tile_param_) {
#define SET_NULL(arr) std::fill_n(reinterpret_cast<void**>(arr), sizeof(arr) / sizeof(nullptr), nullptr)
  SET_NULL(ker_trans_k_);
  SET_NULL(ker_trans_v_);
  SET_NULL(ker_softmax_);
  SET_NULL(ker_qk_gemm_16x_);
  SET_NULL(ker_qk_gemm_32x_);
  SET_NULL(ker_av_gemm_16x_);
  SET_NULL(ker_av_gemm_32x_);
#undef SET_NULL
}

bool mha_dense_k_t::init() {
  if (!ker_amx_cfg_.create_kernel()) return false;
  if (!ker_amx_rls_.create_kernel()) return false;

  const auto hs_max64 = std::min(head_size_, 64);

  // init ker_trans_k
  for (int i = 1; i <= 16; ++i) {
    ker_trans_k_[i] = new jit_trans_AB16a4b(
        {/*.M = */ i, /*.N = */ hs_max64, /*.ld_src = */ ld_src_, /*.pad_n = */ pad_to(hs_max64, 16)});
    if (!ker_trans_k_[i]->create_kernel()) return false;
  }

  // init ker_qk_gemm
  for (int i = 16; i <= MAX_SEQLEN; i += 16) {
    ker_qk_gemm_32x_[i / 16] =
        new jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b(jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b::param_t{
            32,
            head_size_,
            i,
            ld_src_,
            &amx_full_tile_param_,
        });
    if (!ker_qk_gemm_32x_[i / 16]->create_kernel()) return false;
    ker_qk_gemm_16x_[i / 16] =
        new jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b(jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b::param_t{
            16,
            head_size_,
            i,
            ld_src_,
            &amx_full_tile_param_,
        });
    if (!ker_qk_gemm_16x_[i / 16]->create_kernel()) return false;
  }

  // init softmax
  for (int i = 64; i <= MAX_SEQLEN; i += 64) {
    for (int j = 0; j < 16; j++) {
      ker_softmax_[i / 64][j] = new jit_softmax_Ab16a({j, i, has_binary_add, "u8"});
      if (!ker_softmax_[i / 64][j]->create_kernel()) return false;
    }
  }

  // init ker_trans_v
  for (int i = 0; i <= 4; i++) {
    ker_trans_v_[i] = new jit_trans_BA16b4a({/*.M = */ i, /*.N = */ hs_max64, /*.ld_src = */ ld_src_});
    if (!ker_trans_v_[i]->create_kernel()) return false;
  }

  // init ker_av_gemm
  const int ld_dst = head_size_ * head_num_;
  for (int i = 64; i <= MAX_SEQLEN; i += 64) {
    ker_av_gemm_32x_[i / 64] = new jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab(
        {/*.M = */ 32, /*.K_pad = */ i, /*.N = */ head_size_, /*.ld_dst = */ ld_dst, /*.dst_dt = */ dst_dt_});
    if (!ker_av_gemm_32x_[i / 64]->create_kernel()) return false;
    ker_av_gemm_16x_[i / 64] = new jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab(
        {/*.M = */ 16, /*.K_pad = */ i, /*.N = */ head_size_, /*.ld_dst = */ ld_dst, /*.dst_dt = */ dst_dt_});
    if (!ker_av_gemm_16x_[i / 64]->create_kernel()) return false;
  }
  return true;
}

inline void mha_dense_k_t::mha_per_head_32x(const jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b::rt_data_t& rt_data_qk,
                                            const jit_softmax_Ab16a::rt_data_t& rt_data_softmax1,
                                            const jit_softmax_Ab16a::rt_data_t& rt_data_softmax2,
                                            const jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab::rt_data_t& rt_data_av,
                                            const int att_tail, const int col_tile, const int att_tile) const {
  (*ker_qk_gemm_32x_[col_tile])(&rt_data_qk);                // matmul QxK
  (*(ker_softmax_[att_tile][att_tail]))(&rt_data_softmax1);  // softmax 1
  (*(ker_softmax_[att_tile][att_tail]))(&rt_data_softmax2);  // softmax 2
  (*ker_av_gemm_32x_[att_tile])(&rt_data_av);                // matmul AxV
}

inline void mha_dense_k_t::mha_per_head_16x(const jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b::rt_data_t& rt_data_qk,
                                            const jit_softmax_Ab16a::rt_data_t& rt_data_softmax1,
                                            const jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab::rt_data_t& rt_data_av,
                                            const int att_tail, const int col_tile, const int att_tile) const {
  (*ker_qk_gemm_16x_[col_tile])(&rt_data_qk);                // matmul QxK
  (*(ker_softmax_[att_tile][att_tail]))(&rt_data_softmax1);  // softmax
  (*ker_av_gemm_16x_[att_tile])(&rt_data_av);                // matmul AxV
}

bool mha_dense_k_t::execute(const std::vector<const void*>& rt_data) const {
  const int32_t bs = src_bs_;
  n_thread_t with_n_thread(bs * head_num_, true);

  constexpr size_t MAX_BS = 128;
  int32_t sl_accumulate[MAX_BS];
  sl_accumulate[0] = 0;
  // #pragma omp parallel for  // usually bs is too small to use parallel
  for (int32_t i = 1; i < bs; i++) sl_accumulate[i] = src_seq_len_ * i;

  const int badd_stride[] = {
      !has_binary_add || ts_descs_[io::BINARY_ADD].shape().size() < 4 ? 0 : head_num_ * src_seq_len_ * src_seq_len_,
      !has_binary_add || ts_descs_[io::BINARY_ADD].shape().size() < 3 ? 0 : src_seq_len_ * src_seq_len_,
      !has_binary_add || ts_descs_[io::BINARY_ADD].shape().size() < 2 ? 0 : src_seq_len_,
      !has_binary_add || ts_descs_[io::BINARY_ADD].shape().size() < 1 ? 0 : 1,
  };

#pragma omp parallel for collapse(2)
  for (int ibs = 0; ibs < bs; ibs++) {
    for (int ihn = 0; ihn < head_num_; ihn++) {
      const int seq_len = src_seq_len_;
      const int sl_pad16 = pad_to(seq_len, 16);
      const int col_tile = ceil_div(seq_len, 16);
      const int row_loop = col_tile / 2;
      const bool is_even = (col_tile % 2 == 0);
      const int rollback = (seq_len % 16 != 0) ? 16 - (seq_len % 16) : 0;
      const int sl_pad64 = (seq_len + 63) / 64 * 64;
      const int att_tile = sl_pad64 / 64;

      const auto thread_id = omp_get_thread_num();
      const auto thread_workspace = reinterpret_cast<char*>(const_cast<void*>(rt_data[mha_dense_io::WORKSPACE])) +
                                    thread_id * thread_workspace_size_;

      const auto k_scrach = reinterpret_cast<int8_t*>(thread_workspace);
      const auto v_scrach_p64 = reinterpret_cast<int8_t*>(k_scrach + head_size_ * sl_pad16);
      const auto qk_scrach = reinterpret_cast<int32_t*>(v_scrach_p64 + head_size_ * sl_pad64);
      const auto softmax_scrach_p64 = reinterpret_cast<uint8_t*>(qk_scrach + 32 * sl_pad16);
      // softmax_scrach_p64_size = 32 * sl_pad64

      const int src_offset = sl_accumulate[ibs] * ld_src_ + ihn * head_size_;
      const int dst_offset = sl_accumulate[ibs] * ld_dst_ + ihn * head_size_ * get_data_size(dst_dt_);
      // init amx for each omp thread
      ker_amx_cfg_(&amx_full_tile_cfg_);

      const auto curr_q = reinterpret_cast<const int8_t*>(rt_data[mha_dense_io::SRC_Q]) + src_offset;
      const auto curr_dst = reinterpret_cast<char*>(const_cast<void*>(rt_data[mha_dense_io::DST])) + dst_offset;
      const auto curr_k = reinterpret_cast<const int8_t*>(rt_data[mha_dense_io::SRC_K]) + src_offset;
      const auto curr_v = reinterpret_cast<const int8_t*>(rt_data[mha_dense_io::SRC_V]) + src_offset;

      const int badd_offset = ibs * badd_stride[0] + ihn * badd_stride[1];
      const auto badd_f32 =
          has_binary_add ? reinterpret_cast<const float*>(rt_data[io::BINARY_ADD]) + badd_offset : nullptr;

      // reorder K
      for (int i = 0; i < seq_len; i += 16)
        for (int j = 0; j < head_size_; j += 64) {
          jit_trans_AB16a4b::rt_data_t rt_data_tr_k{
              /*.src = */ curr_k + i * ld_src_ + j,
              /*.dst = */ k_scrach + i * head_size_ + j * 16,
          };
          (*ker_trans_k_[std::min(16, seq_len - i)])(&rt_data_tr_k);
        }

      // reorder V
      const auto tr_v_dst_stride = jit_trans_BA16b4a::dst_stride(sl_pad64);
      for (int j = 0; j < head_size_; j += 64)
        for (int i = 0; i < sl_pad64; i += 4) {
          jit_trans_BA16b4a::rt_data_t rt_data_tr_v{
              /*.src = */ curr_v + i * ld_src_ + j,
              /*.dst = */ v_scrach_p64 + i * 16 + j * sl_pad64,
              /*.ld_dst = */ tr_v_dst_stride,
          };
          (*ker_trans_v_[std::max(0, std::min(4, seq_len - i))])(&rt_data_tr_v);
        }

      const auto padding_mask = reinterpret_cast<const int32_t*>(rt_data[mha_dense_io::MASK])[ibs];
      jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b::rt_data_t rt_data_qk{
          /*.src0 = */ nullptr,
          /*.src1 = */ k_scrach,
          /*.dst = */ qk_scrach,
      };
      jit_softmax_Ab16a::rt_data_t rt_data_softmax1{
          /*.src = */ qk_scrach,
          /*.dst = */ softmax_scrach_p64,
          /*.att_tile = */ padding_mask / 16,
          /*.softmax_rescale = */ softmax_rescale_,
          /*.src_badd = */ nullptr,
          /*.ld_badd = */ badd_stride[2],
          /*.QK_rescale = */ QK_rescale_,
      };
      jit_softmax_Ab16a::rt_data_t rt_data_softmax2{
          /*.src = */ qk_scrach + sl_pad16 * 16,           // sl_pad_ / 16 * 16 * 16
          /*.dst = */ softmax_scrach_p64 + sl_pad64 * 16,  // sl_pad64_ / 64 * 16 * 64
          /*.att_tile = */ padding_mask / 16,
          /*.softmax_rescale = */ softmax_rescale_,
          /*.src_badd = */ nullptr,
          /*.ld_badd = */ badd_stride[2],
          /*.QK_rescale = */ QK_rescale_,
      };
      jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab::rt_data_t rt_data_av{
          /*.src0 = */ softmax_scrach_p64,
          /*.src1 = */ v_scrach_p64,
          /*.dst = */ nullptr,
          /*.K = */ padding_mask,
          /*.rescale = */ QKV_rescale_,
          /*.zp = */ QKV_dstzp_,
      };
      const int att_tail = reinterpret_cast<const int32_t*>(rt_data[mha_dense_io::MASK])[ibs] % 16;
      int cur_r_pos = 0;

      for (int j = 0; j < row_loop - 1; j++, cur_r_pos += 32) {
        rt_data_qk.src0 = curr_q + cur_r_pos * ld_src_;
        rt_data_av.dst = curr_dst + cur_r_pos * ld_dst_;
        if (has_binary_add) rt_data_softmax1.src_badd = badd_f32 + cur_r_pos * badd_stride[2];
        if (has_binary_add) rt_data_softmax2.src_badd = badd_f32 + (cur_r_pos + 16) * badd_stride[2];
        mha_per_head_32x(rt_data_qk, rt_data_softmax1, rt_data_softmax2, rt_data_av, att_tail, col_tile, att_tile);
      }

      if (is_even) {
        if (rollback == 0) {
          rt_data_qk.src0 = curr_q + cur_r_pos * ld_src_;
          rt_data_av.dst = curr_dst + cur_r_pos * ld_dst_;
          if (has_binary_add) rt_data_softmax1.src_badd = badd_f32 + cur_r_pos * badd_stride[2];
          if (has_binary_add) rt_data_softmax2.src_badd = badd_f32 + (cur_r_pos + 16) * badd_stride[2];
          mha_per_head_32x(rt_data_qk, rt_data_softmax1, rt_data_softmax2, rt_data_av, att_tail, col_tile, att_tile);
        } else {
          rt_data_qk.src0 = curr_q + cur_r_pos * ld_src_;
          rt_data_av.dst = curr_dst + cur_r_pos * ld_dst_;
          if (has_binary_add) rt_data_softmax1.src_badd = badd_f32 + cur_r_pos * badd_stride[2];
          mha_per_head_16x(rt_data_qk, rt_data_softmax1, rt_data_av, att_tail, col_tile, att_tile);

          cur_r_pos += 16 - rollback;

          rt_data_qk.src0 = curr_q + cur_r_pos * ld_src_;
          rt_data_av.dst = curr_dst + cur_r_pos * ld_dst_;
          if (has_binary_add) rt_data_softmax1.src_badd = badd_f32 + cur_r_pos * badd_stride[2];
          mha_per_head_16x(rt_data_qk, rt_data_softmax1, rt_data_av, att_tail, col_tile, att_tile);
        }
      } else {
        rt_data_qk.src0 = curr_q + cur_r_pos * ld_src_;
        rt_data_av.dst = curr_dst + cur_r_pos * ld_dst_;
        if (has_binary_add) rt_data_softmax1.src_badd = badd_f32 + cur_r_pos * badd_stride[2];
        if (has_binary_add) rt_data_softmax2.src_badd = badd_f32 + (cur_r_pos + 16) * badd_stride[2];
        mha_per_head_32x(rt_data_qk, rt_data_softmax1, rt_data_softmax2, rt_data_av, att_tail, col_tile, att_tile);

        cur_r_pos += 32 - rollback;

        rt_data_qk.src0 = curr_q + cur_r_pos * ld_src_;
        rt_data_av.dst = curr_dst + cur_r_pos * ld_dst_;
        if (has_binary_add) rt_data_softmax1.src_badd = badd_f32 + cur_r_pos * badd_stride[2];
        mha_per_head_16x(rt_data_qk, rt_data_softmax1, rt_data_av, att_tail, col_tile, att_tile);
      }

      // release amx for each omp thread
      ker_amx_rls_.tile_release();
    }
  }

  return true;
}

}  // namespace jd
