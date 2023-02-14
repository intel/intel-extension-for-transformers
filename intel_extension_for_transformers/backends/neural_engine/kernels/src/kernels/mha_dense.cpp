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
#include "jit_domain/jit_mha_vnni.hpp"

#define KERNEL_INIT_CHECK(f)                                         \
  if (!(f)) {                                                        \
    SPARSE_LOG(ERROR) << "Attention kernel requires `" << #f << "`"; \
    return false;                                                    \
  }
namespace jd {

bool mha_dense_kd_t::init() {
  if (!isa_available(avx512_core_amx) && !isa_available(avx512_core_vnni)) return false;
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
  param_.is_package_ = (op_attrs.find("is_package") != op_attrs.end() && op_attrs["is_package"] == "True");

  auto& tensor_desc = op_desc_.tensor_descs();
  auto& src_shape = tensor_desc[mha_dense_io::SRC_Q].shape();
  auto& dst_shape = tensor_desc[mha_dense_io::DST].shape();
  KERNEL_INIT_CHECK(src_shape == tensor_desc[mha_dense_io::SRC_K].shape())
  KERNEL_INIT_CHECK(src_shape == tensor_desc[mha_dense_io::SRC_V].shape())
  KERNEL_INIT_CHECK(src_shape == dst_shape)
  KERNEL_INIT_CHECK(!param_.is_package_ ||
                    src_shape[0] == 1)  // the shape with package  is (1,sum_of_seq_len,head_num,head_size)

  param_.dst_dt_ = tensor_desc.back().dtype();
  param_.src_bs_ = src_shape[0];
  param_.src_seq_len_ = src_shape[1];
  param_.head_num_ = src_shape[2];
  param_.head_size_ = src_shape[3];
  param_.hidden_size_ = param_.head_num_ * param_.head_size_;
  return true;
}

mha_dense_k_t::mha_dense_k_t(const std::shared_ptr<const kernel_desc_t>& kd)
    : kernel_t(kd),
      is_package_(derived_kd()->params().is_package_),
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
  no_amx_ = !isa_available(avx512_core_amx);
}

bool mha_dense_k_t::init() {
  if (no_amx_) {
    return true;
  }
  if (!ker_amx_cfg_.create_kernel()) return false;
  if (!ker_amx_rls_.create_kernel()) return false;

  // init ker_trans_k
  for (int i = 1; i <= 16; ++i) {
    ker_trans_k_[i] = new jit_trans_AB16a4b({i, head_size_, ld_src_, pad_to(head_size_, 16)});
    if (!ker_trans_k_[i]->create_kernel()) return false;
  }

  // init ker_qk_gemm
  for (int i = 16; i <= 384; i += 16) {
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
  jit_softmax_Ab16a::params softmax_param;
  for (int i = 64; i <= 384; i += 64) {
    softmax_param.sl_pad64_ = i;
    softmax_param.output_type = "u8";
    for (int j = 0; j < 16; j++) {
      softmax_param.att_tail = j;
      ker_softmax_[i / 64][j] = new jit_softmax_Ab16a(softmax_param);
      if (!ker_softmax_[i / 64][j]->create_kernel()) return false;
    }
  }

  // init ker_trans_v
  for (int i = 0; i <= 4; i++) {
    ker_trans_v_[i] = new jit_trans_BA16b4a(jit_trans_BA16b4a::params{i, head_size_, ld_src_});
    if (!ker_trans_v_[i]->create_kernel()) return false;
  }

  // init ker_av_gemm
  const int ld_dst = head_size_ * head_num_;
  for (int i = 64; i <= 384; i += 64) {
    ker_av_gemm_32x_[i / 64] = new jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab(
        jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab::param_t{32, i, head_size_, ld_dst, dst_dt_});
    if (!ker_av_gemm_32x_[i / 64]->create_kernel()) return false;
    ker_av_gemm_16x_[i / 64] = new jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab(
        jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab::param_t{16, i, head_size_, ld_dst, dst_dt_});
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
  if (no_amx_) {
    jit_mha_vnni_pad32::forward32_bf16_vnni_NT(
        reinterpret_cast<const int8_t*>(rt_data[mha_dense_io::SRC_Q]),
        reinterpret_cast<const int8_t*>(rt_data[mha_dense_io::SRC_K]),
        reinterpret_cast<const int32_t*>(rt_data[mha_dense_io::MASK]),
        reinterpret_cast<const int8_t*>(rt_data[mha_dense_io::SRC_V]),
        reinterpret_cast<uint8_t*>(const_cast<void*>(rt_data[mha_dense_io::DST])),
        *reinterpret_cast<const int32_t*>(rt_data[mha_dense_io::BS]), head_num_, head_size_, src_seq_len_,
        1.f / Q_scale_, 1.f / K_scale_, 1.f / V_scale_, 1.f / DST_scale_, QKV_dstzp_, QK_output_scale_);
    return true;
  }
  const int32_t* seq_lens = reinterpret_cast<const int32_t*>(rt_data[mha_dense_io::MASK]);
  const int32_t bs = is_package_ ? *reinterpret_cast<const int32_t*>(rt_data[mha_dense_io::BS]) : src_bs_;
  constexpr size_t MAX_BS = 128;
  int32_t sl_accumulate[MAX_BS];
  sl_accumulate[0] = 0;
  if (is_package_)
    for (int32_t i = 1; i < bs; i++) sl_accumulate[i] = sl_accumulate[i - 1] + seq_lens[i];
  else
#pragma omp parallel for
    for (int32_t i = 1; i < bs; i++) sl_accumulate[i] = src_seq_len_ * i;

#pragma omp parallel for collapse(2)
  for (int ibs = 0; ibs < bs; ibs++) {
    for (int ihn = 0; ihn < head_num_; ihn++) {
      const int seq_len = is_package_ ? seq_lens[ibs] : src_seq_len_;
      const int sl_pad = (seq_len + 15) / 16 * 16;
      const int col_tile = sl_pad / 16;
      const int row_loop = col_tile / 2;
      const bool is_even = (col_tile % 2 == 0);
      const int rollback = (seq_len % 16 != 0) ? 16 - (seq_len % 16) : 0;
      const int sl_pad64 = (seq_len + 63) / 64 * 64;
      const int att_tile = sl_pad64 / 64;
      const auto k_scrach = aligned_allocator_t<int8_t>::allocate(head_size_ * sl_pad);
      const auto v_scrach_p64 = aligned_allocator_t<int8_t>::allocate(head_size_ * sl_pad64);
      const auto qk_scrach = aligned_allocator_t<int32_t>::allocate(32 * sl_pad);
      const auto softmax_scrach_p64 = aligned_allocator_t<uint8_t>::allocate(32 * sl_pad64);

      const int src_offset = sl_accumulate[ibs] * ld_src_ + ihn * head_size_;
      const int dst_offset = sl_accumulate[ibs] * ld_dst_ + ihn * head_size_ * get_data_size(dst_dt_);
      // init amx for each omp thread
      ker_amx_cfg_(&amx_full_tile_cfg_);
      const auto curr_q = reinterpret_cast<const int8_t*>(rt_data[mha_dense_io::SRC_Q]) + src_offset;
      const auto curr_dst = reinterpret_cast<char*>(const_cast<void*>(rt_data[mha_dense_io::DST])) + dst_offset;

      // reorder K
      jit_trans_AB16a4b::rt_data_t rt_data_tr_k;
      rt_data_tr_k.src = reinterpret_cast<const int8_t*>(rt_data[mha_dense_io::SRC_K]) + src_offset;
      rt_data_tr_k.dst = k_scrach;
      for (int i = 0; i < col_tile - 1; i++) {
        (*ker_trans_k_[16])(&rt_data_tr_k);
        rt_data_tr_k.src += ld_src_ * 16;
        rt_data_tr_k.dst += head_size_ * 16;
      }
      (*ker_trans_k_[seq_len - 16 * (col_tile - 1)])(&rt_data_tr_k);

      // reorder V
      jit_trans_BA16b4a::rt_data_t rt_data_tr_v;
      rt_data_tr_v.src = reinterpret_cast<const int8_t*>(rt_data[mha_dense_io::SRC_V]) + src_offset;
      rt_data_tr_v.dst = v_scrach_p64;
      rt_data_tr_v.ld_dst = jit_trans_BA16b4a::dst_stride(sl_pad64);
      const int v_buffer_row = sl_pad64 / 4;
      // const int v_stride = sl_pad64 * 16;  // v_buffer_row*64
      const int v_real_step = (seq_len + 3) / 4;
      // int v_tail = seq_len - (v_real_step - 1) * 4;
      int v_loop = 0;
      for (v_loop = 0; v_loop < v_real_step - 1; v_loop++) {
        (*ker_trans_v_[4])(&rt_data_tr_v);
        rt_data_tr_v.src += ld_src_ * 4;
        rt_data_tr_v.dst += 64;
      }
      (*ker_trans_v_[seq_len - 4 * (v_real_step - 1)])(&rt_data_tr_v);  // v_loop=v_real_step - 1
      while (v_loop < v_buffer_row - 1) {
        v_loop++;
        rt_data_tr_v.dst += 64;
        (*ker_trans_v_[0])(&rt_data_tr_v);
      }

      jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b::rt_data_t rt_data_qk;
      rt_data_qk.src1 = k_scrach;
      rt_data_qk.dst = qk_scrach;
      jit_softmax_Ab16a::rt_data_t rt_data_softmax1, rt_data_softmax2;
      rt_data_softmax1.QK_rescale = QK_rescale_;
      rt_data_softmax1.softmax_rescale = softmax_rescale_;
      rt_data_softmax1.att_tile = (reinterpret_cast<const int32_t*>(rt_data[mha_dense_io::MASK])[ibs]) / 16;
      rt_data_softmax1.src = qk_scrach;
      rt_data_softmax1.dst = softmax_scrach_p64;
      rt_data_softmax2.QK_rescale = QK_rescale_;
      rt_data_softmax2.softmax_rescale = softmax_rescale_;
      rt_data_softmax2.att_tile = rt_data_softmax1.att_tile;
      rt_data_softmax2.src = qk_scrach + sl_pad * 16;             // sl_pad_ / 16 * 16 * 16
      rt_data_softmax2.dst = softmax_scrach_p64 + sl_pad64 * 16;  // sl_pad64_ / 64 * 16 * 64
      jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab::rt_data_t rt_data_av;
      rt_data_av.src0 = softmax_scrach_p64;
      rt_data_av.src1 = v_scrach_p64;
      rt_data_av.K = reinterpret_cast<const int32_t*>(rt_data[mha_dense_io::MASK])[ibs];
      rt_data_av.rescale = QKV_rescale_;
      rt_data_av.zp = QKV_dstzp_;
      const int att_tail = reinterpret_cast<const int32_t*>(rt_data[mha_dense_io::MASK])[ibs] % 16;
      int cur_r_pos = 0;

      for (int j = 0; j < row_loop - 1; j++, cur_r_pos += 32) {
        rt_data_qk.src0 = curr_q + cur_r_pos * ld_src_;
        rt_data_av.dst = curr_dst + cur_r_pos * ld_dst_;
        mha_per_head_32x(rt_data_qk, rt_data_softmax1, rt_data_softmax2, rt_data_av, att_tail, col_tile, att_tile);
      }

      if (is_even) {
        if (rollback == 0) {
          rt_data_qk.src0 = curr_q + cur_r_pos * ld_src_;
          rt_data_av.dst = curr_dst + cur_r_pos * ld_dst_;
          mha_per_head_32x(rt_data_qk, rt_data_softmax1, rt_data_softmax2, rt_data_av, att_tail, col_tile, att_tile);
        } else {
          rt_data_qk.src0 = curr_q + cur_r_pos * ld_src_;
          rt_data_av.dst = curr_dst + cur_r_pos * ld_dst_;
          mha_per_head_16x(rt_data_qk, rt_data_softmax1, rt_data_av, att_tail, col_tile, att_tile);

          cur_r_pos += 16 - rollback;

          rt_data_qk.src0 = curr_q + cur_r_pos * ld_src_;
          rt_data_av.dst = curr_dst + cur_r_pos * ld_dst_;
          mha_per_head_16x(rt_data_qk, rt_data_softmax1, rt_data_av, att_tail, col_tile, att_tile);
        }
      } else {
        rt_data_qk.src0 = curr_q + cur_r_pos * ld_src_;
        rt_data_av.dst = curr_dst + cur_r_pos * ld_dst_;
        mha_per_head_32x(rt_data_qk, rt_data_softmax1, rt_data_softmax2, rt_data_av, att_tail, col_tile, att_tile);

        cur_r_pos += 32 - rollback;

        rt_data_qk.src0 = curr_q + cur_r_pos * ld_src_;
        rt_data_av.dst = curr_dst + cur_r_pos * ld_dst_;
        mha_per_head_16x(rt_data_qk, rt_data_softmax1, rt_data_av, att_tail, col_tile, att_tile);
      }

      // release amx for each omp thread
      ker_amx_rls_.tile_release();
      aligned_allocator_t<int8_t>::deallocate(k_scrach);
      aligned_allocator_t<int8_t>::deallocate(v_scrach_p64);
      aligned_allocator_t<int32_t>::deallocate(qk_scrach);
      aligned_allocator_t<int8_t>::deallocate(softmax_scrach_p64);
    }
  }

  return true;
}

}  // namespace jd
