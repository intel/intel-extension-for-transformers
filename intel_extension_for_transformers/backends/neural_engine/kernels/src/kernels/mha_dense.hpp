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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_MHA_DENSE_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_MHA_DENSE_HPP_

#include <memory>
#include <vector>

#include "amx_utils.hpp"
#include "cpu_isa.hpp"
#include "jit_domain/jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b.hpp"
#include "jit_domain/jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab.hpp"
#include "jit_domain/jit_softmax_Ab16a.hpp"
#include "jit_domain/jit_trans_AB16a4b.hpp"
#include "jit_domain/jit_trans_BA16b4a.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "kernels/mha_dense_types.hpp"
#include "operator_desc.hpp"
#include "utils.hpp"

namespace jd {
class mha_dense_k_t;

/**
 * @brief Multi-head attention kernel with static quantization.
 *       Q       K         V
 *       |       |         |
 *       |       |         |
 *       |    Reorder      |
 *        \     /          |
 *         \   /        Reorder
 *        Matmul           |
 *           |            /
 *   [opt] binary_add    /
 *           |          /
 *         Softmax     /
 *            \       /
 *             \     /
 *             Matmul
 *               |
 *               |
 *             Output
 *
 * Currently only support per-tensor quantization.
 */
class mha_dense_kd_t : public kernel_desc_t {
  using io = mha_dense_io::io;

 public:
  explicit mha_dense_kd_t(const jd::operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::mha_dense), op_desc_(op_desc) {}
  virtual ~mha_dense_kd_t() {}

 public:
  bool init() override;
  DECLARE_COMMON_PD_T(mha_dense_k_t, mha_dense_kd_t);

 public:
  const jd::operator_desc& get_operator_desc() const override { return op_desc_; }
  inline std::vector<dim_t> shape() const override { return op_desc_.tensor_descs()[io::DST].shape(); }
  const ssd::mha_dense_param_t& params() const { return param_; }
  bool has_binary_add() const {
    return op_desc_.tensor_descs().size() > io::BINARY_ADD &&
           op_desc_.tensor_descs()[io::BINARY_ADD].dtype() != data_type::undef;
  }

 private:
  jd::operator_desc op_desc_;
  ssd::mha_dense_param_t param_;
  // bool add_kernel_desc(const operator_desc& op_desc, const char* name);
};

class mha_dense_k_t : public kernel_t {
  using io = mha_dense_io::io;

 public:
  using kd_t = mha_dense_kd_t;
  static constexpr int MAX_SEQLEN = 2048;
  explicit mha_dense_k_t(const std::shared_ptr<const kernel_desc_t>& kd);
  virtual ~mha_dense_k_t() {
    for (int j = 1; j <= MAX_SEQLEN / 64; j++) {
      for (int i = 0; i < 16; i++) safe_delete(ker_softmax_[j][i]);
      safe_delete(ker_av_gemm_16x_[j]);
      safe_delete(ker_av_gemm_32x_[j]);
    }
    for (int j = 1; j <= 16; j++) safe_delete(ker_trans_k_[j]);
    for (int j = 0; j <= 4; j++) safe_delete(ker_trans_v_[j]);
    for (int j = 1; j <= MAX_SEQLEN / 16; j++) {
      safe_delete(ker_qk_gemm_16x_[j]);
      safe_delete(ker_qk_gemm_32x_[j]);
    }
  }
  // Delete move constructor and move operator
  mha_dense_k_t(mha_dense_k_t&& other) = delete;
  mha_dense_k_t& operator=(mha_dense_k_t&& other) = delete;
  // Delete copy constructor and copy operator
  mha_dense_k_t(const mha_dense_k_t& other) = delete;
  mha_dense_k_t& operator=(const mha_dense_k_t& other) = delete;

  size_t get_workspace_size() const override { return omp_get_max_threads() * thread_workspace_size_; }
  bool init() override;
  bool execute(const std::vector<const void*>& rt_data) const override;
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  const data_type dst_dt_;
  const int src_bs_, src_seq_len_, head_num_, head_size_, ld_src_, ld_dst_;
  const uint16_t softmax_rescale_;
  const float QK_rescale_, QKV_rescale_, QKV_dstzp_;
  const float Q_scale_, K_scale_, V_scale_, DST_scale_, QK_output_scale_;

  const std::vector<jd::tensor_desc>& ts_descs_;
  const bool has_binary_add;

  const size_t thread_workspace_size_;

  const tile_param_t amx_full_tile_param_;
  const tileconfig_t amx_full_tile_cfg_;
  jd::jit_amx_config_t ker_amx_cfg_;
  jd::jit_amx_release_t ker_amx_rls_;

  jit_trans_AB16a4b* ker_trans_k_[17];
  jit_trans_BA16b4a* ker_trans_v_[5];
  jit_softmax_Ab16a* ker_softmax_[MAX_SEQLEN / 64 + 1][16];
  jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b* ker_qk_gemm_16x_[MAX_SEQLEN / 16 + 1];
  jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b* ker_qk_gemm_32x_[MAX_SEQLEN / 16 + 1];
  jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab* ker_av_gemm_16x_[MAX_SEQLEN / 64 + 1];
  jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab* ker_av_gemm_32x_[MAX_SEQLEN / 64 + 1];
  inline void mha_per_head_32x(const jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b::rt_data_t& rt_data_qk,
                               const jit_softmax_Ab16a::rt_data_t& rt_data_softmax1,
                               const jit_softmax_Ab16a::rt_data_t& rt_data_softmax2,
                               const jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab::rt_data_t& rt_data_av, const int att_tail,
                               const int col_tile, const int att_tile) const;
  inline void mha_per_head_16x(const jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b::rt_data_t& rt_data_qk,
                               const jit_softmax_Ab16a::rt_data_t& rt_data_softmax1,
                               const jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab::rt_data_t& rt_data_av, const int att_tail,
                               const int col_tile, const int att_tile) const;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_MHA_DENSE_HPP_
