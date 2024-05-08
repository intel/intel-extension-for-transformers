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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_MHA_DENSE_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_MHA_DENSE_HPP_

#include <memory>
#include <vector>

#include "kernels/amx_utils.hpp"
#include "src/cpu/cpu_isa.hpp"
#include "kernels/exposed_enum.hpp"
#include "src/cpu/jit_domain/jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b.hpp"
#include "src/cpu/jit_domain/jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab.hpp"
#include "src/cpu/jit_domain/jit_softmax_Ab16a.hpp"
#include "src/cpu/jit_domain/jit_trans_AB16a4b.hpp"
#include "src/cpu/jit_domain/jit_trans_BA16b4a.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "operator_desc.hpp"
#include "src/utils.hpp"

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
  using io = exposed_enum::mha_dense::io;
  using io_src = exposed_enum::mha_dense_src::src;
  using io_dst = exposed_enum::mha_dense_dst::dst;
  using io_shape = exposed_enum::mha_dense_shape::shape;

 public:
  explicit mha_dense_kd_t(const operator_desc& op_desc) : kernel_desc_t(kernel_kind::mha_dense), op_desc_(op_desc) {}
  virtual ~mha_dense_kd_t() {}

  bool init() override;
  DECLARE_COMMON_PD_T(mha_dense_k_t, mha_dense_kd_t);

  const operator_desc& get_operator_desc() const override { return op_desc_; }
  inline std::vector<dim_t> shape() const override { return op_desc_.tensor_descs()[io::DST].shape(); }
  bool has_binary_add() const {
    return op_desc_.tensor_descs().size() > io::BINARY_ADD &&
           op_desc_.tensor_descs()[io::BINARY_ADD].dtype() != data_type::undef;
  }
  bool merged() const { return merged_; }

 private:
  operator_desc op_desc_;
  bool merged_;
};

class mha_dense_k_t : public kernel_t {
  using io = exposed_enum::mha_dense::io;
  using io_src = exposed_enum::mha_dense_src::src;
  using io_dst = exposed_enum::mha_dense_dst::dst;
  using io_shape = exposed_enum::mha_dense_shape::shape;

 public:
  using kd_t = mha_dense_kd_t;
  static constexpr int MAX_SL_N = 2048;
  explicit mha_dense_k_t(const std::shared_ptr<const kernel_desc_t>& kd);
  virtual ~mha_dense_k_t() {
    for (int j = 1; j <= MAX_SL_N / 64; j++) {
      for (int i = 0; i < 16; i++) safe_delete(ker_softmax_[j][i]);
      safe_delete(ker_av_gemm_16x_[j]);
      safe_delete(ker_av_gemm_32x_[j]);
    }
    for (int j = 1; j <= 16; j++) safe_delete(ker_trans_k_[j]);
    for (int j = 0; j <= 4; j++) safe_delete(ker_trans_v_[j]);
    for (int j = 1; j <= MAX_SL_N / 16; j++) {
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
  [[deprecated("Please use exec_context_t instead of rt_data")]] bool execute(
      const std::vector<const void*>& rt_data) const override;
  bool execute(const exec_context_t& context) const override;
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  bool execute_tiny(void** src_data, void** dst_data, void* workspace, dim_t* shape_data) const;
  const std::vector<tensor_desc>& ts_desc;
  const data_type dst_dt_;
  const format_type kv_ft_;
  const int src_bs_, src_sl_m_, src_sl_n_, head_num_, head_size_, ld_q_, ld_kv_, ld_dst_;
  const float softmax_rescale_f32_;
  const float16_t softmax_rescale_;

  const std::vector<tensor_desc>& ts_descs_;
  const bool has_binary_add;

  const size_t thread_workspace_size_;

  const tile_param_t amx_full_tile_param_;
  const tileconfig_t amx_full_tile_cfg_;
  jit_amx_config_t ker_amx_cfg_;
  jit_amx_release_t ker_amx_rls_;

  jit_trans_AB16a4b* ker_trans_k_[17];
  jit_trans_BA16b4a* ker_trans_v_[5];
  jit_softmax_Ab16a* ker_softmax_[MAX_SL_N / 64 + 1][16];
  jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b* ker_qk_gemm_16x_[MAX_SL_N / 16 + 1];
  jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b* ker_qk_gemm_32x_[MAX_SL_N / 16 + 1];
  jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab* ker_av_gemm_16x_[MAX_SL_N / 64 + 1];
  jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab* ker_av_gemm_32x_[MAX_SL_N / 64 + 1];
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
#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_MHA_DENSE_HPP_
