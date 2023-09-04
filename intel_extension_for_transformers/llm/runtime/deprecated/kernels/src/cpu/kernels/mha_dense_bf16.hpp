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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_MHA_DENSE_BF16_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_MHA_DENSE_BF16_HPP_

#include <memory>
#include <vector>

#include "kernels/amx_utils.hpp"
#include "src/cpu/cpu_isa.hpp"
#include "kernels/exposed_enum.hpp"
#include "src/cpu/jit_domain/jit_mha_dense_bf16.hpp"
#include "src/cpu/jit_domain/jit_trans_AB16a4b_16x.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "operator_desc.hpp"
#include "src/utils.hpp"

namespace jd {
class mha_dense_bf16_k_t;

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
class mha_dense_bf16_kd_t : public kernel_desc_t {
  using io = exposed_enum::mha_dense::io;
  using io_src = exposed_enum::mha_dense_src::src;
  using io_dst = exposed_enum::mha_dense_dst::dst;
  using io_shape = exposed_enum::mha_dense_shape::shape;

 public:
  explicit mha_dense_bf16_kd_t(const operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::mha_dense), op_desc_(op_desc) {}
  virtual ~mha_dense_bf16_kd_t() {}

 public:
  bool init() override;
  DECLARE_COMMON_PD_T(mha_dense_bf16_k_t, mha_dense_bf16_kd_t);

 public:
  const operator_desc& get_operator_desc() const override { return op_desc_; }
  inline std::vector<dim_t> shape() const override {
    return {
        op_desc_.tensor_descs()[io::SRC_Q].shape()[0],  // batch_size
        op_desc_.tensor_descs()[io::SRC_Q].shape()[1],  // sl_M
        op_desc_.tensor_descs()[io::SRC_K].shape()[1],  // sl_N
        op_desc_.tensor_descs()[io::SRC_Q].shape()[2],  // head_num
        op_desc_.tensor_descs()[io::SRC_Q].shape()[3],  // head_size
    };
  }

 private:
  operator_desc op_desc_;
};

class mha_dense_bf16_k_t : public kernel_t {
  using io = exposed_enum::mha_dense::io;
  using io_src = exposed_enum::mha_dense_src::src;
  using io_dst = exposed_enum::mha_dense_dst::dst;
  using io_shape = exposed_enum::mha_dense_shape::shape;

 public:
  using kd_t = mha_dense_bf16_kd_t;
  explicit mha_dense_bf16_k_t(const std::shared_ptr<const kernel_desc_t>& kd);
  virtual ~mha_dense_bf16_k_t() {}
  // Delete move constructor and move operator
  mha_dense_bf16_k_t(mha_dense_bf16_k_t&&) = delete;
  mha_dense_bf16_k_t& operator=(mha_dense_bf16_k_t&&) = delete;
  // Delete copy constructor and copy operator
  mha_dense_bf16_k_t(const mha_dense_bf16_k_t&) = delete;
  mha_dense_bf16_k_t& operator=(const mha_dense_bf16_k_t&) = delete;

  size_t get_workspace_size() const override { return workspace_size_; }
  bool init() override;
  [[deprecated("Please use exec_context_t instead of rt_data")]] bool execute(
      const std::vector<const void*>& rt_data) const override;
  bool execute(const exec_context_t& ctx) const override;
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  const std::vector<tensor_desc>& ts_descs_;
  const bool has_pmask;  // padding mask
  const bool has_badd;
  const data_type dt_dst;
  const int bs_, sl_m_, sl_n_, head_num_, head_size_;
  const int ld_src_, ld_dst_;  // in terms of #elements
  static constexpr int PAD_SIZE = 32;
  const int sl_n_pad_, head_size_pad_;

  const size_t reo_k_size_;           // workspace size for storing the reorderd K matrix
  const size_t reo_v_size_;           // workspace size for storing the reorderd K matrix
  const size_t thread_reo_q_size_;    // workspace size for each threads for storing the reorderd Q matrix
  const size_t thread_softmax_size_;  // workspace size for each threads for storing the softmax results
  const size_t thread_dst_size_;      // workspace size for each threads for storing the dst results of the 2nd matmul
  const size_t thread_total_bytes_;   // total workspace size for each threads
  const size_t tmp_badd_size_;        // workspace size for processed binary-add data
  const size_t workspace_size_;

  const tile_param_t amx_full_tile_param_;
  const tileconfig_t amx_full_tile_cfg_;
  jit_amx_config_t ker_amx_cfg_;
  jit_amx_release_t ker_amx_rls_;

  jit_trans_AB16a4b_16x kern_tr_k;
  jit_padding_interleave4b_n kern_tr_v;
  jit_padding_copy2d kern_tr_q;
  jit_mha_bf16_row_amx_32x32_softmax kern_qksoftmax;
  jit_mha_bf16_row_amx_32x32 kern_mmav;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_MHA_DENSE_BF16_HPP_
