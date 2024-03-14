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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_DYNAMIC_QUANT_MHA_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_DYNAMIC_QUANT_MHA_HPP_

#include <memory>
#include <vector>

#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "kernels/amx_utils.hpp"
#include "kernels/exposed_enum.hpp"
#include "operator_desc.hpp"
#include "src/cpu/cpu_isa.hpp"
#include "src/cpu/jit_domain/jit_dynamic_quant.hpp"
#include "src/cpu/jit_domain/jit_dynamic_quant_mha.hpp"
#include "src/cpu/jit_domain/jit_trans_AB16a4b_16x.hpp"
#include "src/cpu/jit_domain/jit_trans_BA16b4a_trq10n_x16.hpp"
#include "src/utils.hpp"

namespace jd {

/**
 * @brief
 *       Q       K       V
 *       |       |       |
 *       |       |       |
 *       |    Reorder    |
 *        \     /        |
 *         \   /      Reorder
 *        Matmul        /
 *           |         /
 *           |        /
 *         Softmax   /
 *            \     /
 *             \   /
 *             Matmul
 *               |
 *               |
 *             Output
 */
class dynamic_quant_mha_k_t;

class dynamic_quant_mha_kd_t : public kernel_desc_t {
 public:
  using io = exposed_enum::mha_dense::io;
  explicit dynamic_quant_mha_kd_t(const operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::mha_dense), op_desc_(op_desc) {}
  virtual ~dynamic_quant_mha_kd_t() {}

  bool init() override;
  DECLARE_COMMON_PD_T(dynamic_quant_mha_k_t, dynamic_quant_mha_kd_t);

  const operator_desc& get_operator_desc() const override { return op_desc_; }
  inline std::vector<dim_t> shape() const override {
    return {
        op_desc_.tensor_descs()[io::SRC_Q].shape()[0],  // batch_size
        op_desc_.tensor_descs()[io::SRC_Q].shape()[2],  // head_num
        op_desc_.tensor_descs()[io::SRC_Q].shape()[1],  // M
        op_desc_.tensor_descs()[io::SRC_Q].shape()[3],  // head_size
        op_desc_.tensor_descs()[io::SRC_K].shape()[1],  // N
    };
  }

 private:
  operator_desc op_desc_;
};

class dynamic_quant_mha_k_t : public kernel_t {
 public:
  using io = exposed_enum::mha_dense::io;
  using kd_t = dynamic_quant_mha_kd_t;
  explicit dynamic_quant_mha_k_t(const std::shared_ptr<const kernel_desc_t>& kd);
  virtual ~dynamic_quant_mha_k_t() {}
  // Delete move constructor and move operator
  dynamic_quant_mha_k_t(dynamic_quant_mha_k_t&&) = delete;
  dynamic_quant_mha_k_t& operator=(dynamic_quant_mha_k_t&&) = delete;
  // Delete copy constructor and copy operator
  dynamic_quant_mha_k_t(const dynamic_quant_mha_k_t&) = delete;
  dynamic_quant_mha_k_t& operator=(const dynamic_quant_mha_k_t&) = delete;

  bool init() override;
  bool execute(const std::vector<const void*>& rt_data) const override;
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }
  size_t get_workspace_size() const override;

 private:
  const std::vector<std::vector<dim_t>> t_shapes_;
  const int32_t batch_size_, head_num_, M_, head_size_, N_;
  const bool has_attscale;
  const bool has_badd;

  jit_amx_config_t ker_amx_cfg_;
  jit_amx_release_t ker_amx_rls_;
  std::unique_ptr<jit_trans_AB16a4b_16x> ker_seq_cpy_k_;
  std::unique_ptr<jit_trans_BA16b4a_trq10n_x16> ker_seq_cpy_v_;
  std::unique_ptr<jit_mmexp_amx_s8_ab_BA16b4a_u8_16x> ker_qxk_;
  std::unique_ptr<jit_scale_mm_amx_u8s8_ab_BA16b_16x> ker_axv_;
  std::unique_ptr<jit_dynamic_quant_t> ker_quant_;

  const tile_param_t amx_full_tile_param_;
  const tileconfig_t amx_full_tile_cfg_;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_DYNAMIC_QUANT_MHA_HPP_
