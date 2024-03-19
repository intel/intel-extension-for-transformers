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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_SPMM_AMX_BF16_X16_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_SPMM_AMX_BF16_X16_HPP_

#include <memory>
#include <vector>

#include "src/cpu/jit_domain/jit_spmm_amx_bf16_x16.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "kernels/sparse_data.hpp"
#include "kernels/spmm_types.hpp"
#include "operator_desc.hpp"
#include "kernels/amx_utils.hpp"
#include "src/cpu/cpu_isa.hpp"

namespace jd {
// By convention,
//   1. xxxx_kd_t is the descriptor of a specific derived primitive/kernel.
//   2. xxxx_k_t is a specific derived primitive/kernel.
//   3. jit_xxxx_t is JIT assembly implementation of a specific derived
//   primitive/kernel. where, "xxxx" represents an algorithm, such as brgemm,
//   GEMM and so on.
class spmm_amx_bf16_x16_k_t;
/**
 * @brief a derived kernel descriptor. amx_bf16_params_t is its class member.
 */
class spmm_amx_bf16_x16_kd_t : public kernel_desc_t {
 public:
  explicit spmm_amx_bf16_x16_kd_t(const operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::sparse_matmul), op_desc_(op_desc) {}
  virtual ~spmm_amx_bf16_x16_kd_t() {}

 public:
  bool init() override;
  // kernel_desc_t::create_primitive() override.
  DECLARE_COMMON_PD_T(spmm_amx_bf16_x16_k_t, spmm_amx_bf16_x16_kd_t);

 public:
  const operator_desc& get_operator_desc() const override { return op_desc_; }
  const std::vector<ssd::amx_bf16_params_t>& params() const { return params_; }
  inline const dim_t& num_kernels() const { return num_kernels_; }
  inline std::vector<dim_t> shape() const {
    return {op_desc_.tensor_descs()[ssd::WEI].shape()[0], op_desc_.tensor_descs()[ssd::WEI].shape()[1],
            op_desc_.tensor_descs()[ssd::SRC].shape()[0] * op_desc_.tensor_descs()[ssd::SRC].shape()[2]};
  }

 private:
  bool spmm_params_init(std::vector<ssd::amx_bf16_params_t>& param_ref,  // NOLINT
                        const operator_desc& op_cfg);

 private:
  operator_desc op_desc_;
  std::vector<ssd::amx_bf16_params_t> params_;
  dim_t num_kernels_;
};

/**
 * @brief a derived kernel. kd_t and jit_domain are its class members.
 */
class spmm_amx_bf16_x16_k_t : public kernel_t {
 public:
  using kd_t = spmm_amx_bf16_x16_kd_t;
  explicit spmm_amx_bf16_x16_k_t(const std::shared_ptr<const kd_t>& kd) : kernel_t(kd) {}
  virtual ~spmm_amx_bf16_x16_k_t() {
    for (auto& kernel : jit_kers_) safe_delete(kernel);
  }
  // Delete move constructor and move operator
  spmm_amx_bf16_x16_k_t(spmm_amx_bf16_x16_k_t&& other) = delete;
  spmm_amx_bf16_x16_k_t& operator=(spmm_amx_bf16_x16_k_t&& other) = delete;
  // Delete copy constructor and copy operator
  spmm_amx_bf16_x16_k_t(const spmm_amx_bf16_x16_k_t& other) = delete;
  spmm_amx_bf16_x16_k_t& operator=(const spmm_amx_bf16_x16_k_t& other) = delete;

 public:
  bool init() override;
  bool execute(const std::vector<const void*>& rt_data) const override;

 public:
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  bool spmm_kernel_create(jit_spmm_amx_bf16_x16_t** ker_pp, const ssd::amx_bf16_params_t& param);
  dim_t tileBS = 0;
  dim_t num_tileBS = 0;
  dim_t tileOC = 0;
  dim_t num_tileOC = 0;
  dim_t IC = 0;
  dim_t OC = 0;
  dim_t thread_num_ = 0;

 private:
  std::vector<jit_spmm_amx_bf16_x16_t*> jit_kers_;
  std::vector<bfloat16_t*> weights_;
  const tile_param_t tile_param_ = tile_param_t(TILE_M, TILE_N, TILE_K, true, 2);
  amx_tile_config_t* amx_config_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_SPMM_AMX_BF16_X16_HPP_
