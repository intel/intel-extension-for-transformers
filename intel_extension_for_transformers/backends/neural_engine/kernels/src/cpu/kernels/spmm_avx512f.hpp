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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_SPMM_AVX512F_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_SPMM_AVX512F_HPP_

#include <glog/logging.h>
#include <algorithm>
#include <memory>
#include <vector>

#include "src/cpu/cpu_isa.hpp"
#include "src/cpu/jit_domain/jit_spmm_avx512f.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"

namespace jd {
// By convention,
//   1. xxxx_kd_t is the descriptor of a specific derived primitive/kernel.
//   2. xxxx_k_t is a specific derived primitive/kernel.
//   3. jit_xxxx_t is JIT assembly implementation of a specific derived
//   primitive/kernel. where, "xxxx" represents an algorithm, such as brgemm,
//   GEMM and so on.
class spmm_avx512f_k_t;

/**
 * @brief a derived kernel descriptor. avx512_fp32_params_t is its class member.
 */
class spmm_avx512f_kd_t : public kernel_desc_t {
 public:
  explicit spmm_avx512f_kd_t(const operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::sparse_matmul), op_desc_(op_desc) {}
  virtual ~spmm_avx512f_kd_t() {}

  bool init() override;
  // kernel_desc_t::create_primitive() override.
  DECLARE_COMMON_PD_T(spmm_avx512f_k_t, spmm_avx512f_kd_t);

  const operator_desc& get_operator_desc() const override { return op_desc_; }
  const std::vector<ssd::avx512_fp32_params_t>& params() const { return params_; }
  const dim_t& block_m() const { return block_m_; }
  inline std::vector<dim_t> shape() const {
    return {op_desc_.tensor_descs()[ssd::WEI].shape()[0], op_desc_.tensor_descs()[ssd::WEI].shape()[1],
            op_desc_.tensor_descs()[ssd::SRC].shape()[1]};
  }

 private:
  bool spmm_params_init(const operator_desc& op_desc);

  operator_desc op_desc_;
  std::vector<ssd::avx512_fp32_params_t> params_;
  dim_t block_m_ = 64;
};

/**
 * @brief a derived kernel. kd_t and jit_domain are its class members.
 */
class spmm_avx512f_k_t : public kernel_t {
 public:
  using kd_t = spmm_avx512f_kd_t;
  explicit spmm_avx512f_k_t(const std::shared_ptr<const kd_t>& kd) : kernel_t(kd) {}
  virtual ~spmm_avx512f_k_t() {
    for (auto& kernel : jit_kers_) safe_delete(kernel);
  }
  // Delete move constructor and move operator
  spmm_avx512f_k_t(spmm_avx512f_k_t&& other) = delete;
  spmm_avx512f_k_t& operator=(spmm_avx512f_k_t&& other) = delete;
  // Delete copy constructor and copy operator
  spmm_avx512f_k_t(const spmm_avx512f_k_t& other) = delete;
  spmm_avx512f_k_t& operator=(const spmm_avx512f_k_t& other) = delete;

 public:
  bool init() override;
  bool execute(const std::vector<const void*>& rt_data) const override;

 public:
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  bool spmm_kernel_create(jit_spmm_avx512f_t** ker_pp, const ssd::avx512_fp32_params_t& param);

 private:
  std::vector<jit_spmm_avx512f_t*> jit_kers_;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_SPMM_AVX512F_HPP_
