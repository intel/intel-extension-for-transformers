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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_MATMUL_REF_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_MATMUL_REF_HPP_

#include <glog/logging.h>
#include <algorithm>
#include <vector>
#include <memory>
#include "src/cpu/cpu_isa.hpp"
#include "operator_desc.hpp"
#include "kernel_desc.hpp"
#include "kernel.hpp"
#include "src/utils.hpp"
#include "kernels/matmul_types.hpp"

namespace jd {
// By convention,
//   1. xxxx_kd_t is the descriptor of a specific derived primitive/kernel.
//   2. xxxx_k_t is a specific derived primitive/kernel.
//   3. jit_xxxx_t is JIT assembly implementation of a specific derived primitive/kernel.
//   where, "xxxx" represents an algorithm, such as brgemm, GEMM and so on.
class matmul_ref_k_t;

// TODO(hengyu): hide reference implementation from users
/**
 * @brief a derived kernel descriptor. ref_param_t is its class member.
 */
class SPARSE_TEST_API_ matmul_ref_kd_t : public kernel_desc_t {
 public:
  explicit matmul_ref_kd_t(const operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::sparse_matmul), op_desc_(op_desc), shape_(5) {}
  virtual ~matmul_ref_kd_t() {}

 public:
  bool init() override;
  // kernel_desc_t::create_primitive() override.
  DECLARE_COMMON_PD_T(matmul_ref_k_t, matmul_ref_kd_t);

 public:
  const operator_desc& get_operator_desc() const override { return op_desc_; }
  inline std::vector<dim_t> shape() const { return shape_; }
  inline dim_t bs0() const { return shape_[0]; }
  inline dim_t bs1() const { return shape_[1]; }
  inline dim_t M() const { return shape_[2]; }
  inline dim_t K() const { return shape_[3]; }
  inline dim_t N() const { return shape_[4]; }
  inline const std::vector<std::vector<dim_t>>& perm() const { return *perm_ptr_; }

 private:
  operator_desc op_desc_;
  std::vector<dim_t> shape_;
  const std::vector<std::vector<dim_t>>* perm_ptr_ = nullptr;
};

/**
 * @brief a derived kernel. kd_t and jit_domain are its class members.
 */
class SPARSE_TEST_API_ matmul_ref_k_t : public kernel_t {
 public:
  using kd_t = matmul_ref_kd_t;
  explicit matmul_ref_k_t(const std::shared_ptr<const kd_t>& kd);
  virtual ~matmul_ref_k_t() {}

  // Delete move constructor and move operator
  matmul_ref_k_t(matmul_ref_k_t&& other) = delete;
  matmul_ref_k_t& operator=(matmul_ref_k_t&& other) = delete;
  // Delete copy constructor and copy operator
  matmul_ref_k_t(const matmul_ref_k_t& other) = delete;
  matmul_ref_k_t& operator=(const matmul_ref_k_t& other) = delete;

 public:
  bool init() override;
  bool execute(const std::vector<const void*>& rt_data) const override;
  bool execute(const exec_context_t& context) const override;
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }
  inline const std::vector<std::vector<dim_t>>& perm() const { return derived_kd()->perm(); }

 private:
  dim_t bs0_;
  dim_t bs1_;
  dim_t M_;
  dim_t K_;
  dim_t N_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_MATMUL_REF_HPP_
