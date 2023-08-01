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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_matmul_AVX512F_8BIT_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_matmul_AVX512F_8BIT_HPP_

#include <memory>
#include <vector>
#include <algorithm>

#include "src/cpu/cpu_isa.hpp"
#include "src/cpu/cpu_parallel.hpp"
#include "src/cpu/jit_domain/jit_gemm_avx512f_8bit.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"

namespace jd {
// By convention,
//   1. xxxx_kd_t is the descriptor of a specific derived primitive/kernel.
//   2. xxxx_k_t is a specific derived primitive/kernel.
//   3. jit_xxxx_t is JIT assembly implementation of a specific derived
//   primitive/kernel. where, "xxxx" represents an algorithm, such as brgemm,
//   GEMM and so on.
class matmul_avx512f_8bit_k_t;

/**
 * @brief a derived kernel descriptor. matmul_param_t is its class member.
 */
class matmul_avx512f_8bit_kd_t : public kernel_desc_t {
 public:
  using io = ssd::matmul_io::io;
  explicit matmul_avx512f_8bit_kd_t(const operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::sparse_matmul), op_desc_(op_desc) {}
  virtual ~matmul_avx512f_8bit_kd_t() {}

  bool init() override;

  // kernel_desc_t::create_primitive() override.
  DECLARE_COMMON_PD_T(matmul_avx512f_8bit_k_t, matmul_avx512f_8bit_kd_t);

  const operator_desc& get_operator_desc() const override { return op_desc_; }
  const ssd::matmul_fp8_param_t& jit_param() const { return jit_param_; }

  inline std::vector<dim_t> shape() const {
    std::vector<dim_t> result(op_desc_.tensor_descs()[io::SRC0].shape());
    result.push_back(op_desc_.tensor_descs()[io::SRC0].shape().back());
    return result;
  }

 private:
  bool params_init();

 private:
  operator_desc op_desc_;
  ssd::matmul_fp8_param_t jit_param_;
};

/**
 * @brief a derived kernel. kd_t and jit_domain are its class members.
 */
class matmul_avx512f_8bit_k_t : public kernel_t {
 public:
  using io = ssd::matmul_io::io;
  using kd_t = matmul_avx512f_8bit_kd_t;
  explicit matmul_avx512f_8bit_k_t(const std::shared_ptr<const kd_t>& kd);
  virtual ~matmul_avx512f_8bit_k_t() { safe_delete(jit_ker_); }

  // Delete move constructor and move operator
  matmul_avx512f_8bit_k_t(matmul_avx512f_8bit_k_t&& other) = delete;
  matmul_avx512f_8bit_k_t& operator=(matmul_avx512f_8bit_k_t&& other) = delete;
  // Delete copy constructor and copy operator
  matmul_avx512f_8bit_k_t(const matmul_avx512f_8bit_k_t& other) = delete;
  matmul_avx512f_8bit_k_t& operator=(const matmul_avx512f_8bit_k_t& other) = delete;

  bool init() override;

  [[deprecated("Please use exec_context_t instead of rt_data")]] bool execute(
      const std::vector<const void*>& rt_data) const override;

  bool execute(const exec_context_t& context) const override;

  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  bool matmul_kernel_create(jit_gemm_avx512f_8bit_t** ker_pp, const ssd::matmul_param_t& param);

 private:
  jit_gemm_avx512f_8bit_t* jit_ker_ = nullptr;
  const std::vector<std::vector<dim_t>> t_shapes_;
  mutable dim_t M_;
  dim_t K_, N_;  // dim of matrix multiplication
  int lda, ldb, ldc, ldd;
  mutable GemmCacheAdpter<float> mCacheAdapter;
  mutable Parallel2DGemmV2<float> mParallel;
};

}  // namespace jd

#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_matmul_AVX512F_8BIT_HPP_
