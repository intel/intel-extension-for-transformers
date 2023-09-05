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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_MATMUL_VNNI_P2031_P2013_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_MATMUL_VNNI_P2031_P2013_HPP_

#include <glog/logging.h>
#include <memory>
#include <vector>
#include <algorithm>

#include "src/cpu/cpu_isa.hpp"
#include "src/cpu/jit_domain/jit_matmul_vnni_8xkx48.hpp"
#include "src/cpu/jit_domain/jit_seq_cpy_2x8x8.hpp"
#include "src/cpu/jit_domain/jit_seq_cpy_48x4.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"

namespace jd {
// By convention,
//   1. xxxx_kd_t is the descriptor of a specific derived primitive/kernel.
//   2. xxxx_k_t is a specific derived primitive/kernel.
//   3. jit_xxxx_t is JIT assembly implementation of a specific derived
//   primitive/kernel. where, "xxxx" represents an algorithm, such as brgemm,
//   GEMM and so on.
class matmul_vnni_p2031_p2013_k_t;

/**
 * @brief a derived kernel descriptor. matmul_param_t is its class member.
 */
class matmul_vnni_p2031_p2013_kd_t : public kernel_desc_t {
 public:
  using io = ssd::matmul_io::io;
  explicit matmul_vnni_p2031_p2013_kd_t(const operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::sparse_matmul), op_desc_(op_desc) {}
  virtual ~matmul_vnni_p2031_p2013_kd_t() {}

  bool init() override;
  inline float src0_scale() const { return src0_scale_; }
  inline float src1_scale() const { return src1_scale_; }
  inline float out_scale() const { return out_scale_; }

  // kernel_desc_t::create_primitive() override.
  DECLARE_COMMON_PD_T(matmul_vnni_p2031_p2013_k_t, matmul_vnni_p2031_p2013_kd_t);

  const operator_desc& get_operator_desc() const override { return op_desc_; }

  inline std::vector<dim_t> shape() const override {
    std::vector<dim_t> result(op_desc_.tensor_descs()[io::SRC0].shape());
    result.push_back(op_desc_.tensor_descs()[io::SRC0].shape().back());
    return result;
  }

 private:
  operator_desc op_desc_;
  float src0_scale_ = 1;
  float src1_scale_ = 1;
  float out_scale_ = 1;
};

/**
 * @brief a derived kernel. kd_t and jit_domain are its class members.
 */
class matmul_vnni_p2031_p2013_k_t : public kernel_t {
 public:
  using io = ssd::matmul_io::io;
  using kd_t = matmul_vnni_p2031_p2013_kd_t;
  explicit matmul_vnni_p2031_p2013_k_t(const std::shared_ptr<const kd_t>& kd);
  virtual ~matmul_vnni_p2031_p2013_k_t() {
    safe_delete(add128_2x8x8_ker_);
    safe_delete(cpy_48x4_ker_);
    safe_delete(matmul_ker_);
    safe_delete(matmul_tile_n_ker_);
    aligned_allocator_t<char>::deallocate(src0_tmp_);
    aligned_allocator_t<char>::deallocate(src1_tmp_);
  }

  // Delete move constructor and move operator
  matmul_vnni_p2031_p2013_k_t(matmul_vnni_p2031_p2013_k_t&& other) = delete;
  matmul_vnni_p2031_p2013_k_t& operator=(matmul_vnni_p2031_p2013_k_t&& other) = delete;
  // Delete copy constructor and copy operator
  matmul_vnni_p2031_p2013_k_t(const matmul_vnni_p2031_p2013_k_t& other) = delete;
  matmul_vnni_p2031_p2013_k_t& operator=(const matmul_vnni_p2031_p2013_k_t& other) = delete;

  bool init() override;
  bool execute(const std::vector<const void*>& rt_data) const override;
  template <typename dt_dst>
  bool execute_(const std::vector<const void*>& rt_data) const;
  template <typename dt_dst>
  bool thread_exec(const std::vector<const void*>& rt_data, const dim_t ibs0, const dim_t ibs1) const;
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  static constexpr int max_cpy_nthr = 16;
  jit_seq_cpy_2x8x8* add128_2x8x8_ker_ = nullptr;
  jit_seq_cpy_48x4* cpy_48x4_ker_ = nullptr;
  jit_matmul_vnni_8xkx48_t* matmul_ker_ = nullptr;
  jit_matmul_vnni_8xkx48_t* matmul_tile_n_ker_ = nullptr;
  const std::vector<std::vector<dim_t>> t_shapes_;
  const std::vector<dim_t> src0_perm_shape_;  // src0 shape after perm2031
  const std::vector<dim_t> src1_perm_shape_;  // src1 shape after perm2013
  const dim_t M_, K_, N_;                     // dim of matrix multiplication
  const dim_t bs0_;                           // outer batch size dim
  const dim_t bs1_;                           // innter batch size dim
  const dim_t M_pad_, K_pad_, N_pad_;
  const dim_t tmp0_bytes, tmp1_bytes, tmp_sum_bytes;
  uint8_t* src0_tmp_ = nullptr;
  int8_t* src1_tmp_ = nullptr;
  int32_t* sum_tmp_ = nullptr;
  const data_type dst_type;
  const bool has_binary_add;
  const bool use_thread_exec_;
};

}  // namespace jd

#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_MATMUL_VNNI_P2031_P2013_HPP_
