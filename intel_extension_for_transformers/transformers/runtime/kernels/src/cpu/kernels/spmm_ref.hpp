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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_SPMM_REF_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_SPMM_REF_HPP_

#include <glog/logging.h>
#include <vector>
#include <memory>
#include "src/cpu/cpu_isa.hpp"
#include "operator_desc.hpp"
#include "kernel_desc.hpp"
#include "kernel.hpp"
#include "kernels/spmm_types.hpp"
#include "kernels/sparse_data.hpp"

namespace jd {
// By convention,
//   1. xxxx_kd_t is the descriptor of a specific derived primitive/kernel.
//   2. xxxx_k_t is a specific derived primitive/kernel.
//   3. jit_xxxx_t is JIT assembly implementation of a specific derived primitive/kernel.
//   where, "xxxx" represents an algorithm, such as brgemm, GEMM and so on.
class spmm_ref_k_t;
/**
 * @brief a derived kernel descriptor. ref_param_t is its class member.
 */
class SPARSE_TEST_API_ spmm_ref_kd_t : public kernel_desc_t {
 public:
  explicit spmm_ref_kd_t(const operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::sparse_matmul), op_desc_(op_desc) {}
  virtual ~spmm_ref_kd_t() {}

 public:
  bool init() override;
  // kernel_desc_t::create_primitive() override.
  DECLARE_COMMON_PD_T(spmm_ref_k_t, spmm_ref_kd_t);

 public:
  const operator_desc& get_operator_desc() const override { return op_desc_; }
  inline std::vector<dim_t> shape() const { return {M(), K(), N()}; }
  inline dim_t M() const { return op_desc_.tensor_descs()[ssd::WEI].shape()[0]; }
  inline dim_t K() const { return op_desc_.tensor_descs()[ssd::WEI].shape()[1]; }
  inline dim_t BN() const {
    auto& ds_src = op_desc_.tensor_descs()[ssd::SRC].shape();
    return ds_src[ds_src.size() - 1];
  }
  inline dim_t N() const {
    auto& ds_src = op_desc_.tensor_descs()[ssd::SRC].shape();
    return BN() * (ds_src.size() == 3 ? ds_src[0] : 1);
  }
  inline dim_t BM() const { return BM_; }
  inline bool has_bias() const { return !op_desc_.tensor_descs()[ssd::BIAS].shape().empty(); }
  inline data_type wei_type() const { return op_desc_.tensor_descs()[ssd::WEI].dtype(); }
  inline data_type dst_type() const { return op_desc_.tensor_descs()[ssd::DST].dtype(); }

 private:
  operator_desc op_desc_;
  dim_t BM_;
};

/**
 * @brief a derived kernel. kd_t and jit_domain are its class members.
 */
class SPARSE_TEST_API_ spmm_ref_k_t : public kernel_t {
 public:
  using kd_t = spmm_ref_kd_t;
  explicit spmm_ref_k_t(const std::shared_ptr<const kd_t>& kd)
      : kernel_t(kd),
        M_(derived_kd()->M()),
        N_(derived_kd()->N()),
        K_(derived_kd()->K()),
        BM_(derived_kd()->BM()),
        BN_(derived_kd()->BN()) {}
  virtual ~spmm_ref_k_t() {}

  // Delete move constructor and move operator
  spmm_ref_k_t(spmm_ref_k_t&& other) = delete;
  spmm_ref_k_t& operator=(spmm_ref_k_t&& other) = delete;
  // Delete copy constructor and copy operator
  spmm_ref_k_t(const spmm_ref_k_t& other) = delete;
  spmm_ref_k_t& operator=(const spmm_ref_k_t& other) = delete;

 public:
  bool init() override;
  bool execute(const std::vector<const void*>& rt_data) const override;
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  inline data_type wei_type() const { return derived_kd()->wei_type(); }
  inline data_type dst_type() const { return derived_kd()->dst_type(); }
  bool execute_s8_(const std::vector<const void*>& rt_data) const;
  bool execute_bf16_(const std::vector<const void*>& rt_data) const;
  bool execute_f32_(const std::vector<const void*>& rt_data) const;

 private:
  const dim_t M_;
  const dim_t N_;
  const dim_t K_;
  const dim_t BM_;
  const dim_t BN_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_SPMM_REF_HPP_
