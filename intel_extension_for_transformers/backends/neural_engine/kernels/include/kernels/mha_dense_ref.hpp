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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_MHA_DENSE_REF_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_MHA_DENSE_REF_HPP_

#include <glog/logging.h>

#include <memory>
#include <vector>

#include "cpu_isa.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "kernels/mha_dense_types.hpp"
#include "operator_desc.hpp"
#include "utils.hpp"

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
class mha_dense_ref_k_t;

class mha_dense_ref_kd_t : public kernel_desc_t {
 public:
  explicit mha_dense_ref_kd_t(const jd::operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::mha_dense), op_desc_(op_desc) {}
  virtual ~mha_dense_ref_kd_t() {}

 public:
  bool init() override;
  DECLARE_COMMON_PD_T(mha_dense_ref_k_t, mha_dense_ref_kd_t);

 public:
  const jd::operator_desc& get_operator_desc() const override { return op_desc_; }
  inline std::vector<dim_t> shape() const override { return op_desc_.tensor_descs().front().shape(); }
  inline dim_t bs() const { return op_desc_.tensor_descs().front().shape()[0]; }
  inline dim_t seq_len() const { return op_desc_.tensor_descs().front().shape()[1]; }
  inline dim_t head_num() const { return op_desc_.tensor_descs().front().shape()[2]; }
  inline dim_t head_size() const { return op_desc_.tensor_descs().front().shape()[3]; }
  inline bool merged_QKV() const { return merged_QKV_; }
  inline bool approx_exp() const { return approx_exp_; }
  inline data_type dst_dt() const { return dst_dt_; }

 private:
  jd::operator_desc op_desc_;
  bool merged_QKV_;
  bool approx_exp_;  // approx exp for the same behavior as the mha approx kernel
  data_type dst_dt_;
};

class mha_dense_ref_k_t : public kernel_t {
 public:
  using kd_t = mha_dense_ref_kd_t;
  explicit mha_dense_ref_k_t(const std::shared_ptr<const kernel_desc_t>& kd);
  virtual ~mha_dense_ref_k_t() {}

  // Delete move/copy constructor/move operator
  mha_dense_ref_k_t(mha_dense_ref_k_t&&) = delete;
  mha_dense_ref_k_t& operator=(mha_dense_ref_k_t&&) = delete;
  mha_dense_ref_k_t(const mha_dense_ref_k_t&) = delete;
  mha_dense_ref_k_t& operator=(const mha_dense_ref_k_t&) = delete;

 public:
  bool init() override;
  bool execute(const std::vector<const void*>& rt_data) const override;
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  template <float (*func_exp)(float)>
  bool execute_(const std::vector<const void*>& rt_data) const;
  const bool approx_exp;
  const data_type dst_dt_;
  const int bs_, seq_len_, head_num_, head_size_, ld_src_, ld_dst_;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_MHA_DENSE_REF_HPP_
