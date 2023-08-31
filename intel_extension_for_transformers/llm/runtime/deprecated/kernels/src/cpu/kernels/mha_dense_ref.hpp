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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_MHA_DENSE_REF_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_MHA_DENSE_REF_HPP_

#include <glog/logging.h>

#include <array>
#include <memory>
#include <vector>

#include "kernels/exposed_enum.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "operator_desc.hpp"

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

class SPARSE_TEST_API_ mha_dense_ref_kd_t : public kernel_desc_t {
  using io = exposed_enum::mha_dense::io;
  using io_src = exposed_enum::mha_dense_src::src;
  using io_dst = exposed_enum::mha_dense_dst::dst;
  using io_shape = exposed_enum::mha_dense_shape::shape;

 public:
  explicit mha_dense_ref_kd_t(const operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::mha_dense), op_desc_(op_desc) {}
  virtual ~mha_dense_ref_kd_t() {}

 public:
  bool init() override;
  DECLARE_COMMON_PD_T(mha_dense_ref_k_t, mha_dense_ref_kd_t);

 public:
  const operator_desc& get_operator_desc() const override { return op_desc_; }
  inline std::vector<dim_t> shape() const override { return op_desc_.tensor_descs()[io::DST].shape(); }
  inline dim_t bs() const { return op_desc_.tensor_descs()[io::SRC_Q].shape()[0]; }
  inline dim_t sl_m() const { return op_desc_.tensor_descs()[io::SRC_Q].shape()[1]; }
  inline dim_t sl_n() const { return op_desc_.tensor_descs()[io::SRC_K].shape()[1]; }
  inline dim_t head_num() const { return op_desc_.tensor_descs()[io::SRC_Q].shape()[2]; }
  inline dim_t head_size() const { return op_desc_.tensor_descs()[io::SRC_Q].shape()[3]; }
  inline bool merged_QKV() const { return merged_QKV_; }
  inline bool approx_exp() const { return approx_exp_; }
  inline bool stable_softmax() const { return stable_softmax_; }
  inline data_type dst_dt() const { return dst_dt_; }
  inline format_type kv_ft() const { return op_desc_.tensor_descs()[io::SRC_K].ftype(); }

 private:
  operator_desc op_desc_;
  bool merged_QKV_;
  bool approx_exp_;      // approx exp for the same behavior as the mha approx kernel
  bool stable_softmax_;  // whether to minus max before calculating exp
  data_type dst_dt_;
};

class SPARSE_TEST_API_ mha_dense_ref_k_t : public kernel_t {
  using io = exposed_enum::mha_dense::io;
  using io_src = exposed_enum::mha_dense_src::src;
  using io_dst = exposed_enum::mha_dense_dst::dst;
  using io_shape = exposed_enum::mha_dense_shape::shape;

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
  [[deprecated("Please use exec_context_t instead of rt_data")]] bool execute(
      const std::vector<const void*>& rt_data) const override;
  bool execute(const exec_context_t& context) const override;
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }
  size_t get_workspace_size() const override { return workspace_size_; }

 private:
  template <float (*func_exp)(float)>
  bool execute_(const exec_context_t& ctx) const;
  const std::vector<tensor_desc>& ts_descs_;
  const bool has_badd;
  const bool approx_exp;
  const bool stable_softmax;
  const data_type dst_dt_, dst_v_;
  const format_type kv_ft_;
  const int bs_, sl_m_, sl_n_, head_num_, head_size_;  // in #elements
  const bool is_dynq10n_dst;
  const size_t workspace_size_;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_MHA_DENSE_REF_HPP_
