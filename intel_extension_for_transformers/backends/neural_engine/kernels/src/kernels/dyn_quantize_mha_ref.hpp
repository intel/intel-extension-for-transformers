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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_DYN_QUANTIZE_MHA_REF_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_DYN_QUANTIZE_MHA_REF_HPP_

#include <memory>
#include <vector>

#include "amx_utils.hpp"
#include "cpu_isa.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "kernels/dyn_quantize_mha_types.hpp"
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
class dyn_quantize_mha_ref_k_t;

class SPARSE_API_ dyn_quantize_mha_ref_kd_t : public kernel_desc_t {
 public:
  using io = ssd::dyn_quantize_mha_io::io;
  explicit dyn_quantize_mha_ref_kd_t(const jd::operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::dyn_quantize_mha), op_desc_(op_desc) {}
  virtual ~dyn_quantize_mha_ref_kd_t() {}

  bool init() override;
  DECLARE_COMMON_PD_T(dyn_quantize_mha_ref_k_t, dyn_quantize_mha_ref_kd_t);

  const jd::operator_desc& get_operator_desc() const override { return op_desc_; }
  inline std::vector<dim_t> shape() const override {
    return {
        op_desc_.tensor_descs()[io::Q].shape()[0],  // batch_size
        op_desc_.tensor_descs()[io::Q].shape()[2],  // head_num
        op_desc_.tensor_descs()[io::Q].shape()[1],  // M
        op_desc_.tensor_descs()[io::Q].shape()[3],  // head_size
        op_desc_.tensor_descs()[io::K].shape()[1],  // N
    };
  }

 private:
  jd::operator_desc op_desc_;
};

class SPARSE_API_ dyn_quantize_mha_ref_k_t : public kernel_t {
 public:
  using io = ssd::dyn_quantize_mha_io::io;
  using kd_t = dyn_quantize_mha_ref_kd_t;
  explicit dyn_quantize_mha_ref_k_t(const std::shared_ptr<const kernel_desc_t>& kd);
  virtual ~dyn_quantize_mha_ref_k_t() {}
  // Delete move constructor and move operator
  dyn_quantize_mha_ref_k_t(dyn_quantize_mha_ref_k_t&&) = delete;
  dyn_quantize_mha_ref_k_t& operator=(dyn_quantize_mha_ref_k_t&&) = delete;
  // Delete copy constructor and copy operator
  dyn_quantize_mha_ref_k_t(const dyn_quantize_mha_ref_k_t&) = delete;
  dyn_quantize_mha_ref_k_t& operator=(const dyn_quantize_mha_ref_k_t&) = delete;

  bool init() override;
  bool execute(const std::vector<const void*>& rt_data) const override;
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  std::vector<std::vector<dim_t>> t_shapes_;
  int32_t batch_size_, head_num_, M_, head_size_, N_;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_DYN_QUANTIZE_MHA_REF_HPP_
