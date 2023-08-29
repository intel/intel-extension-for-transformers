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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_APPENTION_REF_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_APPENTION_REF_HPP_

#include <unordered_map>
#include <vector>
#include <memory>
#include "operator_desc.hpp"
#include "kernel_desc.hpp"
#include "kernel.hpp"
#include "kernels/sparse_data.hpp"

namespace jd {
/**
 * @brief
 *                  Input
 *                    /\
 *                   /  \
 *                  /    \
 *    Matmul Merge(Q K)  Matmul(V)
 *             |              |
 *             |              |
 *           Split            |
 *            / \             |
 *           /   \            |
 *          /     \           |
 *     Reshape  Reshape   Reshape
 *          \    /            |
 *           \  /             |
 *          Matmul            |
 *            |               |
 *            |               |
 *          Softmax           |
 *              \             |
 *               \            |
 *                \           |
 *                 \          |
 *                  \         |
 *                   \        |
 *                    \       |
 *                     \      |
 *                      Matmul
 *                         |
 *                         |
 *                      Output
 */
class attention_ref_k_t;

// TODO(hengyu): hide reference implementation from users
class SPARSE_TEST_API_ attention_ref_kd_t : public kernel_desc_t {
 public:
  explicit attention_ref_kd_t(const operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::attention), op_desc_(op_desc) {}
  virtual ~attention_ref_kd_t();

 public:
  bool init() override;
  DECLARE_COMMON_PD_T(attention_ref_k_t, attention_ref_kd_t);

 public:
  const operator_desc& get_operator_desc() const override { return op_desc_; }
  inline std::vector<dim_t> shape() const override { return {}; }

  std::shared_ptr<const kernel_desc_t> get_kernel_desc(size_t index) const {
    if (index >= kernel_descs_.size()) {
      return nullptr;
    }
    return kernel_descs_[index];
  }

 private:
  operator_desc op_desc_;
  std::vector<std::shared_ptr<const kernel_desc_t>> kernel_descs_;
  /**
   * @brief Create and add a sub-kernel
   *
   * @tparam T_kd kernel desc type
   * @param op_desc kernel's operator desc
   * @param name kernel name used for logging
   * @return true if a new kernel is created successfully and added to the vector
   * @return false if it fails to create the kernel
   */
  template <typename T_kd>
  bool add_kernel_desc(const operator_desc& op_desc, const char* name);
  void* fused_bias_addr_ = nullptr;
  void* fused_scales_addr_ = nullptr;
  bsr_data_t<int8_t>* qk_sparse_ptr_ = nullptr;
  bsr_data_t<int8_t>* v_sparse_ptr_ = nullptr;
  void* qk_weight_addr_ = nullptr;
};

class SPARSE_TEST_API_ attention_ref_k_t : public kernel_t {
 public:
  using kd_t = attention_ref_kd_t;
  explicit attention_ref_k_t(const std::shared_ptr<const kernel_desc_t>& kd) : kernel_t(kd) {}
  virtual ~attention_ref_k_t();

 public:
  bool init() override;
  bool execute(const std::vector<const void*>& rt_data) const override;

 private:
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }
  inline const operator_desc& ker_opdesc(size_t idx) const { return kernels_[idx]->kd()->get_operator_desc(); }
  bool setup_kernel();
  void setup_memory();

 private:
  bool execute_(const std::vector<const void*>& rt_data) const;
  // std::vector<const std::vector<const void*>> set_input_output(const std::vector<const void*>& rt_data) const;
  std::vector<const void*> set_input_output(int index, const std::vector<const void*>& rt_data) const;

 private:
  std::vector<std::shared_ptr<const kernel_t>> kernels_;
  std::vector<std::vector<char*>> mem_{};
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_APPENTION_REF_HPP_
