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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_DYNAMIC_QUANT_MATMUL_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_DYNAMIC_QUANT_MATMUL_HPP_

#include <memory>
#include <vector>
#include <utility>
#include "src/cpu/cpu_isa.hpp"
#include "operator_desc.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "src/utils.hpp"
#include "kernels/exposed_enum.hpp"
#include "kernels/dynamic_quant_matmul_types.hpp"
#include "src/cpu/jit_domain/jit_amx_s8s8_dynamic_quant_matmul.hpp"
#include "src/cpu/jit_domain/jit_amx_s8s8_dynamic_dequant_matmul.hpp"
#include "src/cpu/jit_domain/jit_dynamic_quant_matmul_reduce_scale_quant.hpp"

namespace jd {

enum split_n_strategy { based_L2_cache, based_M };

class dynamic_quant_matmul_k_t;
class dynamic_quant_matmul_kd_t : public kernel_desc_t {
 public:
  explicit dynamic_quant_matmul_kd_t(const operator_desc& op_desc);

  virtual ~dynamic_quant_matmul_kd_t() {}

 public:
  bool init() override;
  DECLARE_COMMON_PD_T(dynamic_quant_matmul_k_t, dynamic_quant_matmul_kd_t);

 public:
  const operator_desc& get_operator_desc() const override { return op_desc_; }
  inline std::vector<dim_t> shape() const override { return prob_size_; }
  const std::vector<ssd::dynamic_quant_matmul_param_t>& params() const { return params_; }
  const std::vector<dynamic_quant_matmul_reduce_scale_quant_param_t>& reduce_scale_quant_params() const {
    return reduce_scale_quant_params_;
  }
  bool check_split_execute() const { return split_execute_; }
  bool split_execute_init();
  const std::pair<int, int>& get_assign_cores() const { return assign_cores_; }

 private:
  operator_desc op_desc_;
  std::vector<ssd::dynamic_quant_matmul_param_t> params_;
  std::vector<dynamic_quant_matmul_reduce_scale_quant_param_t> reduce_scale_quant_params_;
  std::vector<dim_t> prob_size_;
  static constexpr int L2_size_ = 1 << 21;  // 2 Mb L2 cache.
  int core_num;
  std::pair<int, int> assign_cores_;
  split_n_strategy split_strategy_;
  bool split_execute_ = false;
};

class dynamic_quant_matmul_k_t : public kernel_t {
 public:
  using kd_t = dynamic_quant_matmul_kd_t;
  explicit dynamic_quant_matmul_k_t(const std::shared_ptr<const kd_t>& kd) : kernel_t(kd) {}
  virtual ~dynamic_quant_matmul_k_t() {
    for (auto&& ker : jit_kers_) safe_delete(ker);
    for (auto&& ker : jit_s8s8_dynamic_dequant_kers_) safe_delete(ker);
    for (auto&& ker : jit_reduce_scale_quant_kers_) safe_delete(ker);
  }
  // Delete move constructor and move operator
  dynamic_quant_matmul_k_t(dynamic_quant_matmul_k_t&& other) = delete;
  dynamic_quant_matmul_k_t& operator=(dynamic_quant_matmul_k_t&& other) = delete;
  // Delete copy constructor and copy operator
  dynamic_quant_matmul_k_t(const dynamic_quant_matmul_k_t& other) = delete;
  dynamic_quant_matmul_k_t& operator=(const dynamic_quant_matmul_k_t& other) = delete;

 public:
  bool init() override;

  bool execute(const std::vector<const void*>& rt_data) const override;

  size_t get_workspace_size() const override;

 private:
  bool split_execute_init();
  bool split_execute(const std::vector<const void*>& rt_data) const;
  char* get_data_ptr(const void* ptr, int offset) const {
    return reinterpret_cast<char*>(const_cast<void*>(ptr)) + offset;
  }

 public:
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  std::vector<jit_amx_s8s8_dynamic_quant_matmul_t*> jit_kers_;
  std::vector<jit_amx_s8s8_dynamic_dequant_matmul_t*> jit_s8s8_dynamic_dequant_kers_;
  std::vector<jit_dynamic_quant_matmul_reduce_scale_quant_t*> jit_reduce_scale_quant_kers_;
  std::vector<int> m_offset_list_;
  std::vector<int> n_offset_list_;
  std::vector<int> scale_offset_list_;
  int total_tmp_buf_size_;
  int single_tmp_buf_size_;
  int bf16_tmp_buf_offset_;
  int activation_cores_;
  int weight_cores_;
  bool has_bias_;
  bool split_execute_ = false;
  bool quant_stage_ = true;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_DYNAMIC_QUANT_MATMUL_HPP_
