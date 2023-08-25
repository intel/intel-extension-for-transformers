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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_LAYERNORMALIZED_SPMM_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_LAYERNORMALIZED_SPMM_HPP_

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "src/cpu/cpu_isa.hpp"
#include "operator_desc.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "src/utils.hpp"

namespace jd {
class layernormalized_spmm_k_t;

class layernormalized_spmm_kd_t : public kernel_desc_t {
 public:
  explicit layernormalized_spmm_kd_t(const operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::layernormalized_spmm), op_desc_(op_desc) {}

  virtual ~layernormalized_spmm_kd_t() {
    safe_delete(spmm_desc);
    safe_delete(lnorm_desc);
  }

 public:
  bool init() override;
  DECLARE_COMMON_PD_T(layernormalized_spmm_k_t, layernormalized_spmm_kd_t);

 public:
  const operator_desc& get_operator_desc() const override { return op_desc_; }
  operator_desc* spmm_desc = nullptr;
  operator_desc* lnorm_desc = nullptr;
  bool split_output;

 private:
  operator_desc op_desc_;
};

class layernormalized_spmm_k_t : public kernel_t {
 public:
  using kd_t = layernormalized_spmm_kd_t;
  explicit layernormalized_spmm_k_t(const std::shared_ptr<const kd_t>& kd) : kernel_t(kd) { kernels_.resize(2); }
  virtual ~layernormalized_spmm_k_t() {}
  // Delete move constructor and move operator
  layernormalized_spmm_k_t(layernormalized_spmm_k_t&& other) = delete;
  layernormalized_spmm_k_t& operator=(layernormalized_spmm_k_t&& other) = delete;
  // Delete copy constructor and copy operator
  layernormalized_spmm_k_t(const layernormalized_spmm_k_t& other) = delete;
  layernormalized_spmm_k_t& operator=(const layernormalized_spmm_k_t& other) = delete;

 public:
  bool init() override;

  bool execute(const std::vector<const void*>& rt_data) const override;

 public:
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  std::vector<std::shared_ptr<const kernel_t>> kernels_;
  bool split_output;
};

}  // namespace jd

#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_LAYERNORMALIZED_SPMM_HPP_
