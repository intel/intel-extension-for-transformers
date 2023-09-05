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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_GROUPNORM_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_GROUPNORM_HPP_

#include <memory>
#include <utility>
#include <vector>
#include <functional>
#include "src/cpu/cpu_isa.hpp"
#include "operator_desc.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "src/utils.hpp"
#include "kernels/exposed_enum.hpp"
#include "src/cpu/jit_domain/jit_groupnorm.hpp"

namespace jd {
class groupnorm_k_t;
class groupnorm_kd_t : public kernel_desc_t {
 public:
  explicit groupnorm_kd_t(const operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::groupnorm), op_desc_(op_desc) {}

  virtual ~groupnorm_kd_t() {}

 public:
  bool init() override;
  DECLARE_COMMON_PD_T(groupnorm_k_t, groupnorm_kd_t);

 public:
  inline std::vector<dim_t> shape() const override { return op_desc_.tensor_descs()[0].shape(); }
  const operator_desc& get_operator_desc() const override { return op_desc_; }
  const groupnorm_param_t& param() const { return param_; }

 private:
  operator_desc op_desc_;
  groupnorm_param_t param_;
};

class groupnorm_k_t : public kernel_t {
 public:
  using kd_t = groupnorm_kd_t;
  enum parallel_mode { parallelG, parallelC };
  explicit groupnorm_k_t(const std::shared_ptr<const kd_t>& kd) : kernel_t(kd) {}
  virtual ~groupnorm_k_t() {
    safe_delete(jit_sum_ker_);
    safe_delete(jit_norm_ker_);
    safe_delete(jit_groupwise_ker_);
  }
  // Delete move constructor and move operator
  groupnorm_k_t(groupnorm_k_t&& other) = delete;
  groupnorm_k_t& operator=(groupnorm_k_t&& other) = delete;
  // Delete copy constructor and copy operator
  groupnorm_k_t(const groupnorm_k_t& other) = delete;
  groupnorm_k_t& operator=(const groupnorm_k_t& other) = delete;

 public:
  bool init() override;

  bool execute(const std::vector<const void*>& rt_data) const override;
  void parallelC_execute(const std::vector<const void*>& rt_data) const;
  void parallelG_execute(const std::vector<const void*>& rt_data) const;

 public:
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }
  size_t get_workspace_size() const override;

 private:
  jit_channelwise_sum_t* jit_sum_ker_ = nullptr;
  jit_channelwise_norm_t* jit_norm_ker_ = nullptr;
  jit_groupnorm_t* jit_groupwise_ker_ = nullptr;
  int64_t HW_;
  parallel_mode mode_;
  int dt_bytewidth_;
  int batchs_;
  int channels_;
  int groups_;
  int channels_per_group_;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_GROUPNORM_HPP_
