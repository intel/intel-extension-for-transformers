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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_POSTOP_DEFAULT_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_POSTOP_DEFAULT_HPP_

#include <string>
#include <memory>
#include <vector>
#include "operator_desc.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "kernels/postop_types.hpp"
#include "jit_domain/jit_postop_default.hpp"
#include "utils.hpp"

namespace jd {
class postop_default_k_t;

class postop_default_kd_t : public kernel_desc_t {
 public:
  explicit postop_default_kd_t(const jd::operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::postop), op_desc_(op_desc) {
    params_.element_num = 1;
    auto& ts_desc = op_desc_.tensor_descs();
    for (auto&& i : ts_desc[0].shape()) params_.element_num *= i;
    auto dt = ts_desc[0].dtype();
    switch (dt) {
      case jd::data_type::fp32:
        params_.dt = jd::ssd::data_type::fp32;
        break;
      case jd::data_type::bf16:
        params_.dt = jd::ssd::data_type::bf16;
        break;
      default:
        std::runtime_error(std::string("unsupporting data type."));
        break;
    }
  }
  virtual ~postop_default_kd_t() {}

 public:
  bool init() override;
  DECLARE_COMMON_PD_T(postop_default_k_t, postop_default_kd_t);

 public:
  inline std::vector<dim_t> shape() const { return op_desc_.tensor_descs()[0].shape(); }
  const jd::operator_desc& operator_desc() const override { return op_desc_; }
  const ssd::postop_param_t& params() const { return params_; }

 private:
  jd::operator_desc op_desc_;
  ssd::postop_param_t params_;
};

class postop_default_k_t : public kernel_t {
 public:
  using kd_t = postop_default_kd_t;
  explicit postop_default_k_t(const std::shared_ptr<const kd_t>& kd) : kernel_t(kd) {}
  virtual ~postop_default_k_t() {
    if (jit_kers_ != nullptr) {
      delete jit_kers_;
      jit_kers_ = nullptr;
    }
  }
  // Delete move constructor and move operator
  postop_default_k_t(postop_default_k_t&& other) = delete;
  postop_default_k_t& operator=(postop_default_k_t&& other) = delete;
  // Delete copy constructor and copy operator
  postop_default_k_t(const postop_default_k_t& other) = delete;
  postop_default_k_t& operator=(const postop_default_k_t& other) = delete;

 public:
  bool init() override;

  bool execute(const std::vector<const void*>& rt_data) const override;

 public:
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  bool postop_kernel_create(jit_postop_default_t** ker_pp, const ssd::postop_param_t& param);

 private:
  jit_postop_default_t* jit_kers_;
  int64_t nthr_;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_POSTOP_DEFAULT_HPP_
