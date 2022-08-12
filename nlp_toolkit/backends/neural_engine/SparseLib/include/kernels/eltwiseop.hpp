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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_ELTWISEOP_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_ELTWISEOP_HPP_
#include "operator_desc.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "eltwiseop_types.hpp"
#include "jit_domain/jit_eltwiseop.hpp"
#include <vector>
#include "utils.hpp"

namespace jd {
class eltwiseop_k_t;

class eltwiseop_kd_t : public kernel_desc_t {
 public:
  explicit eltwiseop_kd_t(const jd::operator_desc& op_desc) : kernel_desc_t(kernel_kind::eltwiseop), op_desc_(op_desc) {
    params_.element_num = 1;
    auto& ts_desc = op_desc_.tensor_descs();
    for (auto&& i : ts_desc[0].shape()) params_.element_num *= i;

    auto dt = ts_desc[0].dtype();
    params_.dt = dt;
  };
  virtual ~eltwiseop_kd_t() {}

 public:
  bool init() override;
  DECLARE_COMMON_PD_T(eltwiseop_k_t, eltwiseop_kd_t);

 public:
  inline std::vector<dim_t> shape() const { return op_desc_.tensor_descs()[0].shape(); }
  const jd::operator_desc& operator_desc() const override { return op_desc_; }
  const ssd::eltwiseop_param_t& params() const { return params_; }

 private:
  jd::operator_desc op_desc_;
  ssd::eltwiseop_param_t params_;
};

class eltwiseop_k_t : public kernel_t {
 public:
  using kd_t = eltwiseop_kd_t;
  explicit eltwiseop_k_t(const std::shared_ptr<const kd_t>& kd) : kernel_t(kd) {}
  virtual ~eltwiseop_k_t() {}

 public:
  bool init() override;

  bool execute(const std::vector<const void*>& rt_data) const override;

 public:
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  bool eltwiseop_kernel_create(jit_eltwiseop_t** ker_pp, const ssd::eltwiseop_param_t& param);

 private:
  jit_eltwiseop_t* jit_kers_;
  int64_t nthr_;
  std::vector<ssd::eltwiseop_data_t*> td;
};

}  // namespace jd
#endif
