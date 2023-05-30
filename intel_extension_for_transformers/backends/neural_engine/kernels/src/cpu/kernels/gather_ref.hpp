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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_GATHER_REF_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_GATHER_REF_HPP_

#include <glog/logging.h>
#include <memory>
#include <vector>
#include "src/cpu/cpu_isa.hpp"
#include "operator_desc.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "src/utils.hpp"

namespace jd {
class gather_ref_k_t;

class gather_ref_kd_t : public kernel_desc_t {
 public:
  explicit gather_ref_kd_t(const operator_desc& op_desc) : kernel_desc_t(kernel_kind::gather), op_desc_(op_desc) {}

  virtual ~gather_ref_kd_t() {}

  bool init() override { return true; };
  DECLARE_COMMON_PD_T(gather_ref_k_t, gather_ref_kd_t);

  inline std::vector<dim_t> shape() const { return op_desc_.tensor_descs()[2].shape(); }
  const operator_desc& get_operator_desc() const override { return op_desc_; }

 private:
  operator_desc op_desc_;
};

class gather_ref_k_t : public kernel_t {
 public:
  using kd_t = gather_ref_kd_t;
  explicit gather_ref_k_t(const std::shared_ptr<const kd_t>& kd) : kernel_t(kd) {}
  virtual ~gather_ref_k_t() {}

  bool init() override { return true; };
  bool execute(const std::vector<const void*>& rt_data) const override;

  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_GATHER_REF_HPP_
