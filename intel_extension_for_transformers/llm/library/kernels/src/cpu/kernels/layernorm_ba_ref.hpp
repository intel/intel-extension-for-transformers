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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_LAYERNORM_BA_REF_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_LAYERNORM_BA_REF_HPP_

#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "operator_desc.hpp"
#include "src/cpu/cpu_isa.hpp"
#include "src/utils.hpp"
#include <glog/logging.h>
#include <memory>
#include <vector>

namespace jd {
class layernorm_ba_ref_k_t;

class SPARSE_TEST_API_ layernorm_ba_ref_kd_t : public kernel_desc_t {
public:
  explicit layernorm_ba_ref_kd_t(const operator_desc &op_desc)
      : kernel_desc_t(kernel_kind::layernorm_ba), op_desc_(op_desc) {}

  virtual ~layernorm_ba_ref_kd_t() {}

public:
  bool init() override {
    auto op_attrs = op_desc_.attrs();
    auto spec_type = op_attrs["spec_type"];
    if (spec_type == "normal")
      return true;
    else if (spec_type == "direct")
      return true;
    else
      return false;
  };
  DECLARE_COMMON_PD_T(layernorm_ba_ref_k_t, layernorm_ba_ref_kd_t);

public:
  inline std::vector<dim_t> shape() const {
    return op_desc_.tensor_descs()[0].shape();
  }
  const operator_desc &get_operator_desc() const override { return op_desc_; }

private:
  operator_desc op_desc_;
};

class SPARSE_TEST_API_ layernorm_ba_ref_k_t : public kernel_t {
public:
  using kd_t = layernorm_ba_ref_kd_t;
  explicit layernorm_ba_ref_k_t(const std::shared_ptr<const kd_t> &kd)
      : kernel_t(kd) {}
  virtual ~layernorm_ba_ref_k_t() {}
  // Delete move constructor and move operator
  layernorm_ba_ref_k_t(layernorm_ba_ref_k_t &&other) = delete;
  layernorm_ba_ref_k_t &operator=(layernorm_ba_ref_k_t &&other) = delete;
  // Delete copy constructor and copy operator
  layernorm_ba_ref_k_t(const layernorm_ba_ref_k_t &other) = delete;
  layernorm_ba_ref_k_t &operator=(const layernorm_ba_ref_k_t &other) = delete;

public:
  bool init() override { return true; };

  bool execute(const std::vector<const void *> &rt_data) const override;

public:
  const std::shared_ptr<const kd_t> derived_kd() const {
    return std::static_pointer_cast<const kd_t>(kd_);
  }
};

} // namespace jd
#endif
