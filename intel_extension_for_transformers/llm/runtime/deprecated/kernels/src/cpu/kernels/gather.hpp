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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_GATHER_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_GATHER_HPP_

#include <memory>
#include <vector>

#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "kernels/exposed_enum.hpp"
#include "operator_desc.hpp"
#include "src/cpu/cpu_isa.hpp"
#include "src/cpu/jit_domain/jit_gather.hpp"
#include "src/utils.hpp"

namespace jd {
class gather_k_t;

class gather_kd_t : public kernel_desc_t {
  using io = exposed_enum::gather::io;

 public:
  explicit gather_kd_t(const operator_desc& op_desc) : kernel_desc_t(kernel_kind::gather), op_desc_(op_desc) {}

  virtual ~gather_kd_t() {}

 public:
  bool init() override;
  DECLARE_COMMON_PD_T(gather_k_t, gather_kd_t);

 public:
  inline std::vector<dim_t> shape() const override { return op_desc_.tensor_descs()[io::DST].shape(); }
  const jd::operator_desc& get_operator_desc() const override { return op_desc_; }

 private:
  jd::operator_desc op_desc_;
};

class gather_k_t : public kernel_t {
  using io = exposed_enum::gather::io;

 public:
  using kd_t = gather_kd_t;
  explicit gather_k_t(const std::shared_ptr<const kd_t>& kd);
  virtual ~gather_k_t() {}
  // Delete move constructor and move operator
  gather_k_t(gather_k_t&&) = delete;
  gather_k_t& operator=(gather_k_t&&) = delete;
  // Delete copy constructor and copy operator
  gather_k_t(const gather_k_t&) = delete;
  gather_k_t& operator=(const gather_k_t&) = delete;

  bool init() override;
  bool execute(const std::vector<const void*>& rt_data) const override;

  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  std::unique_ptr<jit_generator> jit_kern_ = nullptr;
  const std::vector<tensor_desc> ts_descs;

  const size_t src_axis, idx_axis;
  const int dt_size;
  const int src_axis_size, dst_axis_size;
  const int src_size, idx_size;
  int outer_size, inner_size;
  const std::vector<binaryop_attr> binary_ops;
  std::vector<dim_t> binary_op_sizes;

  const bool has_avx512;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_GATHER_HPP_
