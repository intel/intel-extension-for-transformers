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

#ifndef ENGINE_SPARSELIB_SRC_CPU_KERNELS_SLICE_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_KERNELS_SLICE_HPP_

#include <memory>
#include <vector>

#include "src/cpu/jit_domain/jit_slice.hpp"
#include "kernel.hpp"
#include "kernel_desc.hpp"
#include "operator_desc.hpp"

namespace jd {
class slice_k_t;

class slice_kd_t : public kernel_desc_t {
 public:
  explicit slice_kd_t(const jd::operator_desc& op_desc) : kernel_desc_t(kernel_kind::slice), op_desc_(op_desc) {}
  virtual ~slice_kd_t() {}

  bool init() override;
  DECLARE_COMMON_PD_T(slice_k_t, slice_kd_t);

  inline std::vector<dim_t> shape() const override { return op_desc_.tensor_descs()[1].shape(); }
  const operator_desc& get_operator_desc() const override { return op_desc_; }

  int axis() const { return axis_; }
  int begin() const { return begin_; }
  int step() const { return step_; }

 private:
  jd::operator_desc op_desc_;
  int axis_;
  int begin_;
  int step_;
};

class slice_k_t : public kernel_t {
 public:
  using kd_t = slice_kd_t;
  explicit slice_k_t(const std::shared_ptr<const kd_t>& kd);
  virtual ~slice_k_t() {}
  // Delete move constructor and move operator
  slice_k_t(slice_k_t&&) = delete;
  slice_k_t& operator=(slice_k_t&&) = delete;
  // Delete copy constructor and copy operator
  slice_k_t(const slice_k_t&) = delete;
  slice_k_t& operator=(const slice_k_t&) = delete;

  bool init() override;
  bool execute(const std::vector<const void*>& rt_data) const override;

  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  std::unique_ptr<jit_slice_t> jit_kern_ = nullptr;
  const std::vector<tensor_desc> ts_descs;
  const std::vector<dim_t>& src_shape;
  const std::vector<dim_t>& dst_shape;

  const int axis;
  const int begin;
  const int step;
  const int dt_size;
  const int outer_size;
  const int src_axis_size;
  const int dst_axis_size;
  const int inner_size;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_KERNELS_SLICE_HPP_
