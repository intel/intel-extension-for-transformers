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

#ifndef ENGINE_SPARSELIB_SRC_MATMUL_REF_HPP_
#define ENGINE_SPARSELIB_SRC_MATMUL_REF_HPP_

#include <CL/cl.h>
#include <vector>
#include <memory>

#include "kernel_desc.hpp"
#include "kernel.hpp"
namespace jd {
class gpu_ocl_matmul_ref_k_t;
class SPARSE_TEST_API_ gpu_ocl_matmul_ref_kd_t : public kernel_desc_t {
 public:
  explicit gpu_ocl_matmul_ref_kd_t(const operator_desc& op_desc)
      : kernel_desc_t(kernel_kind::matmul), op_desc_(op_desc) {}
  virtual ~gpu_ocl_matmul_ref_kd_t() {}

 public:
  bool init() override { return true; }
  DECLARE_COMMON_PD_T(gpu_ocl_matmul_ref_k_t, gpu_ocl_matmul_ref_kd_t);

 public:
  const operator_desc& get_operator_desc() const override { return op_desc_; }
  inline dim_t M() const { return op_desc_.tensor_descs()[0].shape()[0]; }
  inline dim_t N() const { return op_desc_.tensor_descs()[1].shape()[1]; }
  inline dim_t K() const { return op_desc_.tensor_descs()[1].shape()[1]; }

 private:
  operator_desc op_desc_;
};

class SPARSE_TEST_API_ gpu_ocl_matmul_ref_k_t : public kernel_t {
 public:
  using kd_t = gpu_ocl_matmul_ref_kd_t;
  explicit gpu_ocl_matmul_ref_k_t(const std::shared_ptr<const kernel_desc_t>& kd)
      : kernel_t(kd), M_(derived_kd()->M()), N_(derived_kd()->N()), K_(derived_kd()->K()) {}
  virtual ~gpu_ocl_matmul_ref_k_t() {}

 public:
  bool init() override;
  bool init(const exec_context_t& context) override;
  bool execute() const override;

 private:
  const std::shared_ptr<const kd_t> derived_kd() const { return std::static_pointer_cast<const kd_t>(kd_); }

 private:
  const dim_t M_;
  const dim_t N_;
  const dim_t K_;
  cl_kernel cl_kernel_;
  cl_command_queue cl_queue_;
  const size_t local_[2] = {8, 8};
  const size_t global_[2] = {static_cast<size_t>(M_), static_cast<size_t>(N_)};
  std::vector<void*> handle_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_MATMUL_REF_HPP_
