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

#ifndef ENGINE_SPARSELIB_SRC_GPU_OCL_ENGINE_HPP_
#define ENGINE_SPARSELIB_SRC_GPU_OCL_ENGINE_HPP_
#include <CL/cl.h>
#include <vector>
#include <memory>

#include "engine.hpp"
#include "impl_list_item.hpp"
#include "param_types.hpp"

namespace jd {
#define GPU_INSTANCE(...) impl_list_item_t(type_deduction_helper_t<__VA_ARGS__::kd_t>())
#define NULL_INSTANCE(...) impl_list_item_t(nullptr)

class gpu_ocl_engine_t : public engine_t {
 public:
  gpu_ocl_engine_t();
  virtual ~gpu_ocl_engine_t() {}

 public:
  bool create_stream(stream_t** stream) const override;
  bool create_memory_storage(memory_storage_t** storage) const override;
  const std::vector<impl_list_item_t>* get_implementation_list(const operator_desc& op_desc) const override;
  bool create_kernel(const operator_desc& op_desc, std::shared_ptr<kernel_t>& kernel,
                     const stream_t* stream) const override;
  cl_device_id get_device() const { return device_; }
  cl_context get_context() const { return context_; }
  cl_platform_id get_platform() const { return platform_; }

 public:
  static const std::vector<impl_list_item_t> empty_list;

 private:
  cl_device_id findGPU();

 private:
  cl_device_id device_;
  cl_context context_;
  cl_platform_id platform_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_GPU_ENGINE_HPP_
