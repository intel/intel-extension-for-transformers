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

#include "gpu_ocl_engine.hpp"
#include "kernel_cache.hpp"
#include "src/gpu/stream/gpu_ocl_stream.hpp"
#include "src/gpu/memory_storage/gpu_ocl_memory_storage.hpp"
#include "src/gpu/kernels/opencl/common.hpp"
#include "src/singleton.hpp"

#define MAX_NUM_DEVICES 16
#define MAX_DEVICE_NAME 1024
#define CURRENT_DEVICE 0

namespace jd {
const std::vector<impl_list_item_t> gpu_ocl_engine_t::empty_list = {};

// C API forward declaration.
#define DECLARE_IMPL_LIST(kind) \
  const std::vector<impl_list_item_t>* get_gpu_##kind##_impl_list(const operator_desc& op_desc);

DECLARE_IMPL_LIST(matmul);

#undef DECLARE_IMPL_LIST

const std::vector<impl_list_item_t>* gpu_ocl_engine_t::get_implementation_list(const operator_desc& op_desc) const {
  // Call C API.
#define CASE(kind)        \
  case kernel_kind::kind: \
    return get_gpu_##kind##_impl_list(op_desc);

  switch (op_desc.kernel_kind()) {
    CASE(matmul);
    default:
      return &gpu_ocl_engine_t::empty_list;
  }

#undef CASE
}

cl_device_id gpu_ocl_engine_t::findGPU() {
  std::vector<cl_platform_id> platforms(16);
  cl_uint count;
  cl_int err;
  err = clGetPlatformIDs(platforms.size(), platforms.data(), &count);
  checkError(err, __LINE__);
  platforms.resize(count);

  for (auto platform : platforms) {
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err == CL_SUCCESS)
      return device;
    else if (err != CL_DEVICE_NOT_FOUND)
      checkError(err, __LINE__);
  }

  printf("No GPUs found.\n");
  return device_;
}

gpu_ocl_engine_t::gpu_ocl_engine_t() : engine_t(engine_kind::gpu, runtime_kind::opencl) {
  cl_int err;
  device_ = findGPU();
  context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
  checkError(err, __LINE__);
}

bool gpu_ocl_engine_t::create_stream(stream_t** stream) const {
  *stream = new gpu_ocl_stream_t(this);
  return true;
}
bool gpu_ocl_engine_t::create_memory_storage(memory_storage_t** storage) const {
  *storage = new gpu_ocl_memory_storage_t(this);
  return true;
}

bool gpu_ocl_engine_t::create_kernel(const operator_desc& op_desc, std::shared_ptr<kernel_t>& kernel,
                                     const stream_t*) const {
  auto impl_list_ = get_implementation_list(op_desc);
  if (impl_list_ == nullptr) {
    return false;
  }
  // Step 1: Get the first && success object in impl_list_.
  std::shared_ptr<const kernel_desc_t> result_kd;
  std::shared_ptr<const kernel_desc_t> candidate_kd;
  auto& impl_list = (*impl_list_);
  for (auto& impl : impl_list) {
    auto status = impl(candidate_kd, op_desc);  // kd->create() + kd->init()
    if (status) {
      result_kd = candidate_kd;
      break;
    }
  }
  // step 2 create kernel
  kernel_cache* global_primitive_cache = Singleton<kernel_cache>::GetInstance();
  const auto& callback = std::bind(&kernel_desc_t::create_primitive, result_kd, std::placeholders::_1,
                                   result_kd);  // k_t->create() + k_t->init()
  std::shared_ptr<const kernel_t> value =
      global_primitive_cache->find_or_construct(result_kd->get_operator_desc(), callback);
  if (value == nullptr) {
    return false;
  }
  kernel = std::const_pointer_cast<kernel_t>(value);
  return true;
}
}  // namespace jd
