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

#ifndef ENGINE_SPARSELIB_SRC_GPU_OCL_MEMORY_STORAGE_HPP_
#define ENGINE_SPARSELIB_SRC_GPU_OCL_MEMORY_STORAGE_HPP_

#include <CL/cl.h>
#include "memory_storage.hpp"

namespace jd {
class gpu_ocl_memory_storage_t : public memory_storage_t {
 public:
  explicit gpu_ocl_memory_storage_t(const engine_t* engine) : memory_storage_t(engine) {}
  ~gpu_ocl_memory_storage_t() {}

  bool allocate(size_t size) override;

  bool get_handle(void** handle) const override {
    *handle = static_cast<void*>(data_);
    return true;
  }
  bool set_handle(void* handle) override {
    data_ = static_cast<cl_mem>(handle);
    return true;
  }
  bool mmap(void** map_ptr, size_t size, const stream_t* stream) override;
  bool unmmap(void* map_ptr, size_t, const stream_t* stream) override;
  bool copy(void* ptr, size_t size, copy_direction_t direction, const stream_t* stream) override;  // NOLINT
  bool is_null() const override { return data_ == nullptr; }
  size_t get_ptr_size() const override { return sizeof(cl_mem); }

 private:
  cl_mem data_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_GPU_OCL_MEMORY_MANAGER_HPP_
