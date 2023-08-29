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

#ifndef ENGINE_SPARSELIB_SRC_CPU_CPU_MEMORY_STORAGE_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_CPU_MEMORY_STORAGE_HPP_

#include "memory_storage.hpp"

namespace jd {
class cpu_memory_storage_t : public memory_storage_t {
 public:
  explicit cpu_memory_storage_t(const engine_t* engine) : memory_storage_t(engine) {}
  ~cpu_memory_storage_t();

  bool allocate(size_t size) override;

  bool get_handle(void** handle) const override {
    *handle = data_;
    return true;
  }
  bool set_handle(void* handle) override {
    data_ = handle;
    external_ = true;
    return true;
  }
  bool mmap(void** map_ptr, size_t size, const stream_t* stream) override;
  bool unmmap(void* map_ptr, size_t size, const stream_t* stream) override;
  bool copy(void* ptr, size_t size, copy_direction_t direction, const stream_t*) override;  // NOLINT
  bool is_null() const override { return data_ == nullptr; }
  size_t get_ptr_size() const override { return sizeof(void*); }

 private:
  void* data_ = nullptr;
  bool external_ = false;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_CPU_MEMORY_MANAGER_HPP_
