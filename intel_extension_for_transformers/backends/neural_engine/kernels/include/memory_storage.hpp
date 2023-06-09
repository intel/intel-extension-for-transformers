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

#ifndef ENGINE_SPARSELIB_INCLUDE_MEMORY_STORAGE_HPP_
#define ENGINE_SPARSELIB_INCLUDE_MEMORY_STORAGE_HPP_

#include <cstddef>

namespace jd {
class engine_t;
class stream_t;
enum class copy_direction_t { host_to_host, host_to_device, device_to_host, device_to_device };
class memory_storage_t {
 public:
  explicit memory_storage_t(const engine_t* engine) : engine_(engine) {}
  virtual ~memory_storage_t() = default;
  const engine_t* get_engine() { return engine_; }

 public:
  virtual bool allocate(size_t size) = 0;
  virtual bool get_handle(void** handle) const = 0;
  virtual bool set_handle(void* handle) = 0;
  virtual bool mmap(void** map_ptr, size_t size, const stream_t* stream) = 0;
  virtual bool unmmap(void* map_ptr, size_t size, const stream_t* stream) = 0;
  virtual bool copy(void* ptr, size_t size, copy_direction_t direction, const stream_t* stream) = 0;  // NOLINT
  virtual bool is_null() const = 0;
  virtual size_t get_ptr_size() const = 0;
  size_t get_size() const { return size_; }
  void set_size(size_t size) { size_ = size; }

 protected:
  size_t size_ = 0;

 private:
  const engine_t* engine_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_MEMORY_STORAGE_HPP_
