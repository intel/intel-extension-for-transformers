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
#include "cpu_memory_storage.hpp"
#include "src/cpu/engine/cpu_engine.hpp"

namespace jd {
cpu_memory_storage_t::~cpu_memory_storage_t() {
  if (!external_) {
    free(data_);
  }
}
bool cpu_memory_storage_t::allocate(size_t size) {
  data_ = malloc(size);
  return true;
}
bool cpu_memory_storage_t::mmap(void** map_ptr, size_t size, const stream_t*) {
  *map_ptr = data_;
  size_ = size;
  return true;
}

bool cpu_memory_storage_t::unmmap(void*, size_t, const stream_t*) { return true; }

bool cpu_memory_storage_t::copy(void* ptr, size_t size, copy_direction_t direction, const stream_t*) {  // NOLINT
  switch (direction) {
    case copy_direction_t::host_to_host:
      if (size_ == 0) {
        allocate(size);
      }
      memcpy(data_, ptr, size);
      break;

    default:
      return false;
      // break;
  }
  return true;
}
}  // namespace jd
