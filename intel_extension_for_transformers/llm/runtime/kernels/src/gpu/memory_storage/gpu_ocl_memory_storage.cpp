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

#include "gpu_ocl_memory_storage.hpp"
#include "src/gpu/engine/gpu_ocl_engine.hpp"
#include "src/gpu/stream/gpu_ocl_stream.hpp"

namespace jd {
bool gpu_ocl_memory_storage_t::allocate(size_t size) {
  cl_int err;
  const gpu_ocl_engine_t* engine = dynamic_cast<const gpu_ocl_engine_t*>(memory_storage_t::get_engine());
  data_ = clCreateBuffer(engine->get_context(), CL_MEM_READ_WRITE, size, nullptr, &err);
  size_ = size;
  return err == CL_SUCCESS;
}
bool gpu_ocl_memory_storage_t::mmap(void** map_ptr, size_t size, const stream_t* stream) {
  if (data_ == nullptr) {
    *map_ptr = nullptr;
    return false;
  }
  cl_mem_flags mem_flags;
  if ((clGetMemObjectInfo(data_, CL_MEM_FLAGS, sizeof(mem_flags), &mem_flags, nullptr)) != CL_SUCCESS) {
    return false;
  }

  size_t mem_bytes;
  if ((clGetMemObjectInfo(data_, CL_MEM_SIZE, sizeof(mem_bytes), &mem_bytes, nullptr)) != CL_SUCCESS ||
      size != mem_bytes) {
    return false;
  }
  cl_map_flags map_flags = 0;
  if (mem_flags & CL_MEM_READ_WRITE) {
    map_flags |= CL_MAP_READ;
    map_flags |= CL_MAP_WRITE;
  } else if (mem_flags & CL_MEM_READ_ONLY) {
    map_flags |= CL_MAP_READ;
  } else if (mem_flags & CL_MEM_WRITE_ONLY) {
    map_flags |= CL_MAP_WRITE;
  }

  cl_command_queue queue = dynamic_cast<const gpu_ocl_stream_t*>(stream)->get_queue();
  cl_int err;
  *map_ptr = clEnqueueMapBuffer(queue, data_, CL_TRUE, map_flags, 0, mem_bytes, 0, nullptr, nullptr, &err);
  return err == CL_SUCCESS;
}

bool gpu_ocl_memory_storage_t::unmmap(void* map_ptr, size_t, const stream_t* stream) {
  if (map_ptr == nullptr) {
    return false;
  }
  cl_command_queue queue = dynamic_cast<const gpu_ocl_stream_t*>(stream)->get_queue();
  clEnqueueUnmapMemObject(queue, data_, map_ptr, 0, nullptr, nullptr);
  clFinish(queue);
  return true;
}

bool gpu_ocl_memory_storage_t::copy(void* ptr, size_t size, copy_direction_t direction,  // NOLINT
                                    const stream_t* stream) {
  cl_command_queue queue = dynamic_cast<const gpu_ocl_stream_t*>(stream)->get_queue();
  cl_int err;
  switch (direction) {
    case copy_direction_t::device_to_host:
      err = clEnqueueReadBuffer(queue, data_, CL_TRUE, 0, size, ptr, 0, nullptr, nullptr);
      break;
    case copy_direction_t::host_to_device:
      if (size_ == 0) {
        allocate(size);
      }
      err = clEnqueueWriteBuffer(queue, data_, CL_TRUE, 0, size, ptr, 0, nullptr, nullptr);
      break;
    case copy_direction_t::device_to_device:
      err = clEnqueueCopyBuffer(queue, data_, static_cast<cl_mem>(ptr), 0, 0, size, 0, nullptr, nullptr);
      break;

    default:
      return false;
      // break;
  }
  return err == CL_SUCCESS;
}
}  // namespace jd
