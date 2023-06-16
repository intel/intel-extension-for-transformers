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

#include "memory_allocator.hpp"

namespace executor {

std::unique_ptr<StaticCompressedBuffer> MemoryAllocator::scpb_manager_;

void MemoryAllocator::InitCompressedBufferManager(const ActivationDAG& dag, const bool& debug_mode) {
  std::unique_ptr<StaticCompressedBuffer> scpb_ptr(new StaticCompressedBuffer(dag, debug_mode));
  scpb_manager_ = std::move(scpb_ptr);
}

void* MemoryAllocator::StaticCompressedBufferGetMemory(size_t size, const int life_count,
                                                       const string& tensor_name) {
  LOG_IF(FATAL, tensor_name == "") << "Please supply tensor name for StaticCompressedBuffer.";
  void* buf;
  try {
    buf = scpb_manager_->GetDataByName(tensor_name);
  } catch (...) {
    LOG(FATAL) << "tensor " << tensor_name << " is not in activation dag.";
  }
  MemoryBuffer& scpb_mem_buffer = CompressedBuffer();
  DLOG(INFO) << "static compressed buffer tensor size is " << scpb_mem_buffer.size();
  if (scpb_mem_buffer.count(buf) == 0) {
    scpb_mem_buffer.insert({buf, vector<size_t>({static_cast<size_t>(life_count), size})});
  } else {
    scpb_mem_buffer[buf] = vector<size_t>({static_cast<size_t>(life_count), size});
  }
  return buf;
}

}  // namespace executor
