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

#ifndef ENGINE_EXECUTOR_INCLUDE_MEMORY_ALLOCATOR_HPP_
#define ENGINE_EXECUTOR_INCLUDE_MEMORY_ALLOCATOR_HPP_

#include <stdlib.h>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <thread>  // NOLINT
#include <vector>
#include <utility>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

#include "i_malloc.hpp"
#include "execution_options.hpp"
#include "static_compressed_buffer.hpp"

namespace executor {
using std::map;
using std::shared_ptr;
using std::string;
using std::vector;
namespace ipc = boost::interprocess;
class MemoryAllocator {
 public:
  // vector will contain the used counts and size of a memory
  typedef std::map<void*, vector<size_t>> MemoryBuffer;
  typedef std::map<void*, string> BufferName;
  typedef std::map<string, bool> StrategyList;
  typedef std::map<std::thread::id, std::unique_ptr<MemoryBuffer>> TreadMemory;
  typedef std::map<std::thread::id, BufferName*> TreadName;

  static char* SharedEnv(char* env_name = "WEIGHT_SHARING") {
    static char* shared_env = getenv(env_name);
    return shared_env;
  }

  static const int InstNum() {
    auto inst_num = getenv("INST_NUM");
    if (inst_num != nullptr) return std::atoi(inst_num);
    return 1;
  }

  static MemoryBuffer& Buffer() {
    static TreadMemory t_memory;
    // (TODO) it's not good for each thread to obtain a MemoryBuffer
    std::thread::id id = std::this_thread::get_id();
    auto count = t_memory.count(id);
    if (count == 0) {
      std::unique_ptr<MemoryBuffer> mu_ptr(new MemoryBuffer());
      t_memory[id] = std::move(mu_ptr);
    }
    return *(t_memory[id]);
  }

  static void InitCompressedBufferManager(const ActivationDAG& dag, const bool& debug_mode = false);

  // use another memory buffer which not be freed by memory_allocator
  static MemoryBuffer& CompressedBuffer() {
    static TreadMemory scpb_memory;
    // (TODO) it's not good for each thread to obtain a MemoryBuffer
    std::thread::id id = std::this_thread::get_id();
    auto count = scpb_memory.count(id);
    if (count == 0) {
      std::unique_ptr<MemoryBuffer> mu_ptr(new MemoryBuffer());
      scpb_memory[id] = std::move(mu_ptr);
    }
    return *(scpb_memory[id]);
  }

  static BufferName& Name() {
    static TreadName t_name;
    // (TODO) it's not good for each thread to obtain a MemoryBuffer
    std::thread::id id = std::this_thread::get_id();
    if (t_name.count(id) == 0) {
      t_name[id] = new BufferName();
    }
    return *(t_name[id]);
  }

  static void SetName(void* data, const string name) {
    BufferName& name_buffer = Name();
    MemoryBuffer& memory_buffer = Buffer();
    auto iter = memory_buffer.find(data);
    if (iter != memory_buffer.end()) {
      name_buffer[iter->first] = name;
    } else {
      DLOG(WARNING) << "name a not existing pointer...";
    }
  }

  static int AliveBuffer() {
    MemoryBuffer& memory_buffer = Buffer();
    BufferName& name_buffer = Name();
    int alive = 0;
    for (auto iter = memory_buffer.begin(); iter != memory_buffer.end(); ++iter) {
      auto buffer_count = iter->second[0];
      if (buffer_count != 0) {
        alive++;
        DLOG(WARNING) << "have alive buffer name " << name_buffer[iter->first];
      }
    }
    DLOG(WARNING) << "buffer alive count " << alive;
    return alive;
  }

  static void ReleaseBuffer() {
    MemoryBuffer& memory_buffer = Buffer();
    for (auto iter = memory_buffer.begin(); iter != memory_buffer.end(); ++iter) {
      auto buffer_count = iter->second[0];
      if (buffer_count != 0) {
        DLOG(WARNING) << "buffer still have life, force release...";
        iter->second[0] = 0;
      }
    }
  }

  static StrategyList& Strategy() {
    static StrategyList* m_strategy_ = new StrategyList({{"cycle_buffer", false},
                                                         {"direct_buffer", false},
                                                         {"unified_buffer", false},
                                                         {"static_compressed_buffer", false}});
    return *m_strategy_;
  }

  static ipc::managed_shared_memory& ManagedShm(char* space_name = "SharedWeight") {
    static ipc::managed_shared_memory shm_ptr(ipc::open_only, space_name);
    return shm_ptr;
  }

  static void InitStrategy(const ExecutionOptions& execution_options = ExecutionOptions()) {
    string memory_strategy;
    if (execution_options.activation_mem_compression) {
      memory_strategy = "static_compressed_buffer";
    } else {
      memory_strategy = getenv("UNIFIED_BUFFER") != NULL
                            ? "unified_buffer"
                            : (getenv("DIRECT_BUFFER") == NULL ? "cycle_buffer" : "direct_buffer");
    }
    SetStrategy(memory_strategy);
  }

  static void SetStrategy(const string strategy) {
    CHECK(strategy == "cycle_buffer" || strategy == "direct_buffer" || strategy == "unified_buffer" ||
          strategy == "static_compressed_buffer")
        << "only support memory strategy cycle buffer, direct buffer, unified buffer and static compressed buffer";
    StrategyList& strategy_list = Strategy();
    strategy_list[strategy] = true;
    DLOG(INFO) << "strategy list set success " << strategy;
  }

  static int CheckMemory(void* data) {
    MemoryBuffer& memory_buffer = Buffer();
    MemoryBuffer& scpb_mem_buffer = CompressedBuffer();
    if (memory_buffer.count(data) != 0) {
      auto iter = memory_buffer.find(data);
      return iter->second[0];
    } else if (scpb_mem_buffer.count(data) != 0) {
      auto iter = scpb_mem_buffer.find(data);
      return iter->second[0];
    } else {
      DLOG(WARNING) << "get life from a not existing memory pointer...";
      return -1;
    }
  }

  // set the data buffer a new life count
  static void ResetMemory(void* data, const int life_count) {
    MemoryBuffer& memory_buffer = Buffer();
    MemoryBuffer& scpb_mem_buffer = CompressedBuffer();
    StrategyList& strategy_list = Strategy();
    if (memory_buffer.count(data) != 0) {
      auto iter = memory_buffer.find(data);
      iter->second[0] = life_count;
    } else if (scpb_mem_buffer.count(data) != 0) {
      auto iter = scpb_mem_buffer.find(data);
      iter->second[0] = life_count;
    } else {
      DLOG(WARNING) << "reset a not existing memory pointer...";
    }
  }

  // will return the left count of one tensor
  static int UnrefMemory(void* data, bool inplace = false) {
    MemoryBuffer& memory_buffer = Buffer();
    MemoryBuffer& scpb_mem_buffer = CompressedBuffer();
    StrategyList& strategy_list = Strategy();
    std::map<void*, vector<size_t>>::iterator iter;
    bool in_scpb_mem_buffer = false;
    bool in_mem_buffer = false;
    if (scpb_mem_buffer.count(data) != 0) {
      iter = scpb_mem_buffer.find(data);
      in_scpb_mem_buffer = true;
    }
    if (memory_buffer.count(data) != 0) {
      iter = memory_buffer.find(data);
      in_mem_buffer = true;
    }
    LOG_IF(FATAL, (in_scpb_mem_buffer && in_mem_buffer))
        << "Find data ptr " << data << "in static compressed buffer and cycle buffer.";
    int status = 0;
    if (in_scpb_mem_buffer || in_mem_buffer) {
      if (iter->second[0] <= 0) {
        DLOG(WARNING) << "free a no-used memory...";
        iter->second[0] = 0;
        status = 0;
      } else {
        (iter->second[0])--;
        status = iter->second[0];
      }
      if (in_mem_buffer && status == 0 && inplace == false) {
        if (strategy_list["direct_buffer"]) {
          auto free_ptr = iter->first;
          aligned_free(free_ptr);
          memory_buffer.erase(free_ptr);
        } else if (strategy_list["unified_buffer"]) {
          auto free_ptr = iter->first;
          i_free(free_ptr);
          memory_buffer.erase(free_ptr);
        }
      }
    } else {
      DLOG(WARNING) << "free not existing memory pointer...";
      status = -1;
    }
    return status;
  }

  static void* GetMemory(size_t size, const int life_count, const string& tensor_name = "") {
    static std::mutex getmem_lock;
    std::lock_guard<std::mutex> lock(getmem_lock);
    if (size == 0) {
      DLOG(INFO) << "please set the tensor size...";
      return nullptr;
    }
    if (life_count <= 0) {
      DLOG(INFO) << "please set the tensor life...";
      return nullptr;
    }
    StrategyList& strategy_list = Strategy();
    if (strategy_list["static_compressed_buffer"]) {
      if (tensor_name != "") {
        // activation memory allocation
        return StaticCompressedBufferGetMemory(size, life_count, tensor_name);
      } else {
        // workspace and other memory allocation
        // (TODO) it's not good to use one more memory management tools in one instance
        return CycleBufferGetMemory(size, life_count);
      }
    } else if (strategy_list["direct_buffer"]) {
      return DirectBufferGetMemory(size, life_count);
    } else if (strategy_list["cycle_buffer"]) {
      return CycleBufferGetMemory(size, life_count);
    } else if (strategy_list["unified_buffer"]) {
      return UnifiedBufferGetMemory(size, life_count);
    } else {
      LOG(ERROR) << "please set the memory strategy";
      return nullptr;
    }
  }

  // Get memory from static compressed buffer manager
  static void* StaticCompressedBufferGetMemory(size_t size, const int life_count, const string& tensor_name = "");

  static void* CycleBufferGetMemory(size_t size, const int life_count) {
    MemoryBuffer& memory_buffer = Buffer();
    DLOG(INFO) << "cycle buffer tensor size is " << memory_buffer.size();
    for (auto iter = memory_buffer.begin(); iter != memory_buffer.end(); ++iter) {
      auto buffer_count = iter->second[0];
      auto buffer_size = iter->second[1];
      if (buffer_count == 0) {
        if (size > buffer_size) {
          auto free_ptr = iter->first;
          aligned_free(free_ptr);
          memory_buffer.erase(free_ptr);
          // allocate new buffer
          void* buf = reinterpret_cast<void*>(aligned_alloc(ALIGNMENT, (size / ALIGNMENT + 1) * ALIGNMENT));
          memory_buffer.insert({buf, vector<size_t>({static_cast<size_t>(life_count), size})});
          return buf;
        } else {
          memory_buffer[iter->first] = vector<size_t>({static_cast<size_t>(life_count), buffer_size});
          return iter->first;
        }
      }
    }
    // allocate new buffer
    void* buf = reinterpret_cast<void*>(aligned_alloc(ALIGNMENT, (size / ALIGNMENT + 1) * ALIGNMENT));
    memory_buffer.insert({buf, vector<size_t>({static_cast<size_t>(life_count), size})});
    return buf;
  }

  static void* DirectBufferGetMemory(size_t size, const int life_count) {
    MemoryBuffer& memory_buffer = Buffer();
    DLOG(INFO) << "direct buffer tensor size is " << memory_buffer.size();
    void* buf = reinterpret_cast<void*>(aligned_alloc(ALIGNMENT, (size / ALIGNMENT + 1) * ALIGNMENT));
    memory_buffer.insert({buf, vector<size_t>({static_cast<size_t>(life_count), size})});
    return buf;
  }

  static void* UnifiedBufferGetMemory(size_t size, const int life_count) {
    MemoryBuffer& memory_buffer = Buffer();
    DLOG(INFO) << "unified buffer tensor size is " << memory_buffer.size();
    void* buf = reinterpret_cast<void*>(i_malloc(size));
    memory_buffer.insert({buf, vector<size_t>({static_cast<size_t>(life_count), size})});
    return buf;
  }

  // MemoryAllocator(MemoryAllocator const&)  = delete;
  // void operator=(MemoryAllocator const&)  = delete;

  static MemoryAllocator& get() {
    static MemoryAllocator instance;
    return instance;
  }

 private:
  // Private constructor to prevent instancing.
  MemoryAllocator() {}
  // static compressed buffer manager
  // init by activation dag
  static std::unique_ptr<StaticCompressedBuffer> scpb_manager_;
};

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_MEMORY_ALLOCATOR_HPP_
