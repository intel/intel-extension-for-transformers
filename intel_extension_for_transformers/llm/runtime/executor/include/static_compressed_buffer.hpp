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

#ifndef ENGINE_EXECUTOR_INCLUDE_STATIC_COMPRESSED_BUFFER_HPP_
#define ENGINE_EXECUTOR_INCLUDE_STATIC_COMPRESSED_BUFFER_HPP_

#include <unordered_map>
#include <vector>
#include <set>
#include <map>
#include <utility>
#include <string>
#include <algorithm>

#include "activation_dag.hpp"

#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_free free
#endif

namespace executor {
using std::map;
using std::pair;
using std::set;
using std::unordered_map;
using std::vector;

struct TensorUsageRecord {
  string name;
  size_t hold_bytes;
  size_t life_begin;
  size_t life_end;
};

class StaticCompressedBuffer {
 public:
  using SharedBlock = pair<size_t, set<string>>;
  explicit StaticCompressedBuffer(const ActivationDAG& dag, bool debug_mode = false) {
    try {
      DagValidCheck(dag);
    } catch (const string& e) {
      LOG(FATAL) << "Invalid DAG." << e;
    }
    if (debug_mode) {
      debug_mode_ = true;
      NoCompressMalloc(dag);
    } else {
      GenTensorUsageRecords(dag);
      GenLowBoundSize(dag);
      GreedyBySize(dag);
    }
  }
  ~StaticCompressedBuffer() {
    if (activation_buffer_ != nullptr) aligned_free(activation_buffer_);
    if (debug_mode_) {
      for (auto&& i : memory_map_) {
        if (i.second != nullptr) aligned_free(i.second);
      }
    }
  }
  void* GetDataByName(const string& name) { return memory_map_[name]; }

 private:
  void* activation_buffer_ = nullptr;
  bool debug_mode_ = false;
  const int align_size = 64;  // TODO(zhe): make it configable based on different arch?
  unordered_map<string, void*> memory_map_;
  map<string, TensorUsageRecord> tensor_usage_records_;
  size_t low_bound_size_ = 0;

  void DagValidCheck(const ActivationDAG& dag) {
    // yaml-serialization-type can assert there will no ring in the DAG. So we needn't to check.
    set<string> tensor_set;
    for (auto&& op : dag.operators()) {
      for (auto&& in_tensor : op->input())
        if (tensor_set.count(in_tensor->name()) == 0)
          throw string("Refer to an inexistent tensor. Please check tensor:" + in_tensor->name());
      for (auto&& out_tensor : op->output()) tensor_set.insert(out_tensor->name());
    }
  }

  void GenTensorUsageRecords(const ActivationDAG& dag) {
    set<string> tensor_names;
    for (const auto& op : dag.operators())
      for (auto&& output_tensor : op->output()) tensor_names.insert(output_tensor->name());
    for (auto&& tensor : tensor_names) {
      for (auto&& op : dag.operators()) {
        for (auto&& out_tensor : op->output()) {
          // in-place tensors can be output tensor even if they were created, so we check whether it was created first.
          if (out_tensor->name() == tensor && tensor_usage_records_.count(tensor) == 0) {
            TensorUsageRecord record;
            record.name = tensor;
            record.hold_bytes = out_tensor->alloc_bytes();
            record.life_begin = op->topological_order();
            tensor_usage_records_[tensor] = record;
          }
        }
        for (auto&& in_tensor : op->input()) {
          if (in_tensor->name() == tensor) {
            auto& record = tensor_usage_records_[tensor];
            record.life_end = op->topological_order();
          }
        }
      }
    }
  }

  void GenLowBoundSize(const ActivationDAG& dag) {
    map<int64_t, vector<size_t>> topological_tensor_width;
    int max_width = 0;
    for (auto&& record : tensor_usage_records_) {
      for (int i = record.second.life_begin; i < record.second.life_end; i++) {
        if (topological_tensor_width.count(i) == 0) {
          topological_tensor_width[i] = {record.second.hold_bytes};
        } else {
          topological_tensor_width[i].push_back(record.second.hold_bytes);
        }
        if (topological_tensor_width[i].size() > max_width) max_width = topological_tensor_width[i].size();
      }
    }
    // sort
    for (auto&& width : topological_tensor_width) {
      std::sort(width.second.begin(), width.second.end(), [](const size_t& a, const size_t& b) { return a > b; });
    }
    // calc low-bound size
    for (int i = 1; i <= max_width; i++) {
      size_t cur_width_max_size = 0;
      for (auto&& width : topological_tensor_width) {
        if (width.second.size() >= i)
          cur_width_max_size = width.second[i - 1] > cur_width_max_size ? width.second[i - 1] : cur_width_max_size;
      }
      low_bound_size_ += cur_width_max_size;
    }
  }

  bool CheckOverlap(const string& tensor, const SharedBlock& shared_block) {
    auto life_begin = tensor_usage_records_[tensor].life_begin;
    auto life_end = tensor_usage_records_[tensor].life_end;
    for (auto&& tensor_name : shared_block.second) {
      auto cur_life_begin = tensor_usage_records_[tensor_name].life_begin;
      auto cur_life_end = tensor_usage_records_[tensor_name].life_end;
      if (cur_life_end >= life_end && cur_life_begin <= life_end) return true;
      if (life_begin >= cur_life_begin && life_begin <= cur_life_end) return true;
      if (life_begin <= cur_life_begin && life_end >= cur_life_end) return true;
    }
    return false;
  }

  void NoCompressMalloc(const ActivationDAG& dag) {
    size_t total_byte = 0;
    for (const auto& op : dag.operators()) {
      for (auto&& output_tensor : op->output()) {
        if (memory_map_.count(output_tensor->name()) == 0) {
          auto allocate_bytes = (output_tensor->alloc_bytes() + align_size - 1) / align_size * align_size;
          total_byte += allocate_bytes;
          memory_map_[output_tensor->name()] = aligned_alloc(align_size, allocate_bytes);
        }
      }
    }
    LOG(INFO) << "static-debug-buffer allocate " << total_byte << " bytes.";
  }

  void GreedyBySize(const ActivationDAG& dag) {
    memory_map_.clear();
    if (activation_buffer_ != nullptr) free(activation_buffer_);
    size_t raw_bytes = 0;
    size_t compressed_bytes = 0;
    vector<SharedBlock> shared_blocks;

    // sort the tensor_usage_record by size
    vector<TensorUsageRecord> sorted_tensor_usage_records;
    for (auto&& i : tensor_usage_records_) {
      raw_bytes += i.second.hold_bytes;
      sorted_tensor_usage_records.push_back(i.second);
    }

    std::sort(sorted_tensor_usage_records.begin(), sorted_tensor_usage_records.end(),
              [](const TensorUsageRecord& a, const TensorUsageRecord& b) { return a.hold_bytes > b.hold_bytes; });
    for (auto&& record : sorted_tensor_usage_records) {
      auto tensor_name = record.name;
      auto allocate_bytes = record.hold_bytes;
      bool find_suitable_block = false;
      for (int i = shared_blocks.size() - 1; i >= 0; i--) {
        if (shared_blocks[i].first >= allocate_bytes) {
          if (!CheckOverlap(tensor_name, shared_blocks[i])) {
            find_suitable_block = true;
            shared_blocks[i].second.insert(tensor_name);
            break;
          }
        }
      }
      if (!find_suitable_block) {
        SharedBlock new_block;
        new_block.first = (allocate_bytes + align_size - 1) / align_size * align_size;
        new_block.second.insert(tensor_name);
        shared_blocks.push_back(new_block);
        compressed_bytes += new_block.first;
      }
    }

    activation_buffer_ = aligned_alloc(align_size, compressed_bytes);
    size_t offset = 0;
    for (auto&& i : shared_blocks) {
      for (auto&& j : i.second) {
        memory_map_[j] = static_cast<char*>(activation_buffer_) + offset;
      }
      offset += i.first;
    }
    // process in-place tensors.
    for (auto&& i : dag.inplace_alias_holder()) {
      for (auto&& j : i.second) {
        memory_map_[j] = memory_map_[i.first];
      }
    }
    LOG(INFO) << "activation buffer raw bytes: " << raw_bytes << " compressed bytes: " << compressed_bytes
              << " compress ratio: " << (static_cast<float>(raw_bytes) - compressed_bytes) / raw_bytes;
    LOG(INFO) << "low-bound compressed bytes: " << low_bound_size_
              << " compressed efficiency: " << (static_cast<float>(low_bound_size_)) / compressed_bytes * 100 << "%";
  }
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_STATIC_COMPRESSED_BUFFER_HPP_
