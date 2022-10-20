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

#ifndef ENGINE_EXECUTOR_INCLUDE_MODEL_HPP_
#define ENGINE_EXECUTOR_INCLUDE_MODEL_HPP_

#include <stdio.h>

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include "common.hpp"
#include "glog/logging.h"
#include "memory_allocator.hpp"
#include "operator.hpp"
#include "dispatcher.hpp"
#include "operator_registry.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"
#include "profiling.hpp"
#include "llga_kernel.hpp"
#include "llga_op_creator.hpp"
namespace executor {

using logical_tensor = dnnl::graph::logical_tensor;
using property_type = dnnl::graph::logical_tensor::property_type;
using llga_op = dnnl::graph::op;
using data_type = dnnl::graph::logical_tensor::data_type;
using layout_type = dnnl::graph::logical_tensor::layout_type;
using compiled_partition = dnnl::graph::compiled_partition;

/**
 * @brief Connects Operator%s together into a directed acyclic graph (DAG)
 *        specified by a ModelConfig.
 *
 */
class Model {
  friend class LLGAKernel;
  friend class LLGAOPCreator;

 public:
  explicit Model(const ModelConfig& conf, const string& weight_root);
  explicit Model(const string& conf_file, const string& weight_root);
  virtual ~Model();

  void Init(const ModelConfig& conf);
  void RemoveSharedWeight(bool is_begin = false, char* count_space_name = "RemovedCount",
                          char* count_name = "removed_count", char* space_name = "SharedWeight");
  void InitSharedWeight(char* space_name = "SharedWeight");
  ipc::managed_shared_memory::handle_t LoadSharedWeight(const string& root, const string& type,
                                                        const vector<int64_t>& shape, const vector<int64_t>& location);
  vector<Tensor>& Forward(vector<Tensor>& input_data);  // NOLINT

  void SetInput(const OperatorConfig& conf, const int operator_id, const int tensor_id,
                map<string, int>* tensor_name_to_idx);

  void SetOutput(const OperatorConfig& conf, const int operator_id, const int tensor_id,
                 map<string, int>* tensor_name_to_idx);

  void SetDispatchKernel(const bool& reshape_model);

  inline const string& name() const { return name_; }
  inline const vector<string>& operator_names() const { return operator_names_; }
  inline const vector<string>& tensor_names() const { return tensor_names_; }
  inline const vector<shared_ptr<Dispatcher> >& operators() const { return operators_; }
  inline const vector<Tensor*>& tensors() const { return tensors_; }

  inline int num_inputs() const { return model_input_tensors_.size(); }
  inline int num_outputs() const { return model_output_tensors_.size(); }

  inline const vector<TensorConfig*>& input_configs() const { return model_input_configs_; }

  inline vector<Tensor>& output_tensors() {
    LOG(INFO) << "Output tensor size is " << model_output_tensors_.size();
    for (int i = 0; i < model_output_tensors_.size(); ++i) {
      output_tensors_[i].set_dtype(model_output_tensors_[i]->dtype());
      auto data_buffer = model_output_tensors_[i]->data();
      auto size = model_output_tensors_[i]->size();
      // copy the data from memory to an output buffer
      if (size > output_tensors_[i].size() ||
          output_tensors_[i].size() < size * type2bytes[output_tensors_[i].dtype()]) {
        free(output_tensors_[i].mutable_data());
        void* out_buffer = malloc(size * type2bytes[output_tensors_[i].dtype()]);
        output_tensors_[i].set_data(out_buffer);
        output_tensors_[i].set_shape(model_output_tensors_[i]->shape());
        memcpy(out_buffer, data_buffer, size * type2bytes[output_tensors_[i].dtype()]);
      } else {
        void* out_buffer = output_tensors_[i].mutable_data();
        output_tensors_[i].set_shape(model_output_tensors_[i]->shape());
        memcpy(out_buffer, data_buffer, size * type2bytes[output_tensors_[i].dtype()]);
      }
    }

    for (auto& tensor_ptr : model_output_tensors_) tensor_ptr->unref_data();
    // MemoryAllocator::get().AliveBuffer();
    return output_tensors_;
  }

  inline void AddLogicalTensor(string tensor_name, logical_tensor dst_desc) {
    name2lts_[tensor_name] = dst_desc;
    id2names_[desc_idx] = tensor_name;
    desc_idx++;
  }

  inline void AddLogicalTensor(logical_tensor dst_desc) {
    // 1 operator maps to multi partitions.
    auto tensor_name = "hardcode_" + std::to_string(dst_desc.get_id());  // TODO(lzw): hardcode
    name2lts_[tensor_name] = dst_desc;
    id2names_[desc_idx] = tensor_name;
    desc_idx++;
  }

  inline void AddLogicalTensor(string tensor_name, logical_tensor dst_desc, int id) {
    name2lts_[tensor_name] = dst_desc;
    id2names_[id] = tensor_name;
  }

  inline logical_tensor GetLogicalTensor(int id) { return name2lts_[id2names_[id]]; }
  inline logical_tensor GetLogicalTensor(string tensor_name) { return name2lts_[tensor_name]; }

  inline void AddLLGAOP(llga_op op, int op_conf_index) {
    g_.add_op(op);
    opid2index_[op_idx++] = op_conf_index;
  }

  void ProcessInput(OperatorConfig* op_conf);
  llga_op CreateOperator(OperatorConfig* op_conf, int index);
  void ConstructLLGA(const vector<OperatorConfig*>& op_configs);


 protected:
  string name_;
  string weight_root_;
  vector<shared_ptr<Dispatcher> > operators_;
  vector<string> operator_names_;
  map<string, int> operator_name_index_;
  vector<Tensor*> tensors_;
  vector<string> tensor_names_;
  map<string, int> tensor_name_index_;

  /// input output weight vecs stores the vectors of each operator.
  vector<vector<Tensor*> > input_vecs_;
  vector<vector<Tensor*> > output_vecs_;

  vector<Tensor*> model_input_tensors_;
  vector<TensorConfig*> model_input_configs_;
  vector<Tensor*> model_output_tensors_;
  vector<Tensor> output_tensors_;
  bool multi_stream_flag = (getenv("MULTI_STREAM") != NULL);
  // collect the op index with parallel thread
  unordered_map<int, int64_t> multi_stream_tasks_;
  ThreadPool tp;
  std::mutex rmutex_;
  // for dispatcher
  bool has_dispatch_table_file_ = false;
  string dispatch_table_file_root_;
  bool is_dispatcher_tuning_ = false;
  // for profiling
  bool engine_profiling_ = false;

  // for llga
  dnnl::graph::graph g_;
  dnnl::graph::engine eng_ {dnnl::graph::engine::kind::cpu, 0};
  dnnl::graph::stream strm_ {eng_};
  vector<dnnl::graph::partition> partitions_;

  int desc_idx = 0;
  int op_idx = 0;

  unordered_map<string, logical_tensor> name2lts_;
  unordered_map<int, string> id2names_;
  unordered_map<int, int> opid2index_;  // <operator_id, index of operator configs>
};

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_MODEL_HPP_
