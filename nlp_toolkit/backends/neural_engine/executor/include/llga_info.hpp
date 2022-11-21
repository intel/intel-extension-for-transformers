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

#ifndef ENGINE_EXECUTOR_INCLUDE_LLGA_INFO_HPP_
#define ENGINE_EXECUTOR_INCLUDE_LLGA_INFO_HPP_

#include <string>
#include <unordered_map>
#include <map>
#include <vector>
#include <memory>

#include "tensor.hpp"
#include "conf.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

namespace executor {

using logical_tensor = dnnl::graph::logical_tensor;
using property_type = dnnl::graph::logical_tensor::property_type;
using llga_op = dnnl::graph::op;
using data_type = dnnl::graph::logical_tensor::data_type;
using layout_type = dnnl::graph::logical_tensor::layout_type;
using compiled_partition = dnnl::graph::compiled_partition;

using std::string;
using std::vector;
using std::unordered_map;
using std::map;
using std::to_string;

class LLGAINFO {
 public:
  LLGAINFO() : g_(dnnl::graph::engine::kind::cpu),
                strm_ {eng_},
                eng_ {dnnl::graph::engine::kind::cpu, 0} {}

  inline void AddLogicalTensor(const string& tensor_name, const logical_tensor& dst_desc) {
    name2lts_.insert({tensor_name, dst_desc});
    id2names_.insert({logical_tensor_idx, tensor_name});
    logical_tensor_idx++;
  }

  inline void AddLogicalTensor(const logical_tensor& dst_desc) {
    // 1 operator maps to multi partitions.
    auto tensor_name = "hardcode_" + to_string(dst_desc.get_id());  // TODO(lzw): hardcode
    name2lts_.insert({tensor_name, dst_desc});
    id2names_.insert({logical_tensor_idx, tensor_name});
    logical_tensor_idx++;
  }

  inline void AddLogicalTensor(const string& tensor_name, const logical_tensor& dst_desc, int id) {
    // new id points to a new tensor_name.
    name2lts_[tensor_name] = dst_desc;
    id2names_[id] = tensor_name;
  }

  inline void AddLogicalTensor(const logical_tensor& dst_desc, int id) {
    // replace old logical tensor with same id
    name2lts_[id2names_[id]] = dst_desc;
  }

  inline logical_tensor GetLogicalTensor(int id) { return name2lts_[id2names_[id]]; }
  inline logical_tensor GetLogicalTensor(const string& tensor_name) { return name2lts_[tensor_name]; }

  inline void AddLLGAOP(const llga_op& op, int op_conf_index) {
    g_.add_op(op);
    opid2index_[llga_op_idx++] = op_conf_index;
  }

  inline vector<dnnl::graph::partition> GetPartitions() { return g_.get_partitions(); }
  inline dnnl::graph::engine& GetEngine() { return eng_; }
  inline dnnl::graph::stream& GetStream() { return strm_; }
  inline size_t GetOPIndex() { return static_cast<size_t>(llga_op_idx); }
  inline size_t GetLTIndex() { return static_cast<size_t>(logical_tensor_idx); }
  inline const string& GetTensorName(int id) { return id2names_[id]; }
  inline int GetIndexFromOPID(int id) { return opid2index_[id]; }

  void InitLTFromTensorConf(const shared_ptr<OperatorConfig>& op_conf, bool from_output_config = true) {
    // Add logical tensor from operator config, for model initialization and test initialization.
    int tensor_count = 0;
    if (from_output_config) {
      tensor_count = op_conf->output_tensor_size();
    } else {
      tensor_count = op_conf->input_tensor_size();
    }
    for (int idx = 0; idx < tensor_count; ++idx) {
      shared_ptr<TensorConfig> tensor_config = nullptr;
      if (from_output_config) {
        tensor_config = op_conf->output_tensors(idx);
      } else {
        tensor_config = op_conf->input_tensors(idx);
      }
      const string& tensor_name = tensor_config->name();
      data_type dtype = ConvertDataType(tensor_config->dtype());
      if (tensor_config->location().size() != 0) {
        logical_tensor dst_desc {GetLTIndex(), dtype, tensor_config->shape(),
                                 layout_type::strided, property_type::constant};
        AddLogicalTensor(tensor_name, dst_desc);
      } else if (from_output_config) {
        logical_tensor dst_desc {GetLTIndex(), dtype, layout_type::strided};
        AddLogicalTensor(tensor_name, dst_desc);
      } else {
        logical_tensor dst_desc {GetLTIndex(), dtype, tensor_config->shape(), layout_type::strided};
        AddLogicalTensor(tensor_name, dst_desc);
      }
    }
  }

  void PrepareLTForOperator(const shared_ptr<OperatorConfig>& op_conf,
       vector<logical_tensor>* inputs, vector<logical_tensor>* outputs) {
    // prepare logical tensors for creating llga operator
    int input_size = op_conf->input_tensor_size();
    int output_size = op_conf->output_tensor_size();
    auto attrs_map = op_conf->attributes();

    for (int input_id = 0; input_id < input_size; ++input_id) {
      const string& tensor_name = op_conf->input_tensors(input_id)->name();
      inputs->push_back(GetLogicalTensor(tensor_name));
    }

    string output_dtype_ = "";
    auto iter = attrs_map.find("output_dtype");
    if (iter != attrs_map.end()) {
      output_dtype_ = attrs_map["output_dtype"];
    }
    for (int output_id = 0; output_id < output_size; ++output_id) {
      // create logical tensor into name2lts
      data_type dtype;
      if (output_dtype_ != "") {
        dtype = ConvertDataType(output_dtype_);
      } else {
        // delivery dtype if output dtype not specifced.
        if (op_conf->type() == "Gather" && inputs->size() >= 2) {
          dtype = inputs->at(1).get_data_type();
        } else if (op_conf->type() == "DequantizeLinear") {
          dtype = data_type::f32;
        } else if ((op_conf->type() == "Matmul" || op_conf->type() == "InnerProduct") &&
                  (inputs->at(0).get_data_type() == data_type::u8 || inputs->at(0).get_data_type() == data_type::s8)) {
          dtype = data_type::f32;
        } else {
          dtype = inputs->at(0).get_data_type();
        }
      }
      logical_tensor dst_desc {GetLTIndex(), dtype, layout_type::any};
      const string& tensor_name = op_conf->output_tensors(output_id)->name();
      AddLogicalTensor(tensor_name, dst_desc);
      outputs->push_back(dst_desc);
    }
  }

  // temporary solution for min/max data.
  inline void SetTensors(vector<Tensor*>* ptr) { ptr_tensors_ = ptr; }
  inline vector<Tensor*>* GetTensors() { return ptr_tensors_; }
  inline void SetTensorNameIndex(map<string, int>* ptr) { ptr_tensor_name_index_ = ptr; }
  inline map<string, int>* GetTensorNameIndex() { return ptr_tensor_name_index_; }

  inline Tensor* GetTensorByID(int id) {
    auto tensor_name = id2names_[id];
    auto index = (*ptr_tensor_name_index_)[tensor_name];
    return (*ptr_tensors_)[index];
  }

  inline void SetTensorByID(int id, Tensor* ptr) {
    auto tensor_name = id2names_[id];
    auto index = (*ptr_tensor_name_index_)[tensor_name];
    (*ptr_tensors_)[index] = ptr;
  }

  data_type ConvertDataType(const string& type) {
    static unordered_map<string, data_type> type_map {
      {"fp32", data_type::f32}, {"s32", data_type::s32},
      {"fp16", data_type::f16}, {"u8", data_type::u8},
      {"s8", data_type::s8},    {"bf16", data_type::bf16},
      {"int32", data_type::s32}};
    if (type_map.count(type) == 0) {
      LOG(ERROR) << "Can't suppport dtype: " << type << " now!";
      return data_type::undef;
    }
    return type_map[type];
  }

  string ConvertType(data_type datatype) {
    static unordered_map<data_type, string> type_map {
      {data_type::f32, "fp32"}, {data_type::s32, "s32"},
      {data_type::f16, "fp16"}, {data_type::u8, "u8"},
      {data_type::s8, "s8"},    {data_type::bf16, "bf16"}};
    if (type_map.count(datatype) == 0) {
      LOG(ERROR) << "Can't suppport data type now!";
      return "";
    }
    return type_map[datatype];
  }

 private:
  dnnl::graph::graph g_;
  dnnl::graph::engine eng_;
  dnnl::graph::stream strm_;

  int logical_tensor_idx = 0;
  int llga_op_idx = 0;

  unordered_map<string, logical_tensor> name2lts_;
  unordered_map<int, string> id2names_;
  unordered_map<int, int> opid2index_;  // <operator_id, index of operator configs>

  // temporary solution for min/max data.
  vector<Tensor*>* ptr_tensors_ = nullptr;
  map<string, int>* ptr_tensor_name_index_ = nullptr;
};

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_LLGA_INFO_HPP_
