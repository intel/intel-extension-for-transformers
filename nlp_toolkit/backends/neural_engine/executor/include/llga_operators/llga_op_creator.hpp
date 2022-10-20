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

#ifndef ENGINE_EXECUTOR_INCLUDE_LLGA_OPERATORS_LLGA_OP_CREATOR_HPP_
#define ENGINE_EXECUTOR_INCLUDE_LLGA_OPERATORS_LLGA_OP_CREATOR_HPP_
#include <string>
#include <unordered_map>
#include <vector>

#include "model.hpp"
#include "conf.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

namespace executor {

using logical_tensor = dnnl::graph::logical_tensor;
using property_type = dnnl::graph::logical_tensor::property_type;
using llga_op = dnnl::graph::op;
using data_type = dnnl::graph::logical_tensor::data_type;
using layout_type = dnnl::graph::logical_tensor::layout_type;
using compiled_partition = dnnl::graph::compiled_partition;
using std::to_string;

class Model;  // forward declaration

class LLGAOPCreator {
 public:
  typedef bool (LLGAOPCreator::*Creator)(Model *model, OperatorConfig* op_conf, int index);
  typedef std::unordered_map<string, Creator> CreatorRegistry;

  static LLGAOPCreator& GetInstance() {
    static LLGAOPCreator ins;
    return ins;
  }

  void CreateOP(Model *model, OperatorConfig* op_conf, int index, bool is_wildcard) {
    auto operator_name = op_conf->name();
    auto op_type = op_conf->type();
    LOG(INFO) << "creating operator " << operator_name << ", " << op_type;

    if (is_wildcard || !creator_list.count(op_conf->type())) {
      CreateWildcardOP(model, op_conf, index);
    } else {
      Creator f = creator_list[op_conf->type()];
      // create llga op, and if it fails, create wildcard op.
      if (!(this->*f)(model, op_conf, index))
        CreateWildcardOP(model, op_conf, index);
    }
  }

  void InitLogicalTensors(Model *model, OperatorConfig* op_conf,
                          vector<logical_tensor>* inputs, vector<logical_tensor>* outputs);
  void CreateWildcardOP(Model *model, OperatorConfig* op_conf, int index);
  bool CreateSoftmaxOp(Model *model, OperatorConfig* op_conf, int index);
  int CreateInnerProductOpFp32(Model *model, const vector<logical_tensor> &inputs, int index,
                               bool has_bias, bool transpose_a_, bool transpose_b_);
  int CreateInnerProductOpInt8(Model *model, const vector<logical_tensor> &inputs, int index,
                               bool has_bias, bool transpose_a_, bool transpose_b_, bool append_sum);
  bool CreateInnerProductOp(Model *model, OperatorConfig* op_conf, int index);
  bool CreateQuantizeOp(Model *model, OperatorConfig* op_conf, int index);
  bool CreateBinaryAddOp(Model *model, OperatorConfig* op_conf, int index);
  bool CreateLayerNormOp(Model *model, OperatorConfig* op_conf, int index);
  bool CreateReshapeOp(Model *model, OperatorConfig* op_conf, int index);
  bool CreateMatmulOp(Model *model, OperatorConfig* op_conf, int index);

 private:
  LLGAOPCreator() {
    bool llga_enable = (getenv("LLGA_ENABLE") != NULL);
    LOG(INFO) << "LLGA_ENABLE: " << llga_enable << std::endl;
    if (llga_enable) {
      creator_list["InnerProduct"] = &LLGAOPCreator::CreateInnerProductOp;
      creator_list["Quantize"] = &LLGAOPCreator::CreateQuantizeOp;
      creator_list["Softmax"] = &LLGAOPCreator::CreateSoftmaxOp;
      // TODO(lzw): following operators have bugs.
      // creator_list["BinaryAdd"] = &LLGAOPCreator::CreateBinaryAddOp;
      // creator_list["LayerNorm"] = &LLGAOPCreator::CreateLayerNormOp;
    //   creator_list["Reshape"] = &LLGAOPCreator::CreateReshapeOp;
      // creator_list["Matmul"] = &LLGAOPCreator::CreateMatmulOp;
    }
  }
  LLGAOPCreator(const LLGAOPCreator&) {}

  CreatorRegistry creator_list;
};

}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_LLGA_OPERATORS_LLGA_OP_CREATOR_HPP_
