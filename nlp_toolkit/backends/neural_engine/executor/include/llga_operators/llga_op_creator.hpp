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

#include "llga_info.hpp"
#include "conf.hpp"

namespace executor {

class LLGAOPCreator {
 public:
  typedef bool (LLGAOPCreator::*Creator)(LLGAINFO* llga_info, const OperatorConfig& op_conf, int index);
  typedef std::unordered_map<string, Creator> CreatorRegistry;

  static LLGAOPCreator& GetInstance() {
    static LLGAOPCreator ins;
    return ins;
  }

  void CreateOP(LLGAINFO* llga_info, const OperatorConfig& op_conf, int index = 0, bool fallback = false) {
    auto operator_name = op_conf.name();
    auto op_type = op_conf.type();
    LOG(INFO) << "creating operator " << operator_name << ", " << op_type;

    if (fallback || !creator_list.count(op_conf.type())) {
      if (!fallback)
        LOG(WARNING) << "Failed to create " << op_conf.name() << " by llga, " << op_conf.type() << " is not supported"
                     << ", fallback will be executed";
      CreateWildcardOP(llga_info, op_conf, index);
    } else {
      Creator f = creator_list[op_conf.type()];
      // create llga op, and if it fails, create wildcard op.
      if (!(this->*f)(llga_info, op_conf, index)) {
        LOG(WARNING) << "Failed to create " << op_conf.name() << " by llga, fallback will be executed";
        CreateWildcardOP(llga_info, op_conf, index);
      }
    }
  }

  void CreateWildcardOP(LLGAINFO* llga_info, const OperatorConfig& op_conf, int index);
  bool CreateSoftmaxOp(LLGAINFO* llga_info, const OperatorConfig& op_conf, int index);
  int CreateInnerProductOpFp32(LLGAINFO* llga_info, const vector<logical_tensor> &inputs, int index,
                               bool has_bias, bool transpose_a_, bool transpose_b_);
  int CreateInnerProductOpInt8(LLGAINFO* llga_info, const vector<logical_tensor> &inputs, int index,
                               bool has_bias, bool transpose_a_, bool transpose_b_, bool append_sum);
  bool CreateInnerProductOp(LLGAINFO* llga_info, const OperatorConfig& op_conf, int index);
  bool CreateQuantizeOp(LLGAINFO* llga_info, const OperatorConfig& op_conf, int index);
  bool CreateBinaryAddOp(LLGAINFO* llga_info, const OperatorConfig& op_conf, int index);
  bool CreateLayerNormOp(LLGAINFO* llga_info, const OperatorConfig& op_conf, int index);
  bool CreateReshapeOp(LLGAINFO* llga_info, const OperatorConfig& op_conf, int index);
  bool CreateMatmulOp(LLGAINFO* llga_info, const OperatorConfig& op_conf, int index);

 private:
  LLGAOPCreator() {
    creator_list["InnerProduct"] = &LLGAOPCreator::CreateInnerProductOp;
    creator_list["Quantize"] = &LLGAOPCreator::CreateQuantizeOp;
    creator_list["Softmax"] = &LLGAOPCreator::CreateSoftmaxOp;
    creator_list["BinaryAdd"] = &LLGAOPCreator::CreateBinaryAddOp;
    creator_list["LayerNorm"] = &LLGAOPCreator::CreateLayerNormOp;  // can not inplace, affecting performance
  //   creator_list["Reshape"] = &LLGAOPCreator::CreateReshapeOp;
    // creator_list["Matmul"] = &LLGAOPCreator::CreateMatmulOp;
  }
  LLGAOPCreator(const LLGAOPCreator&) {}

  CreatorRegistry creator_list;
};

}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_LLGA_OPERATORS_LLGA_OP_CREATOR_HPP_
