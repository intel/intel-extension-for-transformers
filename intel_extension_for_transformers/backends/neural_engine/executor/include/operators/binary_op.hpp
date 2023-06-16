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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_BINARY_OP_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_BINARY_OP_HPP_
#include <unordered_map>
#include <vector>
#include <memory>
#include <string>

#include "../operator.hpp"
#include "oneapi/dnnl/dnnl.hpp"

namespace executor {
using dnnl::algorithm;
using dnnl::engine;
using dnnl::memory;
using dnnl::prop_kind;

/**
 * @brief A BinaryOp operator.
 *
 */

class BinaryOpOperator : public Operator {
 public:
  explicit BinaryOpOperator(const shared_ptr<OperatorConfig>& conf);
  virtual ~BinaryOpOperator() {}

  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  vector<vector<string>> InplacePairs(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  std::string output_dtype_ = "fp32";
  dnnl::engine eng_;
  dnnl::stream stream_;
  algorithm algo_ = algorithm::undef;
  dnnl::binary binary_prim_;
  memory src_0_mem_, src_1_mem_, dst_mem_;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_BINARY_OP_HPP_
