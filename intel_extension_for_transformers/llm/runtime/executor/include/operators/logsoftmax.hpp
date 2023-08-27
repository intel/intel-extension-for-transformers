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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_LOGSOFTMAX_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_LOGSOFTMAX_HPP_
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "../operator.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "kernels/include/interface.hpp"

namespace executor {
using dnnl::algorithm;
using dnnl::engine;
using dnnl::memory;
using dnnl::prop_kind;

/**
 * @brief A Logsoftmax operator.
 *
 */

class LogSoftmaxOperator : public Operator {
 public:
  explicit LogSoftmaxOperator(const shared_ptr<OperatorConfig>& conf);
  virtual ~LogSoftmaxOperator() {}
  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  int axis_;
  string output_dtype_ = "fp32";
  dnnl::engine eng_ = engine(engine::kind::cpu, 0);
  dnnl::softmax_forward logsoftmax_p_;
  memory src_m_;
  memory dst_m_;
  unordered_map<int, memory> memory_args_;
  Tensor* src_ = nullptr;
  Tensor* dst_ = nullptr;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_LOGSOFTMAX_HPP_

