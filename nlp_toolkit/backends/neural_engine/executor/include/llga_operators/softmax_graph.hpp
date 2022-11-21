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

#ifndef ENGINE_EXECUTOR_INCLUDE_LLGA_OPERATORS_SOFTMAX_GRAPH_HPP_
#define ENGINE_EXECUTOR_INCLUDE_LLGA_OPERATORS_SOFTMAX_GRAPH_HPP_
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "../operator.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

namespace executor {

using logical_tensor = dnnl::graph::logical_tensor;
using data_type = dnnl::graph::logical_tensor::data_type;
using layout_type = dnnl::graph::logical_tensor::layout_type;
using property_type = dnnl::graph::logical_tensor::property_type;

/**
 * @brief A Softmax operator.
 *
 */

class SoftmaxGraphOperator : public Operator {
 public:
  explicit SoftmaxGraphOperator(const shared_ptr<OperatorConfig>& conf);
  virtual ~SoftmaxGraphOperator() {}

  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  dnnl::graph::graph g_;
  dnnl::graph::engine eng_ {dnnl::graph::engine::kind::cpu, 0};
  dnnl::graph::stream strm_ {eng_};
  vector<logical_tensor> logical_inputs_;
  vector<logical_tensor> logical_outputs_;
  dnnl::graph::partition partition_;
  dnnl::graph::compiled_partition cp_;

  int axis_;
  string output_dtype_ = "fp32";
  Tensor* dst_min_ = nullptr;
  Tensor* dst_max_ = nullptr;
  Tensor* src_ = nullptr;
  Tensor* dst_ = nullptr;

  void MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output);
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_LLGA_OPERATORS_SOFTMAX_GRAPH_HPP_

