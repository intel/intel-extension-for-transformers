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

#ifndef  ENGINE_EXECUTOR_INCLUDE_LLGA_OPERATORS_INNER_PRODUCT_GRAPH_HPP_
#define  ENGINE_EXECUTOR_INCLUDE_LLGA_OPERATORS_INNER_PRODUCT_GRAPH_HPP_
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
 * @brief A InnerProduct operator.
 *
 */

class InnerProductGraphOperator : public Operator {
 public:
  explicit InnerProductGraphOperator(const shared_ptr<OperatorConfig>& conf);
  virtual ~InnerProductGraphOperator() {}

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

  string output_dtype_ = "fp32";
  Tensor* dst_min_ = nullptr;
  Tensor* dst_max_ = nullptr;
  Tensor* src_ = nullptr;
  Tensor* dst_ = nullptr;
  vector<int64_t> src0_perm_;
  vector<int64_t> src1_perm_;
  Tensor* src0_ = nullptr;
  Tensor* src1_ = nullptr;
  Tensor* bias_ = nullptr;
  bool has_bias_ = false;
  bool transpose_a_ = false;
  bool transpose_b_ = true;
  bool append_sum_ = false;
  bool binary_add_ = false;
  bool tanh_ = false;
  bool gelu_tanh_ = false;
  bool gelu_erf_ = false;
  bool gelu_split_ = false;
  bool sigmoid_ = false;
  bool relu_ = false;
  bool append_eltwise_ = false;
  string append_op_;

  void MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output);
};
}  // namespace executor
#endif  //  ENGINE_EXECUTOR_INCLUDE_LLGA_OPERATORS_INNER_PRODUCT_GRAPH_HPP_

