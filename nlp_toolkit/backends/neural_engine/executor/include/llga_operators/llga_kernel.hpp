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

#ifndef ENGINE_EXECUTOR_INCLUDE_LLGA_OPERATORS_LLGA_KERNEL_HPP_
#define ENGINE_EXECUTOR_INCLUDE_LLGA_OPERATORS_LLGA_KERNEL_HPP_
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

#include "../operator.hpp"
#include "model.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

namespace executor {

/**
 * @brief LLGA kernel for partition to compile and execute.
 *
 */

using logical_tensor = dnnl::graph::logical_tensor;
class Model;  // forward declaration

class LLGAKernel : public Operator {
 public:
  LLGAKernel(const OperatorConfig& conf,
                     Model* model,
                     const dnnl::graph::partition& partition)
                     : Operator(conf), model_(model), partition_(partition) {}

  virtual ~LLGAKernel() {}

  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  dnnl::graph::partition partition_;
  dnnl::graph::compiled_partition cp_;
  Model *model_;
  bool inplace_ = false;
  std::pair<int, int> inplace_index_;
  vector<logical_tensor> inputs_lt, outputs_lt;
  vector<dnnl::graph::tensor> inputs_ts, outputs_ts;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_LLGA_OPERATORS_LLGA_KERNEL_HPP_
