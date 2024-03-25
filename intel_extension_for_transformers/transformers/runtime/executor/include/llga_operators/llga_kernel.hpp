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
#include <memory>

#include "../operator.hpp"
#include "llga_op_creator.hpp"

namespace executor {

/**
 * @brief LLGA kernel for partition to compile and execute.
 *
 */

class LLGAKernel : public Operator {
 public:
  LLGAKernel(const shared_ptr<OperatorConfig>& conf, LLGAINFO* llga_info) : Operator(conf), llga_info_(llga_info) {
    // create llgakernel from llga_info, just for gtest.
    // one config maps only one llga kernel.
    auto partitions = llga_info->GetPartitions();
    assert(partitions.size() == 1);
    partition_ = partitions[0];
  }

  LLGAKernel(const shared_ptr<OperatorConfig>& conf,
                     LLGAINFO* llga_info,
                     const dnnl::graph::partition& partition)
                     : Operator(conf), llga_info_(llga_info), partition_(partition) {}

  virtual ~LLGAKernel() {}

  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  dnnl::graph::partition partition_;
  dnnl::graph::compiled_partition cp_;
  LLGAINFO* llga_info_ = nullptr;
  std::pair<int, int> inplace_index_{-1, -1};
  vector<logical_tensor> inputs_lt, outputs_lt;
  vector<dnnl::graph::tensor> inputs_ts, outputs_ts;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_LLGA_OPERATORS_LLGA_KERNEL_HPP_
