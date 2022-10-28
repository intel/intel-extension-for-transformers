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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_GATHER_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_GATHER_HPP_
#include <vector>
#include <string>

#include "../operator.hpp"
#include "SparseLib/include/interface.hpp"

namespace executor {

/**
 * @brief A Gather operator.
 *
 */

class GatherOperator : public Operator {
 public:
  explicit GatherOperator(const OperatorConfig& conf);
  virtual ~GatherOperator() {}

  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

  void MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output);

 private:
  Tensor* idx_ = nullptr;
  Tensor* src_ = nullptr;
  Tensor* dst_ = nullptr;
  Tensor* append_ = nullptr;
  std::string idx_axis_;
  std::string src_axis_;
  bool binary_add_ = false;
  jd::gather gather_;
  vector<int64_t> reshape_;
  vector<int64_t> reshape_dims_;
  vector<int64_t> mul_;

  std::vector<void*> rt_data_;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_GATHER_HPP_
