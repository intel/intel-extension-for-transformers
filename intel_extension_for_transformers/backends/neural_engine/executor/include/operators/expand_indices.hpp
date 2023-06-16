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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_EXPAND_INDICES_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_EXPAND_INDICES_HPP_
#include <algorithm>
#include <vector>
#include <memory>
#include <string>

#include "../common.hpp"
#include "../operator.hpp"

namespace executor {

/**
 * @brief A Reshape operator.
 *
 */

class ExpandIndicesOperator : public Operator {
 public:
  explicit ExpandIndicesOperator(const shared_ptr<OperatorConfig>& conf);
  virtual ~ExpandIndicesOperator();

  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  vector<vector<string>> InplacePairs(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

  vector<int64_t> position_;
  vector<int64_t> src_shape_;
  vector<int64_t> dst_shape_;
  vector<int64_t> src_strides_;
  vector<int64_t> dst_strides_;
  vector<int64_t> new2old_;  // negative represent exand dim, and record the related old dim
  vector<int64_t> old2new_;  // record where the old dim in new shape.
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_EXPAND_INDICES_HPP_
