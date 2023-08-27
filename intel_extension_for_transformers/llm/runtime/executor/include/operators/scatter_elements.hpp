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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_GATHER_ELEMENTS_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_GATHER_ELEMENTS_HPP_
#include <vector>
#include <memory>

#include "../operator.hpp"

namespace executor {

/**
 * @brief A Scatter element operator.
 *
 */

class ScatterElementsOperator : public Operator {
 public:
  explicit ScatterElementsOperator(const shared_ptr<OperatorConfig>& conf);
  virtual ~ScatterElementsOperator() {}

  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  int64_t axis_ = -1;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_GATHER_ELEMENTS_HPP_
