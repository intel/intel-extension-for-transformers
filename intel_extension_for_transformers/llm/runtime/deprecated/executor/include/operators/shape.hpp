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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_SHAPE_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_SHAPE_HPP_
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "../operator.hpp"

namespace executor {

/**
 * @brief A Shape operator.
 *
 */

class ShapeOperator : public Operator {
 public:
  explicit ShapeOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
    auto attrs_map = operator_conf_->attributes();
    auto iter = attrs_map.find("start");
    if (iter != attrs_map.end())
      start_ = stoi(iter->second);
  }

  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  int64_t start_ = 0, end_ = 0;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_SHAPE_HPP_
