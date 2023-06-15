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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_CONSTANTOFSHAPE_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_CONSTANTOFSHAPE_HPP_

#include <string>
#include <vector>
#include <memory>

#include "../operator.hpp"


namespace executor {


/**
 * @brief A CONSTANTOFSHAPE operator.
 *
 */

class ConstantOfShapeOperator : public Operator {
 public:
  explicit ConstantOfShapeOperator(const shared_ptr<OperatorConfig>& conf);
  virtual ~ConstantOfShapeOperator();

  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  string output_dtype_ = "fp32";
  vector<int64_t> dst_shape_;
  // default constant value is 0
  int constant_value_ = 0;
  bool is_tensor_ = false;
  bool is_trilu_ = false;
  int trilu_k_ = 0;
  int trilu_upper = 1;
  int array_size_;
  string mode_ = "None";
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_CONSTANTOFSHAPE_HPP_
