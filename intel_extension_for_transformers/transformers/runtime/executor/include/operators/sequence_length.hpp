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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_SEQUENCE_LENGTH_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_SEQUENCE_LENGTH_HPP_
#include <memory>
#include <vector>

#include "../operator.hpp"

namespace executor {

/**
 * @brief Calculate the seq_len w/o padding.
 *
 */

class SequenceLengthOperator : public Operator {
 public:
  explicit SequenceLengthOperator(const shared_ptr<OperatorConfig>& conf);
  virtual ~SequenceLengthOperator() {}

  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  std::vector<int64_t> mask_shape_;
  std::vector<int64_t> mask_stride_;
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_SEQUENCE_LENGTH_HPP_
