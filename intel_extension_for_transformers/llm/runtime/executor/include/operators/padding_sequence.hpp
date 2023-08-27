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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_PADDING_SEQUENCE_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_PADDING_SEQUENCE_HPP_
#include <vector>
#include <string>
#include <memory>

#include "../operator.hpp"

namespace executor {

/**
 * @brief A Padding Sequence Mask operator.
 *
 */

class PaddingSequenceOperator : public Operator {
 public:
  explicit PaddingSequenceOperator(const shared_ptr<OperatorConfig>& conf);
  virtual ~PaddingSequenceOperator() {}

  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) override;
  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) override;

 private:
  std::vector<int64_t> src_shape_;
  std::vector<int64_t> src_stride_;
  std::vector<int64_t> pad_dst_shape_;
  std::vector<int64_t> pad_dst_stride_;
  float padding_;
  std::vector<int64_t> attr_dst_shape_;
  std::vector<int64_t> dims_;
  bool seq_len_first_ = false;
  string mode_ = "None";
};
}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_PADDING_SEQUENCE_HPP_
