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

#include "sequence_length.hpp"

#include "common.hpp"

namespace executor {
SequenceLengthOperator::SequenceLengthOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {}
void SequenceLengthOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  output[0]->set_dtype("s32");
}

void SequenceLengthOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  mask_shape_ = input[0]->shape();
  mask_stride_ = GetStrides(mask_shape_);
  int bs = mask_shape_[0];
  output[0]->set_shape({bs});
}

void SequenceLengthOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  auto mask_data = static_cast<const int32_t*>(input[0]->data());
  auto dst_data = static_cast<int32_t*>(output[0]->mutable_data());
#pragma omp parallel for
  for (int i = 0; i < mask_shape_[0]; ++i) {
    for (int j = mask_shape_[1] - 1; j >= 0; --j) {
      auto idx = i * mask_stride_[0] + j;
      if (mask_data[idx] != 0) {
        dst_data[i] = j + 1;
        break;
      }
    }
  }
  // 2. unref tensors
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(SequenceLength);
}  // namespace executor
