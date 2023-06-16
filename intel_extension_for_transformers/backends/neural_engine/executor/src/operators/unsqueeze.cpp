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

#include "unsqueeze.hpp"

#include "common.hpp"

namespace executor {

void UnsqueezeOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  output[0]->set_dtype(input[0]->dtype());
}

void UnsqueezeOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  auto output_rank = input[0]->shape().size() + axes_.size();
  for (auto item : axes_) {
    if ((item <= output_rank - 1) && (item >= -1 * output_rank)) {
      LOG(ERROR) << "Axis out of range. Accepted range is [-r, r-1] where r = rank";
    }
  }
  vector<int64_t> out_shape(output_rank, -1);
  for (auto item : axes_) {
    if (item < 0) item = output_rank + item;
    if (out_shape[item] == 1) {
      LOG(ERROR) << "Axis duplicates";
    }
    out_shape[item] = 1;
  }

  int idx = 0;
  for (int i = 0; i < out_shape.size(); i++) {
    if (out_shape[i] == 1) {
      continue;
    }
    out_shape[i] = input[0]->shape()[idx++];
  }

  output[0]->set_shape(out_shape);
}

vector<vector<string>> UnsqueezeOperator::InplacePairs(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<vector<string>> inplace_pairs;
  // skip inplace in debug mode
  if (this->get_execution_mode() == ExecutionMode::DEBUG) {
    return inplace_pairs;
  }
  // input[0] -> output[0]
  if (input[0] != nullptr && input[0]->left_life() == 1) {
    inplace_pairs.emplace_back(vector<string>({input[0]->name(), output[0]->name()}));
  }
  return inplace_pairs;
}

void UnsqueezeOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  Tensor* src_ptr = input[0];
  Tensor* dst_ptr = output[0];
  auto src_data = src_ptr->mutable_data();
  if (input[0]->left_life() == 1 && this->get_execution_mode() != ExecutionMode::DEBUG) {
    src_ptr->unref_data(true);
    dst_ptr->set_data(src_data);
  } else {
    // just copy data
    memcpy(dst_ptr->mutable_data(), src_data, src_ptr->size() * type2bytes[src_ptr->dtype()]);
    this->unref_tensors(input);
  }
}

REGISTER_OPERATOR_CLASS(Unsqueeze);
}  // namespace executor
