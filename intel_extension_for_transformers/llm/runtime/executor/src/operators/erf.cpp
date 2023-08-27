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

#include "erf.hpp"

#include "common.hpp"

namespace executor {

void ErfOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  output[0]->set_dtype(input[0]->dtype());
}

void ErfOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  output[0]->set_shape(input[0]->shape());
}

vector<vector<string>> ErfOperator::InplacePairs(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<vector<string>> inplace_pairs;
  // skip inplace in debug mode
  if (this->get_execution_mode() == ExecutionMode::DEBUG) {
    return inplace_pairs;
  }
  // input[0] -> output[0]
  if (input[0]->left_life() == 1) {
    inplace_pairs.emplace_back(vector<string>({input[0]->name(), output[0]->name()}));
  }
  return inplace_pairs;
}

void ErfOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  Tensor* src_ptr = input[0];
  Tensor* dst_ptr = output[0];
  auto src_data = reinterpret_cast<float*>(src_ptr->mutable_data());
  if (input[0]->left_life() == 1 && this->get_execution_mode() != ExecutionMode::DEBUG) {
    src_ptr->unref_data(true);
    dst_ptr->set_data(src_data);
  }

  auto dst_data = reinterpret_cast<float*>(dst_ptr->mutable_data());
  auto size = input[0]->size();
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    dst_data[i] = std::erf(src_data[i]);
  }
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(Erf);
}  // namespace executor
