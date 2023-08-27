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

#include "squeeze.hpp"

#include "common.hpp"

namespace executor {

void SqueezeOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  output[0]->set_dtype(input[0]->dtype());
}

void SqueezeOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  auto in_rank = input[0]->shape().size();
  vector<int64_t> out_shape = input[0]->shape();
  if (axes_.empty()) {
    out_shape.erase(std::remove(out_shape.begin(), out_shape.end(), 1), out_shape.end());
  } else {
    for (auto axis : axes_) {
      if (axis < 0) axis = in_rank + axis;
      if (axis >= in_rank) {
        LOG(ERROR) << "Axis out of range. Accepted range is [-r, r-1] where r = rank";
        return;
      }
      if (out_shape[axis] == 1) {
        out_shape[axis] = -1;
      } else {
        LOG(ERROR) << "cannot select an axis to squeeze out which has size not equal to one";
        return;
      }
    }
    out_shape.erase(std::remove(out_shape.begin(), out_shape.end(), -1), out_shape.end());
  }
  output[0]->set_shape(out_shape);
}

vector<vector<string>> SqueezeOperator::InplacePairs(const vector<Tensor*>& input, const vector<Tensor*>& output) {
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

void SqueezeOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
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

REGISTER_OPERATOR_CLASS(Squeeze);
}  // namespace executor
