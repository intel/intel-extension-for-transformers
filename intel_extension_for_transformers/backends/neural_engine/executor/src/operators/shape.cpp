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

#include "shape.hpp"

#include "common.hpp"

namespace executor {

void ShapeOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  output[0]->set_dtype("int32");
}

void ShapeOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  auto rank = input[0]->shape().size();
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("end");
  if (iter != attrs_map.end()) {
    end_ = stoi(iter->second);
    if (end_ < 0) end_ += rank;
  } else {
    end_ = rank;
  }
  if (start_ < 0) start_ += rank;
  vector<int64_t> out_shape{end_ - start_};
  output[0]->set_shape(out_shape);
}

void ShapeOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  auto dst_data = reinterpret_cast<int32_t*>(output[0]->mutable_data());
  auto src_shape = input[0]->shape();
  int index = 0;
  for (int i = start_; i < end_; i++) {
    dst_data[index] = src_shape[i];
    index++;
  }

  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(Shape);
}  // namespace executor
