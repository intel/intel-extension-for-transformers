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

#include "latrange.hpp"

#include "common.hpp"

namespace executor {

LatRangeOperator::LatRangeOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("start");
  start_ = (iter != attrs_map.end() && iter->second != "") ? StringToNum<int>(iter->second) : 0;
  iter = attrs_map.find("step");
  step_ = (iter != attrs_map.end() && iter->second != "") ? StringToNum<int>(iter->second) : 1;
}

void LatRangeOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  shape_ = input[0]->shape();
  output[0]->set_shape(shape_);
  output[0]->set_dtype("int32");
}

void LatRangeOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  auto dst_data = static_cast<int*>(output[0]->mutable_data());
#pragma omp parallel for
  for (int i = 0; i < shape_[1]; ++i) dst_data[i] = start_ + i * step_;
  int stride = shape_[1] * type2bytes["int32"];
#pragma omp parallel for
  for (int i = 1; i < shape_[0]; ++i) memcpy(&dst_data[i * shape_[1]], dst_data, stride);
  // 2. unref tensors
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(LatRange);
}  // namespace executor
