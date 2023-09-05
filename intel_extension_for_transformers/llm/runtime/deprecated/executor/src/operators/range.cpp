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

#include "range.hpp"

#include "common.hpp"

namespace executor {

RangeOperator::RangeOperator(const shared_ptr<OperatorConfig>& conf)
    : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("start");
  start_ = (iter != attrs_map.end() && iter->second != "")
               ? StringToNum<int>(iter->second)
               : 0;
  iter = attrs_map.find("step");
  step_ = (iter != attrs_map.end() && iter->second != "")
              ? StringToNum<int>(iter->second)
              : 1;
  iter = attrs_map.find("end_with_shape");
  end_with_tensor_ = (iter != attrs_map.end() && iter->second != "")
                         ? StringToNum<int>(iter->second)
                         : -1;
  iter = attrs_map.find("end");
  end_ = (iter != attrs_map.end() && iter->second != "")
             ? StringToNum<int>(iter->second)
             : -1;
  iter = attrs_map.find("algorithm");
  algo_ = (iter != attrs_map.end() && iter->second != "") ? true : false;
}

void RangeOperator::Reshape(const vector<Tensor*>& input,
                            const vector<Tensor*>& output) {
  shape_ = input[0]->shape();
  output[0]->set_shape(shape_);
  output[0]->set_dtype("fp32");
  if (end_ != -1) {
    std::vector<int64_t> newshape;
    newshape.push_back(end_ / step_);
    output[0]->set_shape(newshape);
  }
  if (end_with_tensor_ != -1) {
    end_ = input[0]->shape()[end_with_tensor_];
    if (algo_) {
      end_ += input[1]->shape()[1];
    }
    std::vector<int64_t> newshape;
    newshape.push_back(end_ / step_);
    output[0]->set_shape(newshape);
  }
}

void RangeOperator::Forward(const vector<Tensor*>& input,
                            const vector<Tensor*>& output) {
  std::vector<int64_t> tmp = output[0]->shape();
  if (output[0]->shape()[0] < 128) {
    std::vector<int64_t> newshape{128};
    output[0]->set_shape(newshape);
  }
  auto dst_data = reinterpret_cast<float*>(output[0]->mutable_data());
#pragma omp parallel for
  for (int i = 0; i < end_; ++i) dst_data[i] = start_ + i * step_;
  output[0]->set_shape(tmp);
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(Range);
}  // namespace executor
