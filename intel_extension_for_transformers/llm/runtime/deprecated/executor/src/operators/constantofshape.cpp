//  Copyright (c) 2023 Intel Corporation
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

#include "constantofshape.hpp"

namespace executor {

ConstantOfShapeOperator::ConstantOfShapeOperator(
    const shared_ptr<OperatorConfig>& conf)
    : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("value");
  constant_value_ = (iter != attrs_map.end() && iter->second != "")
                        ? StringToNum<int>(attrs_map["value"])
                        : 0;
  iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    output_dtype_ = attrs_map["output_dtype"];
  }
  iter = attrs_map.find("tensor");
  if (iter != attrs_map.end()) {
    is_tensor_ = true;
  }
  iter = attrs_map.find("trilu");
  if (iter != attrs_map.end()) {
    is_trilu_ = true;
    trilu_k_ = StringToNum<int>(attrs_map["trilu"]);
  }
  iter = attrs_map.find("mode");
  if (iter != attrs_map.end()) {
    mode_ = (attrs_map["mode"]);
  }
}

ConstantOfShapeOperator::~ConstantOfShapeOperator() {}

void ConstantOfShapeOperator::Reshape(const vector<Tensor*>& input,
                                      const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  if (is_tensor_) {
    dst_shape_ = input[0]->shape();
    if (mode_ == "llama") {
      dst_shape_ = {1, 1, input[1]->shape()[1], input[0]->shape()[3]};
    }
  } else {
    for (int i = 0; i < input[0]->size(); ++i) {
      auto src_data = input[0]->mutable_data();
      dst_shape_.emplace_back(*(reinterpret_cast<int*>(src_data)));
    }
  }
  // 1.2 Get tensor's adjusted shapes
  // dst shape
  output[0]->set_shape(dst_shape_);
  output[0]->set_dtype(output_dtype_);
  array_size_ = output[0]->size();
}
// 2. inference kernel(for int8 and f32)
void ConstantOfShapeOperator::Forward(const vector<Tensor*>& input,
                                      const vector<Tensor*>& output) {
  float* dst_data = static_cast<float*>(output[0]->mutable_data());
  if (output_dtype_ == "fp32") {
    if (mode_ == "llama" && (dst_shape_[3] - dst_shape_[2] != 1)) {
        memset(dst_data, 0, output[0]->size() * sizeof(float));
    } else {
        for (int i = 0; i < output[0]->size(); ++i) {
          dst_data[i] = static_cast<float>(constant_value_);
        }
    }
  }

  if (is_trilu_ && dst_shape_.size() == 3) {
    for (int i = 0; i < dst_shape_[0]; ++i) {
      for (int j = 0; j < dst_shape_[1]; ++j) {
        int range =
            j + trilu_k_ >= dst_shape_[2] ? dst_shape_[2] : j + trilu_k_;
        for (int k = 0; k < range; ++k) {
          reinterpret_cast<float*>(dst_data)[dst_shape_[1] * i + dst_shape_[2] * j + k] = 0;
        }
      }
    }
  }

  if (is_trilu_ && dst_shape_.size() == 4 && mode_ == "llama") {
    int col = dst_shape_[2];
    int row = dst_shape_[3];
    for (int j = 0; j < col; ++j) {
      for (int k = 0; k < j + trilu_k_; ++k) {
        reinterpret_cast<float*>(dst_data)[row * j  + k ] = 0;
      }
    }
  }

  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(ConstantOfShape);

}  // namespace executor
