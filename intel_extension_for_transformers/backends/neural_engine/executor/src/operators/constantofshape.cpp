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

#include "constantofshape.hpp"

namespace executor {

ConstantOfShapeOperator::ConstantOfShapeOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("value");
  constant_value_ = (iter != attrs_map.end() && iter->second != "") ? StringToNum<int>(attrs_map["value"]) : 0;
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
}

ConstantOfShapeOperator::~ConstantOfShapeOperator() {
}

void ConstantOfShapeOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  if (is_tensor_) {
    dst_shape_ = input[0]->shape();
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
void ConstantOfShapeOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  auto dst_data = output[0]->mutable_data();
  if (output_dtype_ == "fp32") {
    Eigen::Map<Eigen::ArrayXf> output_array(reinterpret_cast<float*>(dst_data), array_size_);
    output_array = Eigen::ArrayXf::Constant(array_size_, static_cast<float>(constant_value_));
  // } else if (output_dtype_ == "int32"){
  //   Eigen::Map<Eigen::ArrayXd> output_array(reinterpret_cast<int*>(dst_data), array_size_);
  //   output_array = Eigen::Dynamic1D::Constant(array_size_, constant_value_);
  }
  if (is_trilu_ && dst_shape_.size() == 3) {
    // upper == 0
    // for (int i = 0; i < dst_shape_[0]; ++i) {
    //   for (int j = 0; j < dst_shape_[1] - trilu_k_; ++j) {
    //     for (int k = trilu_k_ + j; k < dst_shape_[2]; ++k) {
    //       reinterpret_cast<float*>(dst_data)[dst_shape_[1] * i + dst_shape_[2] * j + k] = 0;
    //     }
    //   }
    // }

    // upper == 1
    for (int i = 0; i < dst_shape_[0]; ++i) {
      for (int j = 0; j < dst_shape_[1]; ++j) {
        int range = j + trilu_k_ >= dst_shape_[2] ? dst_shape_[2] : j + trilu_k_;
        for (int k = 0; k < range; ++k) {
          reinterpret_cast<float*>(dst_data)[dst_shape_[1] * i + dst_shape_[2] * j + k] = 0;
        }
      }
    }
  }
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(ConstantOfShape);
}  // namespace executor
