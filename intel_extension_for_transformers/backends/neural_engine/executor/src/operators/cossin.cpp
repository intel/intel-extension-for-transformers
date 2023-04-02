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

#include "cossin.hpp"

namespace executor {

CosSinOperator::CosSinOperator(const shared_ptr<OperatorConfig>& conf)
    : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("algorithm");
  if (iter != attrs_map.end()) {
    algorithm_ = iter->second;
  }
}

CosSinOperator::~CosSinOperator() {}

void CosSinOperator::Reshape(const vector<Tensor*>& input,
                             const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  vector<int64_t> src_shape = input[0]->shape();

  // 1.2 Get tensor's adjusted shapes
  // dst shape
  output[0]->set_shape(src_shape);
  output[0]->set_dtype(output_dtype_);
  array_size_ = input[0]->size();
}
// 2. inference kernel(for int8 and f32)
void CosSinOperator::Forward(const vector<Tensor*>& input,
                             const vector<Tensor*>& output) {
  auto src_data = input[0]->mutable_data();
  auto dst_data = output[0]->mutable_data();

  Eigen::Map<Eigen::ArrayXf> input_array(reinterpret_cast<float*>(src_data),
                                         array_size_);
  Eigen::Map<Eigen::ArrayXf> output_array(reinterpret_cast<float*>(dst_data),
                                          array_size_);
  if (algorithm_ == "sin") {
    output_array = input_array.sin();
  } else {
    output_array = input_array.cos();
  }

  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(CosSin);

}  // namespace executor
