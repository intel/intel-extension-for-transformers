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

#include "split.hpp"

#include "common.hpp"

namespace executor {

SplitOperator::SplitOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("axis");
  if (iter != attrs_map.end()) {
    axis_ = StringToNum<int64_t>(attrs_map["axis"]);
  }
  iter = attrs_map.find("split");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&split_, attrs_map["split"], ",");
  }
}

void SplitOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  for (auto tensor : output) {
    tensor->set_dtype(input[0]->dtype());
  }
}

void SplitOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  src_shape_ = input[0]->shape();
  dst_num_ = output.size();

  // 1.2 Get tensor's adjusted shapes
  // dst shape
  for (int i = 0; i < dst_num_; i++) {
    vector<int64_t> dst_shape = src_shape_;
    dst_shape[axis_] = split_[i];
    auto& dst_tensor_ptr = output[i];
    dst_tensor_ptr->set_shape(dst_shape);
    dst_tensor_ptr->set_dtype(input[0]->dtype());
  }
}
template <typename T>
T* AddrAddOffset(T* src_data, int element_num) {
  auto ret = (intptr_t)src_data + element_num * sizeof(T);
  return reinterpret_cast<T*>(ret);
}
template <typename T>
void SplitCopy(const void* src_data, void* dst_data, int src_index, int dst_index) {
  const T* src_data_T = static_cast<const T*>(src_data);
  T* dst_data_T = reinterpret_cast<T*>(dst_data);
  dst_data_T[dst_index] = src_data_T[src_index];
}

void SplitOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  CHECK_NOTNULL(input[0]);
  // when change data value please use mutable_data
  if (axis_ == 0 && input[0]->left_life() == 1) {
    auto src_data = input[0]->mutable_data();
    input[0]->unref_data(true);
    size_t element_offset = 0;
    for (int output_index = 0; output_index < dst_num_; output_index++) {
      vector<int64_t> dst_shape = src_shape_;
      dst_shape[axis_] = split_[output_index];
      if (input[0]->dtype() == "int32") {
        int32_t* dst_data = AddrAddOffset<int32_t>(reinterpret_cast<int32_t*>(src_data), element_offset);
        output[output_index]->set_data(reinterpret_cast<void*>(dst_data));
      } else if (input[0]->dtype() == "fp32") {
        float* dst_data = AddrAddOffset<float>(reinterpret_cast<float*>(src_data), element_offset);
        output[output_index]->set_data(reinterpret_cast<void*>(dst_data));
      } else if (input[0]->dtype() == "s8") {
        int8_t* dst_data = AddrAddOffset<int8_t>(reinterpret_cast<int8_t*>(src_data), element_offset);
        output[output_index]->set_data(reinterpret_cast<void*>(dst_data));
      }
      element_offset += dst_shape[axis_] * std::accumulate(dst_shape.begin() + 1, dst_shape.end(), 0);
    }
  } else if (axis_ == 1) {
    for (int output_index = 0; output_index < dst_num_; output_index++) {
      vector<int64_t> dst_shape = src_shape_;
      dst_shape[axis_] = split_[output_index];
      for (int i = 0; i < dst_shape[0]; i++) {
        for (int j = 0; j < dst_shape[1]; j++) {
          if (input[0]->dtype() == "fp32") {
            SplitCopy<float>(input[0]->data(), output[output_index]->mutable_data(),
                             i * src_shape_[1] + j + dst_shape[1] * output_index, i * dst_shape[1] + j);
          } else if (input[0]->dtype() == "int32") {
            SplitCopy<int32_t>(input[0]->data(), output[output_index]->mutable_data(),
                               i * src_shape_[1] + j + dst_shape[1] * output_index, i * dst_shape[1] + j);
          } else if (input[0]->dtype() == "s8") {
            SplitCopy<int8_t>(input[0]->data(), output[output_index]->mutable_data(),
                              i * src_shape_[1] + j + dst_shape[1] * output_index, i * dst_shape[1] + j);
          }
        }
      }
    }
    this->unref_tensors(input);
  }
}

REGISTER_OPERATOR_CLASS(Split);
}  // namespace executor
