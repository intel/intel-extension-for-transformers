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

#include "expand_indices.hpp"

namespace executor {

ExpandIndicesOperator::ExpandIndicesOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("position");
  if (iter != attrs_map.end()) {
    executor::StringSplit<int64_t>(&position_, attrs_map["position"], ",");
  }
}

ExpandIndicesOperator::~ExpandIndicesOperator() {}

void ExpandIndicesOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  const vector<int64_t>& tgt_shape = input[1]->shape();
  src_shape_ = input[0]->shape();
  src_strides_ = GetStrides(input[0]->shape());
  dst_shape_ = src_shape_;
  new2old_.resize(dst_shape_.size());
  old2new_.resize(dst_shape_.size());
  for (int i = 0; i < dst_shape_.size(); i++) new2old_[i] = i;
  std::vector<int64_t>::iterator dst_it = dst_shape_.begin();
  std::vector<int64_t>::iterator new2old_it = new2old_.begin();

  for (auto pos : position_)
    if (pos == -1) {
      dst_shape_.push_back(tgt_shape[tgt_shape.size() - 1]);
      new2old_.push_back(-1);
    } else {
      dst_shape_.insert(dst_it + pos, tgt_shape[pos]);
      dst_it = dst_shape_.begin();
      new2old_.insert(new2old_it + pos, -1);
      new2old_it = new2old_.begin();
    }
  int last = -1;
  for (int i = 0; i < new2old_.size(); i++)
    if (new2old_[i] != -1) {
      old2new_[new2old_[i]] = i;
      last--;
    } else {
      new2old_[i] = last;
    }
  dst_strides_ = GetStrides(dst_shape_);
  // 1.2 Set dst dtype and shape
  output[0]->set_shape(dst_shape_);
  output[0]->set_dtype("int32");  // only support int32 now
}

vector<vector<string>> ExpandIndicesOperator::InplacePairs(const vector<Tensor*>& input,
                                                           const vector<Tensor*>& output) {
  vector<vector<string>> inplace_pairs;
  // skip inplace in debug mode
  if (this->get_execution_mode() == ExecutionMode::DEBUG) {
    return inplace_pairs;
  }
  // input[0] -> output[0]
  if (input[0]->size() == output[0]->size() && input[0]->left_life() == 1) {
    inplace_pairs.emplace_back(vector<string>({input[0]->name(), output[0]->name()}));
  }
  return inplace_pairs;
}

void ExpandIndicesOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  int32_t* src_data = static_cast<int32_t*>(input[0]->mutable_data());
  int32_t* dst_data = static_cast<int32_t*>(output[0]->mutable_data());
  int old_size = input[0]->size();
  vector<Tensor*> inputs(input);
  if (output[0]->size() == old_size) {
    if (input[0]->left_life() == 1 && this->get_execution_mode() != ExecutionMode::DEBUG) {
      input[0]->unref_data(true);
      output[0]->set_data(src_data);
      inputs = {input[1]};
    } else {
      string data_type = input[0]->dtype();
      memcpy(dst_data, src_data, old_size * type2bytes[data_type]);
      LOG(WARNING) << "post tensor will be used by multi node...";
    }
  } else {
    // fill the last dim without repeat
    for (int old_index = 0; old_index < input[0]->size(); old_index++) {
      int tmp = old_index;
      int new_index = 0;
      for (int j = 0; j < old2new_.size(); j++) {
        new_index += tmp / src_strides_[j] * dst_strides_[old2new_[j]];
        tmp %= src_strides_[j];
      }
      if (new2old_.back() < 0) std::fill_n(&dst_data[new_index], dst_shape_.back(), src_data[old_index]);
      // for (int j = 0; j < dst_shape_.back(); j++) dst_data[new_index + j] = src_data[old_index];
      else
        dst_data[new_index] = src_data[old_index];
    }
    // repeat dims (the last dim is filled)
    for (int new_layer = dst_shape_.size() - 2; new_layer >= 0; new_layer--)
      if (new2old_[new_layer] < 0) {
        int inner_block = dst_strides_[new_layer];
        int related_dim_in_old = -new2old_[new_layer] - 1;  // the related dim should be repeated.
        if (related_dim_in_old == 0) {
#pragma omp parallel for
          for (int i = 1; i < dst_shape_[new_layer]; i++)
            std::copy(dst_data, &dst_data[inner_block], &dst_data[i * inner_block]);
        } else {
          int outer_num = old_size / src_strides_[related_dim_in_old - 1];
          for (int old_idx = 0; old_idx < outer_num; old_idx++) {
            int offset = 0;
            int tmp = old_idx * src_strides_[related_dim_in_old - 1];
            for (int j = 0; j < related_dim_in_old; j++) {
              offset += tmp / src_strides_[j] * dst_strides_[old2new_[j]];
              tmp %= src_strides_[j];
            }
#pragma omp parallel for
            for (int i = 1; i < dst_shape_[new_layer]; i++)
              std::copy(&dst_data[offset], &dst_data[offset + inner_block], &dst_data[offset + i * inner_block]);
          }
        }
      }
  }
  this->unref_tensors(inputs);
}

REGISTER_OPERATOR_CLASS(ExpandIndices);
}  // namespace executor
