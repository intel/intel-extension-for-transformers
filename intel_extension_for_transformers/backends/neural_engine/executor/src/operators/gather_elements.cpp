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

#include "gather_elements.hpp"

namespace executor {

GatherElementsOperator::GatherElementsOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("axis");
  axis_ = (iter != attrs_map.end() && iter->second != "") ? stoi(iter->second) : 0;
}

void GatherElementsOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  Tensor* data = input[0];

  Tensor* indices = input[1];
  // 1.2 Get tensor's adjusted shapes
  dst_shape_ = indices->shape();
  vector<int64_t> data_shape(data->shape());

  // 1.5 Set dst tensor shape
  auto& dst_tensor_ptr = output[0];
  auto& input_data_dtype = data->dtype();
  dst_tensor_ptr->set_shape(dst_shape_);
  dst_tensor_ptr->set_dtype(input_data_dtype);
  outer_ = 1;
  inner_ = 1;
  for (int i = 0; i < input[0]->shape().size(); i++) {
    if (i < axis_) outer_ *= input[0]->shape()[i];
    if (i > axis_) inner_ *= input[0]->shape()[i];
  }

  src_stride_ = GetStrides(input[0]->shape());
  dst_stride_ = GetStrides(dst_shape_);
  inner_block_size_ = inner_ * type2bytes[input[0]->dtype()];
}

void GatherElementsOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  char* old_data = reinterpret_cast<char*>(input[0]->mutable_data());
  int32_t* idx_data = reinterpret_cast<int32_t*>(input[1]->mutable_data());
  char* new_data = reinterpret_cast<char*>(output[0]->mutable_data());
  if (axis_ == dst_shape_.size() - 1) {
#pragma omp parallel for
    for (int i = 0; i < outer_; i++) {
      int src_outer = i * src_stride_[axis_ - 1];
      int dst_outer = i * dst_stride_[axis_ - 1];
      if (i + 1 < outer_)
        _mm_prefetch(old_data + (i + 1) * src_stride_[axis_ - 1] * type2bytes[input[0]->dtype()], _MM_HINT_T0);
      int len = 0;

#if __AVX512F__
      len = dst_shape_[axis_] - (dst_shape_[axis_] % 16);
#pragma omp simd
      for (int j = 0; j < len; j += 16) {
        __m512i vidx = _mm512_loadu_si512(idx_data + dst_outer + j);
        __m512 tmp = _mm512_i32gather_ps(vidx, old_data + src_outer * type2bytes[input[0]->dtype()], 4);
        _mm512_storeu_ps(new_data + (dst_outer + j) * type2bytes[input[0]->dtype()], tmp);
      }
#endif
#pragma omp simd
      for (int j = len; j < dst_shape_[axis_]; j++) {
        int idx = idx_data[dst_outer + j];
        memcpy(new_data + (dst_outer + j) * type2bytes[input[0]->dtype()],
               old_data + (src_outer + idx) * type2bytes[input[0]->dtype()], type2bytes[input[0]->dtype()]);
      }
    }
  } else {
#pragma omp parallel for
    for (int i = 0; i < outer_; i++) {
      int src_outer = i * src_stride_[axis_ - 1];
      int dst_outer = i * dst_stride_[axis_ - 1];
#pragma omp simd
      for (int j = 0; j < dst_shape_[axis_]; j++) {
        if (j + 1 < dst_shape_[axis_])
          _mm_prefetch(
              old_data + (src_outer + idx_data[dst_outer + (j + 1) * inner_] * inner_) * type2bytes[input[0]->dtype()],
              _MM_HINT_T0);
        memcpy(new_data + (dst_outer + j * inner_) * type2bytes[input[0]->dtype()],
               old_data + (src_outer + idx_data[dst_outer + j * inner_] * inner_) * type2bytes[input[0]->dtype()],
               inner_block_size_);
      }
    }
  }
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(GatherElements);
}  // namespace executor
