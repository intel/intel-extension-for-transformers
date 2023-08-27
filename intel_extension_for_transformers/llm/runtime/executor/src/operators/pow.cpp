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

#include "pow.hpp"

#include "common.hpp"

namespace executor {

void PowOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (input[0]->dtype() != "fp32") {
    LOG(ERROR) << "dtype " << input[0]->dtype() << " is not supported by pow.";
  }
  if (input[1]->dtype() != "fp32") {
    LOG(ERROR) << "dtype " << input[1]->dtype() << " is not supported by pow.";
  }
  output[0]->set_dtype(input[0]->dtype());
}

void PowOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  src0_shape_ = input[0]->shape();
  src1_shape_ = input[1]->shape();

  src0_stride_ = GetStrides(src0_shape_);
  src1_stride_ = GetStrides(src1_shape_);

  if (src0_shape_.size() < src1_shape_.size()) {
    int diff_len = src1_shape_.size() - src0_shape_.size();
    for (int i = 0; i < diff_len; i++) {
      src0_shape_.insert(src0_shape_.begin(), 1);
      src0_stride_.insert(src0_stride_.begin(), 0);
    }
  } else if (src0_shape_.size() > src1_shape_.size()) {
    int diff_len = src0_shape_.size() - src1_shape_.size();
    for (int i = 0; i < diff_len; i++) {
      src1_shape_.insert(src1_shape_.begin(), 1);
      src1_stride_.insert(src1_stride_.begin(), 0);
    }
  }

  vector<int64_t> out_shape;
  for (int i = 0; i < src0_shape_.size(); i++) {
    if (src0_shape_[i] != src1_shape_[i] && src0_shape_[i] != 1 && src1_shape_[i] != 1) {
      LOG(ERROR) << "can not broadcast!";
    }
    out_shape.push_back(max(src0_shape_[i], src1_shape_[i]));
  }
  out_stride_ = GetStrides(out_shape);
  output[0]->set_shape(out_shape);
}

void PowOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  auto src0_data = reinterpret_cast<const float*>(input[0]->data());
  auto src1_data = reinterpret_cast<const float*>(input[1]->data());
  auto dst_data = reinterpret_cast<float*>(output[0]->mutable_data());

  int size = output[0]->size();
  int dims = src0_shape_.size();

#pragma omp parallel for
  for (int out_idx = 0; out_idx < size; out_idx++) {
    vector<int64_t> coord(dims, 0);
    // 1. recover original coordinates
    int remain = 0;
    for (int s = 0; s < out_stride_.size(); s++) {
      if (s == 0) {
        coord[s] = static_cast<int>(out_idx / out_stride_[s]);
      } else {
        coord[s] = static_cast<int>(remain / out_stride_[s]);
      }
      remain = out_idx % out_stride_[s];
    }
    // 2. find corresponding index of src0/src1
    int src0_index = 0, src1_index = 0;
    for (int d = 0; d < dims; d++) {
      if (src0_shape_[d] != 1) {
        src0_index += coord[d] * src0_stride_[d];
      }
      if (src1_shape_[d] != 1) {
        src1_index += coord[d] * src1_stride_[d];
      }
    }

    // 3. do actual computation
    dst_data[out_idx] = pow(src0_data[src0_index], src1_data[src1_index]);
  }

  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(Pow);
}  // namespace executor
