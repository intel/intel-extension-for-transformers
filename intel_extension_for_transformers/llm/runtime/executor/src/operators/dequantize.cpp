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

#include "dequantize.hpp"

namespace executor {

void DequantizeLinearOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  output[0]->set_dtype("fp32");
}

void DequantizeLinearOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  has_zp_ = input.size() == 3;
  size_ = input[0]->size();
  scales_size_ = input[1]->size();
  src0_stride_ = GetStrides(input[0]->shape());
  output[0]->set_shape(input[0]->shape());
  if (input.size() == 3) {
    assert(input[2]->dtype() == input[0]->dtype());
  }
}

template <typename T>
void DequantizeLinearOperator::ForwardImpl(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  auto src0_data = static_cast<const T*>(input[0]->data());
  auto src1_data = static_cast<const float*>(input[1]->data());
  const T* src2_data = has_zp_ ? static_cast<const T*>(input[2]->data()) : nullptr;
  auto dst_data = static_cast<float*>(output[0]->mutable_data());

#if __AVX512F__
  if (scales_size_ == 1) {
    int avx512_loop_len = size_ >> 4;
    __m512 _scale = _mm512_set1_ps(src1_data[0]);
    T zero_point = has_zp_ ? src2_data[0] : 0;
    __m512 _zero_point = _mm512_set1_ps(zero_point * src1_data[0]);
#pragma omp parallel for
    for (int i = 0; i < avx512_loop_len; ++i) {
      __m128i _src_data_8 = _mm_loadu_si128((const __m128i *)(src0_data + (i << 4)));
      __m512i _src_data_32 = _mm512_setzero_epi32();
      if (typeid(T) == typeid(int8_t)) {
        _src_data_32 = _mm512_cvtepi8_epi32(_src_data_8);
      } else if (typeid(T) == typeid(uint8_t)) {
        _src_data_32 = _mm512_cvtepu8_epi32(_src_data_8);
      }
      __m512 _src_data = _mm512_cvtepi32_ps(_src_data_32);
      __m512 data = _mm512_fmsub_ps(_src_data, _scale, _zero_point);
      _mm512_storeu_ps(reinterpret_cast<__m512*>(dst_data + (i << 4)), data);
    }
#pragma omp parallel for
    for (int i = (avx512_loop_len << 4); i < size_; i++) {
      dst_data[i] = (src0_data[i] - zero_point) * src1_data[0];
    }
    return;
  }
#endif

#pragma omp parallel for
  for (int out_idx = 0; out_idx < size_; out_idx++) {
    int scale_index = 0;
    if (scales_size_ != 1) {
      int remain = out_idx;
      for (int a = 0; a < axis_; a++) {
        remain = remain % src0_stride_[a];
      }
      scale_index = static_cast<int>(remain / src0_stride_[axis_]);
    }
    if (has_zp_)
      dst_data[out_idx] = (src0_data[out_idx] - src2_data[scale_index]) * src1_data[scale_index];
    else
      dst_data[out_idx] = src0_data[out_idx] * src1_data[scale_index];
  }
}

void DequantizeLinearOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (input[0]->dtype() == "u8") {
    ForwardImpl<uint8_t>(input, output);
  } else if (input[0]->dtype() == "s8") {
    ForwardImpl<int8_t>(input, output);
  }
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(DequantizeLinear);
}  // namespace executor
