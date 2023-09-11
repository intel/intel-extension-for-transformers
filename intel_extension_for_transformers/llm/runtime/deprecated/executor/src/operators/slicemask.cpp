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

#include <stdint.h>
#include "slicemask.hpp"
#include "common.hpp"

#define AVX512_BYTES 64

namespace executor {

SliceMaskOperator::SliceMaskOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("starts");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&starts_, attrs_map["starts"], ",");
  }
  iter = attrs_map.find("ends");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&ends_, attrs_map["ends"], ",");
  }
  iter = attrs_map.find("axes");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&axes_, attrs_map["axes"], ",");
  }
  iter = attrs_map.find("steps");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&steps_, attrs_map["steps"], ",");
  }
  iter = attrs_map.find("ends_with_tensor");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&ends_with_tensor_, attrs_map["ends_with_tensor"], ",");
  }
}

void SliceMaskOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  output[0]->set_dtype("fp32");
}

void SliceMaskOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  const vector<int64_t>& src_shape = input[0]->shape();
  auto input_ids_shape = input[1]->shape();
  auto past_key_shape = input[2]->shape();
  vector<int64_t> dst_shape = src_shape;
  dst_shape[2] = input_ids_shape[1];
  dst_shape[3] = past_key_shape[1] + input_ids_shape[1];
  output[0]->set_shape(dst_shape);
}

void SliceMaskOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  Tensor* src = input[0];
  Tensor* dst = output[0];
  // auto input_ids_shape = input[1]->shape();
  auto past_key_shape = input[2]->shape();
  const vector<int64_t>& src_shape = src->shape();
  const vector<int64_t>& dst_shape = dst->shape();


  std::vector<int64_t> tmp = output[0]->shape();
  if (dst_shape[3] < 128) {
      std::vector<int64_t> newshape {1024};
      output[0]->set_shape(newshape);
  }

  void* dst_data = output[0]->mutable_data();
  int64_t slice_bytes = dst_shape[3] * type2bytes[output[0]->dtype()];

  int64_t src_bytes = src_shape[3]  * type2bytes[output[0]->dtype()];


#if __AVX512F__
#pragma omp parallel for
  for (int i = past_key_shape[1]; i < dst_shape[2] + past_key_shape[1]; ++i) {
    auto dst_addr = reinterpret_cast<char*>(dst_data) + (i - past_key_shape[1]) * slice_bytes;
    const char* src_addr = reinterpret_cast<const char*>(input[0]->data()) + i * src_bytes;
    int64_t loop_size = slice_bytes & ~(AVX512_BYTES - 1);
    __mmask64 tail_mask = (1ULL << (slice_bytes - loop_size)) - 1;
    for (int j = 0; j < loop_size; j += AVX512_BYTES) {
        __m512 reg = _mm512_loadu_ps(src_addr + j);
        _mm512_storeu_ps(dst_addr + j, reg);
    }
    __m512i reg = _mm512_maskz_loadu_epi8(tail_mask, src_addr + loop_size);
    _mm512_mask_storeu_epi8(dst_addr + loop_size, tail_mask, reg);
  }
#endif

  output[0]->set_shape(tmp);
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(SliceMask);
}  // namespace executor
