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

#include "memory.h"

void ne_attention_padding_mask_f32_forward(const int bs, const int nr_qk, const int qlen, const int ith, const int nth,
                                           const void* padding, const float p_value, struct ne_tensor* dst) {
  // mask padding token (padding left)
  for (int b = 0; b < bs; b++) {
    const int n_padding = (reinterpret_cast<const int32_t*>(padding))[b];
    if (n_padding == 0) continue;
    for (int k = 0; k < (nr_qk / bs); k++) {
      for (int j = ith; j < qlen; j += nth) {
        // it will not affect next token if don't mask the pad_token row
        ne_vec_set_f32(n_padding,
                       reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + b * dst->nb[3] + k * dst->nb[2] +
                                                j * dst->nb[1]),
                       p_value);
      }
    }
  }
}
