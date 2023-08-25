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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_ATTENTION_TYPES_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_ATTENTION_TYPES_HPP_
#include <cstdint>

namespace jd {
enum attention_io {
  MERGE_SRC = 0,
  MERGE_DST = 1,
  Q_WEIGHT = 2,
  K_WEIGHT = 3,
  V_WEIGHT = 4,
  Q_BIAS = 5,
  K_BIAS = 6,
  V_BIAS = 7,
  Q_SCALES = 8,
  K_SCALES = 9,
  V_SCALES = 10,
  RESHAPE_INPUT = 11,           // bs * seq_len
  Q_K_SRC2 = 12,                // q X k output scale(to
  Q_K_SCALES = 13,              // q X k output scale(to
  QK_V_OUTPUT_ZERO_POINT = 14,  // qk X v output zero poi
  QK_V_OUTPUT_SCALES = 15       // qk X v output scale
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_ATTENTION_TYPES_HPP_
