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

#include "data_type/bf16.hpp"

namespace jd {
typedef union {
  float f;
  unsigned int u;
  uint16_t b[2];
} union_b;

bfloat16_t::operator float() const {
  union_b tmp;
  tmp.u = 0;
  tmp.b[1] = data;
  return tmp.f;
}

bfloat16_t::bfloat16_t() { data = 0; }

bfloat16_t::bfloat16_t(int32_t val) { (*this) = static_cast<float>(val); }
bfloat16_t::bfloat16_t(uint16_t val) : data(val) {}
bfloat16_t::bfloat16_t(float val) { (*this) = val; }

bfloat16_t& bfloat16_t::operator=(float val) {
  union_b tmp;
  tmp.f = val;
  // See document of VCVTNEPS2BF16 in Intel® 64 and IA-32 Architectures Software Developer’s Manual Volume 2
  const auto lsb = tmp.b[1] & 1;
  tmp.u += 0x7fff + lsb;
  data = tmp.b[1];
  return *this;
}
}  // namespace jd
