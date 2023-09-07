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

#include <cstring>
#include "data_type/fp16.hpp"

namespace {
template <typename T2, typename T1>
inline const T2 bit_cast(T1 i) {
  static_assert(sizeof(T1) == sizeof(T2), "Bit-casting must preserve size.");
  T2 o;
  memcpy(&o, &i, sizeof(T2));
  return o;
}
}  // namespace

namespace jd {
float16_t::float16_t() { data = 0; }
float16_t::float16_t(uint16_t val) { data = val; }
float16_t::float16_t(float val) { (*this) = val; }

float16_t& float16_t::operator=(float val) {
  // round-to-nearest-even: add last bit after truncated mantissa
  const uint32_t b = bit_cast<uint32_t>(val) + 0x00001000;
  const uint32_t e = (b & 0x7F800000) >> 23;  // exponent
  // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
  const uint32_t m = b & 0x007FFFFF;
  // sign : normalized : denormalized : saturate

  this->data = (b & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
               ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) | (e > 143) * 0x7FFF;
  return *this;
}
float16_t::operator float() const {
  // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15,
  // +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
  const uint32_t e = (data & 0x7C00) >> 10;  // exponent
  const uint32_t m = (data & 0x03FF) << 13;  // mantissa
  // evil log2 bit hack to count leading zeros in denormalized format
  const uint32_t v = bit_cast<uint32_t>(static_cast<float>(m)) >> 23;
  // sign : normalized : denormalized
  return bit_cast<float>((data & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
                         ((e == 0) & (m != 0)) * ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000)));
}
}  // namespace jd
