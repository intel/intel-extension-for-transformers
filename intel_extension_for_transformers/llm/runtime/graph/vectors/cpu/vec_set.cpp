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

#include "vec_set.hpp"

inline fp32x16 set1_fp32x16(const float x) {
#if __AVX512F__
  return _mm512_set1_ps(x);
#else
  return {_mm256_set1_ps(x), _mm256_set1_ps(x)};
#endif
}

inline int32x16 set1_int8x16(const int8_t x) {
#if __AVX512F__
  return _mm512_set1_epi8(x);
#else
  return {_mm256_set1_epi8(x), _mm256_set1_epi8(x)};
#endif
}

inline int32x16 set1_int16x16(const int16_t x) {
#if __AVX512F__
  return _mm512_set1_epi16(x);
#else
  return {_mm256_set1_epi16(x), _mm256_set1_epi16(x)};
#endif
}

inline int32x16 set1_fp16x16(const uint16_t x) {
#if __AVX512F__
  return _mm512_set1_epi16(x);
#else
  return {_mm256_set1_epi16(x), _mm256_set1_epi16(x)};
#endif
}

inline int32x16 set1_int32x16(const int16_t x) {
#if __AVX512F__
  return _mm512_set1_epi32(x);
#else
  return {_mm256_set1_epi32(x), _mm256_set1_epi32(x)};
#endif
}

inline int32x16 setzero_int32x16() {
#if __AVX512F__
  return _mm512_setzero_epi32();
#else
  return {_mm256_setzero_si256(), _mm256_setzero_si256()};
#endif
}

inline fp32x16 setzero_fp32x16() {
#if __AVX512F__
  return _mm512_setzero_ps();
#else
  return {_mm256_setzero_ps(), _mm256_setzero_ps()};
#endif
}
