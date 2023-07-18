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

#include "vec_compare.hpp"

inline fp32x16 min_fp32x16(fp32x16 a, fp32x16 b) {
#if __AVX512F__
  return _mm512_min_ps(a, b);
#else
  return {_mm256_min_ps(a.first, b.first), _mm256_min_ps(a.second, b.second)};
#endif
}

inline int32x16 max_int32x16(int32x16 a, int32x16 b) {
#if __AVX512F__
  return _mm512_max_epi32(a, b);
#else
  return {_mm256_max_epi32(a.first, b.first), _mm256_max_epi32(a.second, b.second)};
#endif
}

inline fp32x16 max_fp32x16(fp32x16 a, fp32x16 b) {
#if __AVX512F__
  return _mm512_max_ps(a, b);
#else
  return {_mm256_max_ps(a.first, b.first), _mm256_max_ps(a.second, b.second)};
#endif
}

inline float reduce_max_fp32x16(fp32x16 x) {
#if __AVX512F__
  return _mm512_reduce_max_ps(x);
#else
  const __m256 x256 = _mm256_max_ps(x.first, x.second);
  const __m128 x128 = _mm_max_ps(_mm256_extractf128_ps(x256, 1), _mm256_castps256_ps128(x256));
  const __m128 x64 = _mm_max_ps(x128, _mm_movehl_ps(x128, x128));
  const __m128 x32 = _mm_max_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
  return _mm_cvtss_f32(x32);
#endif
}
