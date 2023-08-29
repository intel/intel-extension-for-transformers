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

#include "vec_arithmetic.hpp"

inline fp32x16 sub_fp32x16(fp32x16 x, fp32x16 y) {
#if __AVX512F__
  return _mm512_sub_ps(x, y);
#else
  return {_mm256_sub_ps(x.first, y.first), _mm256_sub_ps(x.second, y.second)};
#endif
}

inline fp32x16 fmsub_fp32x16(fp32x16 x, fp32x16 y, fp32x16 z) {
#if __AVX512F__
  return _mm512_fmsub_ps(x, y, z);
#else
  return {_mm256_fmsub_ps(x.first, y.first, z.first), _mm256_fmsub_ps(x.second, y.second, z.second)};
#endif
}

inline fp32x16 maskz_fmsub_fp32x16(int mask, fp32x16 x, fp32x16 y, fp32x16 z) {
#if __AVX512F__
  return _mm512_maskz_fmsub_ps(mask, x, y, z);
#else
  __m256 first, second;
  MASK_DECORATOR(_mm256_blend_ps, _mm256_setzero_ps(), _mm256_fmsub_ps(x.first, y.first, z.first), mask & 255, first);
  MASK_DECORATOR(_mm256_blend_ps, _mm256_setzero_ps(), _mm256_fmsub_ps(x.second, y.second, z.second), mask >> 8,
                 second);
  return {first, second};
#endif
}

inline fp32x16 add_fp32x16(fp32x16 x, fp32x16 y) {
#if __AVX512F__
  return _mm512_add_ps(x, y);
#else
  return {_mm256_add_ps(x.first, y.first), _mm256_add_ps(x.second, y.second)};
#endif
}

inline fp32x16 fmadd_fp32x16(fp32x16 x, fp32x16 y, fp32x16 z) {
#if __AVX512F__
  return _mm512_fmadd_ps(x, y, z);
#else
  return {_mm256_fmadd_ps(x.first, y.first, z.first), _mm256_fmadd_ps(x.second, y.second, z.second)};
#endif
}

inline fp32x16 mul_fp32x16(fp32x16 x, fp32x16 y) {
#if __AVX512F__
  return _mm512_mul_ps(x, y);
#else
  return {_mm256_mul_ps(x.first, y.first), _mm256_mul_ps(x.second, y.second)};
#endif
}

inline fp32x16 maskz_mul_fp32x16(int mask, fp32x16 x, fp32x16 y) {
#if __AVX512F__
  return _mm512_maskz_mul_ps(mask, x, y);
#else
  __m256 first, second;
  MASK_DECORATOR(_mm256_blend_ps, _mm256_setzero_ps(), _mm256_mul_ps(x.first, y.first), mask & 255, first);
  MASK_DECORATOR(_mm256_blend_ps, _mm256_setzero_ps(), _mm256_mul_ps(x.second, y.second), mask >> 8, second);
  return {first, second};
#endif
}

template <int rounding>
inline fp32x16 mul_round_fp32x16(fp32x16 x, fp32x16 y) {
  static_assert(rounding == (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) ||
                    rounding == (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) ||
                    rounding == (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC) ||
                    rounding == (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC) || rounding == (_MM_FROUND_CUR_DIRECTION),
                "ERROR: Not support rounding");
#if __AVX512F__
  return _mm512_mul_round_ps(x, y, rounding);
#else
  return {_mm256_round_ps(_mm256_mul_ps(x.first, y.first), rounding),
          _mm256_round_ps(_mm256_mul_ps(x.second, y.second), rounding)};
#endif
}

inline fp32x16 div_fp32x16(fp32x16 x, fp32x16 y) {
#if __AVX512F__
  return _mm512_div_ps(x, y);
#else
  return {_mm256_div_ps(x.first, y.first), _mm256_div_ps(x.second, y.second)};
#endif
}

inline float reduce_add_fp32x16(fp32x16 x) {
#if __AVX512F__
  return _mm512_reduce_add_ps(x);
#else
  const __m256 x256 = _mm256_add_ps(x.first, x.second);
  const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x256, 1), _mm256_castps256_ps128(x256));
  const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
  const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
  return _mm_cvtss_f32(x32);
#endif
}

inline fp32x16 sqrt_fp32x16(fp32x16 x) {
#if __AVX512F__
  return _mm512_sqrt_ps(x);
#else
  return {_mm256_sqrt_ps(x.first), _mm256_sqrt_ps(x.second)};
#endif
}

inline fp32x16 rsqrt14_fp32x16(fp32x16 x) {
#if __AVX512F__
  return _mm512_rsqrt14_ps(x);
#else
  // the max relative error is 6x than avx512
  return {_mm256_rsqrt_ps(x.first), _mm256_rsqrt_ps(x.second)};
#endif
}
inline fp32x16 ceil_fp32x16(fp32x16 x) {
#if __AVX512F__
  return _mm512_ceil_ps(x);
#else
  // the max relative error is 6x than avx512
  return {_mm256_ceil_ps(x.first), _mm256_ceil_ps(x.second)};
#endif
}

inline fp32x16 scale_fp32x16(fp32x16 x, fp32x16 y) {
#if __AVX512F__
  return _mm512_scalef_ps(x, y);
#else
  // No intrinsic
  assert("No intrinsic");
  return {_mm256_rsqrt_ps(x.first), _mm256_rsqrt_ps(x.second)};
#endif
}

inline float dot_fp32x16(fp32x16 x, fp32x16 y) { return reduce_add_fp32x16(mul_fp32x16(x, y)); }

inline fp32x16 abs_fp32x16(fp32x16 x) {
#if __AVX512F__
  return _mm512_abs_ps(x);
#else
  return {_mm256_castsi256_ps(_mm256_abs_epi32(_mm256_castps_si256(x.first))),
          _mm256_castsi256_ps(_mm256_abs_epi32(_mm256_castps_si256(x.second)))};
#endif
}
