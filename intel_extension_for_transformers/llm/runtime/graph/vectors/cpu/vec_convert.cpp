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

#include "vec_store.hpp"
#include "vec_convert.hpp"

template <int rounding>
s32x16 cvt_roundfp32x16_s32x16(fp32x16 a) {
  static_assert(rounding == (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) ||
                    rounding == (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) ||
                    rounding == (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC) ||
                    rounding == (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC) || rounding == (_MM_FROUND_CUR_DIRECTION),
                "ERROR: Not support rounding");
#if __AVX512F__
  return {_mm512_cvt_roundps_epi32(a.first, rounding)};
#else
  return {_mm256_cvtps_epi32(_mm256_round_ps(a.first, rounding)),
          _mm256_cvtps_epi32(_mm256_round_ps(a.second, rounding))};
#endif
}
template <int rounding>
s32x16 maskz_cvt_roundfp32x16_s32x16(int mask, fp32x16 a) {
  static_assert(rounding == (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) ||
                    rounding == (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) ||
                    rounding == (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC) ||
                    rounding == (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC) || rounding == (_MM_FROUND_CUR_DIRECTION),
                "ERROR: Not support rounding");
#if __AVX512F__
  return {_mm512_maskz_cvt_roundps_epi32(mask, a.first, rounding)};
#else
  __m256i first, second;
  first = _mm256_cvtps_epi32(_mm256_round_ps(a.first, rounding));
  second = _mm256_cvtps_epi32(_mm256_round_ps(a.second, rounding));
  MASK_DECORATOR(_mm256_blend_epi32, _mm256_setzero_si256(), first, mask & 255, first);
  MASK_DECORATOR(_mm256_blend_epi32, _mm256_setzero_si256(), second, mask >> 8, second);
  return {first, second};
#endif
}

bf16x16 cvt_fp32x16_bf16x16(fp32x16 a) {
#if __AVX512F__
#if __AVX512BF16__ && __GNUC__ > 11
  return {_mm512_cvtneps_pbh(a.first)};
#else
  return {_mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(a.first), 2))};
#endif
#else
  __m256i first = _mm256_bsrli_epi128(_mm256_castps_si256(a.first), 2);
  __m256i second = _mm256_bsrli_epi128(_mm256_castps_si256(a.second), 2);
  __m256i res = _mm256_packus_epi32(first, second);
  return {_mm256_permute4x64_epi64(res, 0x18)};
#endif
}

fp32x16 cvt_bf16x16_fp32x16(bf16x16 a) {
#if __AVX512F__
#if __AVX512BF16__ && __GNUC__ > 11
  return {_mm512_cvtpbh_ps(a.first)};
#else
  return {_mm512_castsi512_ps(_mm512_bslli_epi128(_mm512_cvtepu16_epi32(a.first), 2))};
#endif
#else
  __m128i second = _mm256_extractf128_si256(a.first, 1);
  __m256 second_fp32 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_cvtepu16_epi32(second), 2));
  __m128i first = _mm256_castsi256_si128(a.first);
  __m256 first_fp32 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_cvtepu16_epi32(first), 2));
  return {first_fp32, second_fp32};
#endif
}

fp32x16 maskz_cvt_bf16x16_fp32x16(int mask, bf16x16 a) {
#if __AVX512F__
#if __AVX512BF16__ && __GNUC__ > 11
  return {_mm512_maskz_cvtpbh_ps(mask, a.first)};
#else
  return {_mm512_castsi512_ps(_mm512_bslli_epi128(_mm512_maskz_cvtepu16_epi32(mask, a.first), 2))};
#endif
#else
  __m128i second = _mm256_extractf128_si256(a.first, 1);
  __m256 second_fp32 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_cvtepu16_epi32(second), 2));
  __m128i first = _mm256_castsi256_si128(a.first);
  __m256 first_fp32 = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_cvtepu16_epi32(first), 2));
  MASK_DECORATOR(_mm256_blend_ps, _mm256_setzero_ps(), first_fp32, mask & 255, first_fp32);
  MASK_DECORATOR(_mm256_blend_ps, _mm256_setzero_ps(), second_fp32, mask >> 8, second_fp32);
  return {first_fp32, second_fp32};
#endif
}

u8x16 cvt_u32x16_u8x16(u32x16 a) {
#if __AVX512F__
  return {_mm512_cvtusepi32_epi8(a.first)};
#else
  __m256i first = _mm256_min_epi32(_mm256_set1_epi32(255), a.first);
  __m256i second = _mm256_min_epi32(_mm256_set1_epi32(255), a.second);
  first = _mm256_shuffle_epi8(first, _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1,
                                                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0));
  second = _mm256_shuffle_epi8(second, _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1, -1, -1, -1));
  __m256i result = _mm256_or_si256(first, second);
  result = _mm256_permutevar8x32_epi32(result, _mm256_set_epi32(7, 6, 3, 2, 5, 1, 4, 0));
  return {_mm256_castsi256_si128(result)};
#endif
}

u8x16 maskz_cvt_u32x16_u8x16(int mask, u32x16 a) {
#if __AVX512F__
  return {_mm512_maskz_cvtusepi32_epi8(mask, a.first)};
#else
  __m256i first, second;
  MASK_DECORATOR(_mm256_blend_epi32, _mm256_setzero_si256(), _mm256_min_epi32(_mm256_set1_epi32(255), a.first),
                 mask & 255, first);
  MASK_DECORATOR(_mm256_blend_epi32, _mm256_setzero_si256(), _mm256_min_epi32(_mm256_set1_epi32(255), a.second),
                 mask >> 8, second);
  first = _mm256_shuffle_epi8(first, _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1,
                                                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0));
  second = _mm256_shuffle_epi8(second, _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1, -1, -1, -1));
  __m256i result = _mm256_or_si256(first, second);
  result = _mm256_permutevar8x32_epi32(result, _mm256_set_epi32(7, 6, 3, 2, 5, 1, 4, 0));
  return {_mm256_castsi256_si128(result)};
#endif
}

s8x16 cvt_s32x16_s8x16(s32x16 a) {
#if __AVX512F__
  return {_mm512_cvtsepi32_epi8(a.first)};
#else
  __m256i first = _mm256_min_epi32(_mm256_set1_epi32(127), a.first);
  __m256i second = _mm256_min_epi32(_mm256_set1_epi32(127), a.second);
  first = _mm256_max_epi32(_mm256_set1_epi32(-128), first);
  second = _mm256_max_epi32(_mm256_set1_epi32(-128), second);
  first = _mm256_shuffle_epi8(first, _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1,
                                                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0));
  second = _mm256_shuffle_epi8(second, _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1, -1, -1, -1));
  __m256i result = _mm256_or_si256(first, second);
  result = _mm256_permutevar8x32_epi32(result, _mm256_set_epi32(7, 6, 3, 2, 5, 1, 4, 0));
  return {_mm256_castsi256_si128(result)};
#endif
}

s8x16 maskz_cvt_s32x16_s8x16(const int mask, s32x16 a) {
#if __AVX512F__
  return {_mm512_maskz_cvtsepi32_epi8(mask, a.first)};
#else
  __m256i first, second;
  MASK_DECORATOR(_mm256_blend_epi32, _mm256_setzero_si256(), _mm256_min_epi32(_mm256_set1_epi32(127), a.first),
                 mask & 255, first);
  MASK_DECORATOR(_mm256_blend_epi32, _mm256_setzero_si256(), _mm256_min_epi32(_mm256_set1_epi32(127), a.second),
                 mask >> 8, second);
  first = _mm256_max_epi32(_mm256_set1_epi32(-128), first);
  second = _mm256_max_epi32(_mm256_set1_epi32(-128), second);
  first = _mm256_shuffle_epi8(first, _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1,
                                                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0));
  second = _mm256_shuffle_epi8(second, _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1, -1, -1, -1));
  __m256i result = _mm256_or_si256(first, second);
  result = _mm256_permutevar8x32_epi32(result, _mm256_set_epi32(7, 6, 3, 2, 5, 1, 4, 0));
  return {_mm256_castsi256_si128(result)};
#endif
}

void cvtu32x16_store_u8x16(void* base_addr, u32x16 a) {
#ifdef __AVX512F__
  _mm512_mask_cvtusepi32_storeu_epi8(base_addr, 0xffff, a.first);
#else
  store_u8x16(base_addr, cvt_u32x16_u8x16(a));
#endif
}

void mask_cvtu32x16_store_u8x16(void* base_addr, int mask, u32x16 a) {
#ifdef __AVX512F__
  _mm512_mask_cvtusepi32_storeu_epi8(base_addr, mask, a.first);
#else
  mask_store_u8x16(base_addr, mask, maskz_cvt_u32x16_u8x16(mask, a));
#endif
}
