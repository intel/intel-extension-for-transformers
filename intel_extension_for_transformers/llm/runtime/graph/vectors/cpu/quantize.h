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
#pragma once

#include <assert.h>
#if defined(_MSC_VER) || defined(__MINGW32__)
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#include "core/data_types.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
//
// quantization
//

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
// multiply int8_t, add results pairwise twice
static inline __m128i mul_sum_i8_pairs(const __m128i x, const __m128i y) {
  // Get absolute values of x vectors
  const __m128i ax = _mm_sign_epi8(x, x);
  // Sign the values of the y vectors
  const __m128i sy = _mm_sign_epi8(y, x);
  // Perform multiplication and create 16-bit values
  const __m128i dot = _mm_maddubs_epi16(ax, sy);
  const __m128i ones = _mm_set1_epi16(1);
  return _mm_madd_epi16(ones, dot);
}

#if __AVX__ || __AVX2__ || __AVX512F__
// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
  __m128 res = _mm256_extractf128_ps(x, 1);
  res = _mm_add_ps(res, _mm256_castps256_ps128(x));
  res = _mm_add_ps(res, _mm_movehl_ps(res, res));
  res = _mm_add_ss(res, _mm_movehdup_ps(res));
  return _mm_cvtss_f32(res);
}

// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {
  const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
  const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
  const __m128i sum64 = _mm_add_epi32(hi64, sum128);
  const __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
  return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// horizontally add 4 int32_t
static inline int hsum_i32_4(const __m128i a) {
  const __m128i hi64 = _mm_unpackhi_epi64(a, a);
  const __m128i sum64 = _mm_add_epi32(hi64, a);
  const __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
  return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

#if __AVX2__ || __AVX512F__
// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t* x) {
  uint32_t x32;
  memcpy(&x32, x, sizeof(uint32_t));
  const __m256i shuf_mask =
      _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
  __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf_mask);
  const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
  bytes = _mm256_or_si256(bytes, bit_mask);
  return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t* rsi) {
  const __m128i tmp = _mm_loadu_si128((const __m128i*)rsi);
  const __m256i bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
  const __m256i lowMask = _mm256_set1_epi8(0xF);
  return _mm256_and_si256(lowMask, bytes);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m256i x) {
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
  return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if __AVXVNNI__
  const __m256i zero = _mm256_setzero_si256();
  const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
  return _mm256_cvtepi32_ps(summed_pairs);
#else
  // Perform multiplication and create 16-bit values
  const __m256i dot = _mm256_maddubs_epi16(ax, sy);
  return sum_i16_pairs_float(dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
  const __m256i zero = _mm256_setzero_si256();
  const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
  return _mm256_cvtepi32_ps(summed_pairs);
#else
  // Get absolute values of x vectors
  const __m256i ax = _mm256_sign_epi8(x, x);
  // Sign the values of the y vectors
  const __m256i sy = _mm256_sign_epi8(y, x);
  return mul_sum_us8_pairs_float(ax, sy);
#endif
}

static inline __m128i packNibbles(__m256i bytes) {
  // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
#if __AVX512F__
  const __m256i bytes_srli_4 = _mm256_srli_epi16(bytes, 4);  // 0000_0000_abcd_0000
  bytes = _mm256_or_si256(bytes, bytes_srli_4);              // 0000_abcd_abcd_efgh
  return _mm256_cvtepi16_epi8(bytes);                        // abcd_efgh
#else
  const __m256i lowByte = _mm256_set1_epi16(0xFF);
  __m256i high = _mm256_andnot_si256(lowByte, bytes);
  __m256i low = _mm256_and_si256(lowByte, bytes);
  high = _mm256_srli_epi16(high, 4);
  bytes = _mm256_or_si256(low, high);

  // Compress uint16_t lanes into bytes
  __m128i r0 = _mm256_castsi256_si128(bytes);
  __m128i r1 = _mm256_extracti128_si256(bytes, 1);
  return _mm_packus_epi16(r0, r1);
#endif
}
#elif defined(__AVX__)
// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t* x) {
  uint32_t x32;
  memcpy(&x32, x, sizeof(uint32_t));
  const __m128i shuf_maskl = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
  const __m128i shuf_maskh = _mm_set_epi64x(0x0303030303030303, 0x0202020202020202);
  __m128i bytesl = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskl);
  __m128i bytesh = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskh);
  const __m128i bit_mask = _mm_set1_epi64x(0x7fbfdfeff7fbfdfe);
  bytesl = _mm_or_si128(bytesl, bit_mask);
  bytesh = _mm_or_si128(bytesh, bit_mask);
  bytesl = _mm_cmpeq_epi8(bytesl, _mm_set1_epi64x(-1));
  bytesh = _mm_cmpeq_epi8(bytesh, _mm_set1_epi64x(-1));
  return _mm256_set_m128i(bytesh, bytesl);
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t* rsi) {
  // Load 16 bytes from memory
  __m128i tmpl = _mm_loadu_si128((const __m128i*)rsi);
  __m128i tmph = _mm_srli_epi16(tmpl, 4);
  const __m128i lowMask = _mm_set1_epi8(0xF);
  tmpl = _mm_and_si128(lowMask, tmpl);
  tmph = _mm_and_si128(lowMask, tmph);
  return _mm256_set_m128i(tmph, tmpl);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m128i xh, const __m128i xl) {
  const __m128i ones = _mm_set1_epi16(1);
  const __m128i summed_pairsl = _mm_madd_epi16(ones, xl);
  const __m128i summed_pairsh = _mm_madd_epi16(ones, xh);
  const __m256i summed_pairs = _mm256_set_m128i(summed_pairsh, summed_pairsl);
  return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
  const __m128i axl = _mm256_castsi256_si128(ax);
  const __m128i axh = _mm256_extractf128_si256(ax, 1);
  const __m128i syl = _mm256_castsi256_si128(sy);
  const __m128i syh = _mm256_extractf128_si256(sy, 1);
  // Perform multiplication and create 16-bit values
  const __m128i dotl = _mm_maddubs_epi16(axl, syl);
  const __m128i doth = _mm_maddubs_epi16(axh, syh);
  return sum_i16_pairs_float(doth, dotl);
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
  const __m128i xl = _mm256_castsi256_si128(x);
  const __m128i xh = _mm256_extractf128_si256(x, 1);
  const __m128i yl = _mm256_castsi256_si128(y);
  const __m128i yh = _mm256_extractf128_si256(y, 1);
  // Get absolute values of x vectors
  const __m128i axl = _mm_sign_epi8(xl, xl);
  const __m128i axh = _mm_sign_epi8(xh, xh);
  // Sign the values of the y vectors
  const __m128i syl = _mm_sign_epi8(yl, xl);
  const __m128i syh = _mm_sign_epi8(yh, xh);
  // Perform multiplication and create 16-bit values
  const __m128i dotl = _mm_maddubs_epi16(axl, syl);
  const __m128i doth = _mm_maddubs_epi16(axh, syh);
  return sum_i16_pairs_float(doth, dotl);
}

static inline __m128i packNibbles(__m128i bytes1, __m128i bytes2) {
  // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
  const __m128i lowByte = _mm_set1_epi16(0xFF);
  __m128i high = _mm_andnot_si128(lowByte, bytes1);
  __m128i low = _mm_and_si128(lowByte, bytes1);
  high = _mm_srli_epi16(high, 4);
  bytes1 = _mm_or_si128(low, high);
  high = _mm_andnot_si128(lowByte, bytes2);
  low = _mm_and_si128(lowByte, bytes2);
  high = _mm_srli_epi16(high, 4);
  bytes2 = _mm_or_si128(low, high);

  return _mm_packus_epi16(bytes1, bytes2);
}
#endif
#elif defined(__SSSE3__)
// horizontally add 4x4 floats
static inline float hsum_float_4x4(const __m128 a, const __m128 b, const __m128 c, const __m128 d) {
  __m128 res_0 = _mm_hadd_ps(a, b);
  __m128 res_1 = _mm_hadd_ps(c, d);
  __m128 res = _mm_hadd_ps(res_0, res_1);
  res = _mm_hadd_ps(res, res);
  res = _mm_hadd_ps(res, res);

  return _mm_cvtss_f32(res);
}
#endif  // __AVX__ || __AVX2__ || __AVX512F__
#endif  // defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)

// reference implementation for deterministic creation of model files
static void quantize_row_q4_0_reference(const float* x, block_q4_0* y, int k) {
  static const int qk = QK4_0;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    float amax = 0.0f;  // absolute max
    float max = 0.0f;

    for (int j = 0; j < qk; j++) {
      const float v = x[i * qk + j];
      if (amax < fabsf(v)) {
        amax = fabsf(v);
        max = v;
      }
    }

    const float d = max / -8;
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = NE_FP32_TO_FP16(d);

    for (int j = 0; j < qk / 2; ++j) {
      const float x0 = x[i * qk + 0 + j] * id;
      const float x1 = x[i * qk + qk / 2 + j] * id;

      const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
      const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

      y[i].qs[j] = xi0;
      y[i].qs[j] |= xi1 << 4;
    }
  }
}

static void quantize_row_q4_0(const float* x, void* y, int k) { quantize_row_q4_0_reference(x, (block_q4_0*)y, k); }

static void quantize_row_q4_1_reference(const float* x, block_q4_1* y, int k) {
  const int qk = QK4_1;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    float min = FLT_MAX;
    float max = -FLT_MAX;

    for (int j = 0; j < qk; j++) {
      const float v = x[i * qk + j];

      if (v < min) min = v;
      if (v > max) max = v;
    }

    const float d = (max - min) / ((1 << 4) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = NE_FP32_TO_FP16(d);
    y[i].m = NE_FP32_TO_FP16(min);

    for (int j = 0; j < qk / 2; ++j) {
      const float x0 = (x[i * qk + 0 + j] - min) * id;
      const float x1 = (x[i * qk + qk / 2 + j] - min) * id;

      const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
      const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

      y[i].qs[j] = xi0;
      y[i].qs[j] |= xi1 << 4;
    }
  }
}

static void quantize_row_q4_1(const float* x, void* y, int k) { quantize_row_q4_1_reference(x, (block_q4_1*)y, k); }

static void quantize_row_q5_0_reference(const float* x, block_q5_0* y, int k) {
  static const int qk = QK5_0;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    float amax = 0.0f;  // absolute max
    float max = 0.0f;

    for (int j = 0; j < qk; j++) {
      const float v = x[i * qk + j];
      if (amax < fabsf(v)) {
        amax = fabsf(v);
        max = v;
      }
    }

    const float d = max / -16;
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = NE_FP32_TO_FP16(d);

    uint32_t qh = 0;

    for (int j = 0; j < qk / 2; ++j) {
      const float x0 = x[i * qk + 0 + j] * id;
      const float x1 = x[i * qk + qk / 2 + j] * id;

      const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
      const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

      y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

      // get the 5-th bit and store it in qh at the right position
      qh |= ((xi0 & 0x10) >> 4) << (j + 0);
      qh |= ((xi1 & 0x10) >> 4) << (j + qk / 2);
    }

    memcpy(&y[i].qh, &qh, sizeof(qh));
  }
}

static void quantize_row_q5_0(const float* x, void* y, int k) { quantize_row_q5_0_reference(x, (block_q5_0*)y, k); }

static void quantize_row_q5_1_reference(const float* x, block_q5_1* y, int k) {
  const int qk = QK5_1;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    float min = FLT_MAX;
    float max = -FLT_MAX;

    for (int j = 0; j < qk; j++) {
      const float v = x[i * qk + j];

      if (v < min) min = v;
      if (v > max) max = v;
    }

    const float d = (max - min) / ((1 << 5) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = NE_FP32_TO_FP16(d);
    y[i].m = NE_FP32_TO_FP16(min);

    uint32_t qh = 0;

    for (int j = 0; j < qk / 2; ++j) {
      const float x0 = (x[i * qk + 0 + j] - min) * id;
      const float x1 = (x[i * qk + qk / 2 + j] - min) * id;

      const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
      const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

      y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

      // get the 5-th bit and store it in qh at the right position
      qh |= ((xi0 & 0x10) >> 4) << (j + 0);
      qh |= ((xi1 & 0x10) >> 4) << (j + qk / 2);
    }

    memcpy(&y[i].qh, &qh, sizeof(y[i].qh));
  }
}

static void quantize_row_q5_1(const float* x, void* y, int k) { quantize_row_q5_1_reference(x, (block_q5_1*)y, k); }

// reference implementation for deterministic creation of model files
static void quantize_row_q8_0_reference(const float* x, block_q8_0* y, int k) {
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  for (int i = 0; i < nb; i++) {
    float amax = 0.0f;  // absolute max

    for (int j = 0; j < QK8_0; j++) {
      const float v = x[i * QK8_0 + j];
      amax = MAX(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = NE_FP32_TO_FP16(d);

    for (int j = 0; j < QK8_0; ++j) {
      const float x0 = x[i * QK8_0 + j] * id;

      y[i].qs[j] = roundf(x0);
    }
  }
}

static void quantize_row_q8_0(const float* x, void* vy, int k) {
  assert(QK8_0 == 32);
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  block_q8_0* y = (block_q8_0*)vy;

#if defined(__AVX2__) || defined(__AVX__)
  for (int i = 0; i < nb; i++) {
    // Load elements into 4 AVX vectors
    __m256 v0 = _mm256_loadu_ps(x);
    __m256 v1 = _mm256_loadu_ps(x + 8);
    __m256 v2 = _mm256_loadu_ps(x + 16);
    __m256 v3 = _mm256_loadu_ps(x + 24);
    x += 32;

    // Compute max(abs(e)) for the block
    const __m256 signBit = _mm256_set1_ps(-0.0f);
    __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
    maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
    maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
    maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

    __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
    max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
    const float maxScalar = _mm_cvtss_f32(max4);

    // Quantize these floats
    const float d = maxScalar / 127.f;
    y[i].d = NE_FP32_TO_FP16(d);
    const float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
    const __m256 mul = _mm256_set1_ps(id);

    // Apply the multiplier
    v0 = _mm256_mul_ps(v0, mul);
    v1 = _mm256_mul_ps(v1, mul);
    v2 = _mm256_mul_ps(v2, mul);
    v3 = _mm256_mul_ps(v3, mul);

    // Round to nearest integer
    v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
    v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
    v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
    v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

    // Convert floats to integers
    __m256i i0 = _mm256_cvtps_epi32(v0);
    __m256i i1 = _mm256_cvtps_epi32(v1);
    __m256i i2 = _mm256_cvtps_epi32(v2);
    __m256i i3 = _mm256_cvtps_epi32(v3);

#if defined(__AVX2__)
    // Convert int32 to int16
    i0 = _mm256_packs_epi32(i0, i1);  // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
    i2 = _mm256_packs_epi32(i2, i3);  // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                      // Convert int16 to int8
    i0 = _mm256_packs_epi16(i0, i2);  // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12,
                                      // 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

    // We got our precious signed bytes, but the order is now wrong
    // These AVX2 pack instructions process 16-byte pieces independently
    // The following instruction is fixing the order
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    i0 = _mm256_permutevar8x32_epi32(i0, perm);

    _mm256_storeu_si256((__m256i*)y[i].qs, i0);
#else
    // Since we don't have in AVX some necessary functions,
    // we split the registers in half and call AVX2 analogs from SSE
    __m128i ni0 = _mm256_castsi256_si128(i0);
    __m128i ni1 = _mm256_extractf128_si256(i0, 1);
    __m128i ni2 = _mm256_castsi256_si128(i1);
    __m128i ni3 = _mm256_extractf128_si256(i1, 1);
    __m128i ni4 = _mm256_castsi256_si128(i2);
    __m128i ni5 = _mm256_extractf128_si256(i2, 1);
    __m128i ni6 = _mm256_castsi256_si128(i3);
    __m128i ni7 = _mm256_extractf128_si256(i3, 1);

    // Convert int32 to int16
    ni0 = _mm_packs_epi32(ni0, ni1);
    ni2 = _mm_packs_epi32(ni2, ni3);
    ni4 = _mm_packs_epi32(ni4, ni5);
    ni6 = _mm_packs_epi32(ni6, ni7);
    // Convert int16 to int8
    ni0 = _mm_packs_epi16(ni0, ni2);
    ni4 = _mm_packs_epi16(ni4, ni6);

    _mm_storeu_si128((__m128i*)(y[i].qs + 0), ni0);
    _mm_storeu_si128((__m128i*)(y[i].qs + 16), ni4);
#endif
  }
#else
  // scalar
  quantize_row_q8_0_reference(x, y, k);
#endif
}

// reference implementation for deterministic creation of model files
static void quantize_row_q8_1_reference(const float* x, block_q8_1* y, int k) {
  assert(QK8_1 == 32);
  assert(k % QK8_1 == 0);
  const int nb = k / QK8_1;

  for (int i = 0; i < nb; i++) {
    float amax = 0.0f;  // absolute max

    for (int j = 0; j < QK8_1; j++) {
      const float v = x[i * QK8_1 + j];
      amax = MAX(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = d;

    int sum = 0;

    for (int j = 0; j < QK8_1 / 2; ++j) {
      const float v0 = x[i * QK8_1 + j] * id;
      const float v1 = x[i * QK8_1 + QK8_1 / 2 + j] * id;

      y[i].qs[j] = roundf(v0);
      y[i].qs[QK8_1 / 2 + j] = roundf(v1);

      sum += y[i].qs[j];
      sum += y[i].qs[QK8_1 / 2 + j];
    }

    y[i].s = sum * d;
  }
}

static void quantize_row_q8_1(const float* x, void* vy, int k) {
  assert(k % QK8_1 == 0);
  const int nb = k / QK8_1;

  block_q8_1* y = (block_q8_1*)vy;

#if defined(__AVX2__) || defined(__AVX__)
  for (int i = 0; i < nb; i++) {
    // Load elements into 4 AVX vectors
    __m256 v0 = _mm256_loadu_ps(x);
    __m256 v1 = _mm256_loadu_ps(x + 8);
    __m256 v2 = _mm256_loadu_ps(x + 16);
    __m256 v3 = _mm256_loadu_ps(x + 24);
    x += 32;

    // Compute max(abs(e)) for the block
    const __m256 signBit = _mm256_set1_ps(-0.0f);
    __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
    maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
    maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
    maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

    __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
    max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
    const float maxScalar = _mm_cvtss_f32(max4);

    // Quantize these floats
    const float d = maxScalar / 127.f;
    y[i].d = d;
    const float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
    const __m256 mul = _mm256_set1_ps(id);

    // Apply the multiplier
    v0 = _mm256_mul_ps(v0, mul);
    v1 = _mm256_mul_ps(v1, mul);
    v2 = _mm256_mul_ps(v2, mul);
    v3 = _mm256_mul_ps(v3, mul);

    // Round to nearest integer
    v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
    v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
    v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
    v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

    // Convert floats to integers
    __m256i i0 = _mm256_cvtps_epi32(v0);
    __m256i i1 = _mm256_cvtps_epi32(v1);
    __m256i i2 = _mm256_cvtps_epi32(v2);
    __m256i i3 = _mm256_cvtps_epi32(v3);

#if defined(__AVX2__)
    // Compute the sum of the quants and set y[i].s
    y[i].s = d * hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));

    // Convert int32 to int16
    i0 = _mm256_packs_epi32(i0, i1);  // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
    i2 = _mm256_packs_epi32(i2, i3);  // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                      // Convert int16 to int8
    i0 = _mm256_packs_epi16(i0, i2);  // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12,
                                      // 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

    // We got our precious signed bytes, but the order is now wrong
    // These AVX2 pack instructions process 16-byte pieces independently
    // The following instruction is fixing the order
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    i0 = _mm256_permutevar8x32_epi32(i0, perm);

    _mm256_storeu_si256((__m256i*)y[i].qs, i0);
#else
    // Since we don't have in AVX some necessary functions,
    // we split the registers in half and call AVX2 analogs from SSE
    __m128i ni0 = _mm256_castsi256_si128(i0);
    __m128i ni1 = _mm256_extractf128_si256(i0, 1);
    __m128i ni2 = _mm256_castsi256_si128(i1);
    __m128i ni3 = _mm256_extractf128_si256(i1, 1);
    __m128i ni4 = _mm256_castsi256_si128(i2);
    __m128i ni5 = _mm256_extractf128_si256(i2, 1);
    __m128i ni6 = _mm256_castsi256_si128(i3);
    __m128i ni7 = _mm256_extractf128_si256(i3, 1);

    // Compute the sum of the quants and set y[i].s
    const __m128i s0 = _mm_add_epi32(_mm_add_epi32(ni0, ni1), _mm_add_epi32(ni2, ni3));
    const __m128i s1 = _mm_add_epi32(_mm_add_epi32(ni4, ni5), _mm_add_epi32(ni6, ni7));
    y[i].s = d * hsum_i32_4(_mm_add_epi32(s0, s1));

    // Convert int32 to int16
    ni0 = _mm_packs_epi32(ni0, ni1);
    ni2 = _mm_packs_epi32(ni2, ni3);
    ni4 = _mm_packs_epi32(ni4, ni5);
    ni6 = _mm_packs_epi32(ni6, ni7);
    // Convert int16 to int8
    ni0 = _mm_packs_epi16(ni0, ni2);
    ni4 = _mm_packs_epi16(ni4, ni6);

    _mm_storeu_si128((__m128i*)(y[i].qs + 0), ni0);
    _mm_storeu_si128((__m128i*)(y[i].qs + 16), ni4);
#endif
  }
#else
  // scalar
  quantize_row_q8_1_reference(x, y, k);
#endif
}

static void dequantize_row_q4_0(const block_q4_0* x, float* y, int k) {
  static const int qk = QK4_0;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    const float d = NE_FP16_TO_FP32(x[i].d);

    for (int j = 0; j < qk / 2; ++j) {
      const int x0 = (x[i].qs[j] & 0x0F) - 8;
      const int x1 = (x[i].qs[j] >> 4) - 8;

      y[i * qk + j + 0] = x0 * d;
      y[i * qk + j + qk / 2] = x1 * d;
    }
  }
}

static void dequantize_row_q4_1(const block_q4_1* x, float* y, int k) {
  static const int qk = QK4_1;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    const float d = NE_FP16_TO_FP32(x[i].d);
    const float m = NE_FP16_TO_FP32(x[i].m);

    for (int j = 0; j < qk / 2; ++j) {
      const int x0 = (x[i].qs[j] & 0x0F);
      const int x1 = (x[i].qs[j] >> 4);

      y[i * qk + j + 0] = x0 * d + m;
      y[i * qk + j + qk / 2] = x1 * d + m;
    }
  }
}

static void dequantize_row_q5_0(const block_q5_0* x, float* y, int k) {
  static const int qk = QK5_0;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    const float d = NE_FP16_TO_FP32(x[i].d);

    uint32_t qh;
    memcpy(&qh, x[i].qh, sizeof(qh));

    for (int j = 0; j < qk / 2; ++j) {
      const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
      const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;

      const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
      const int32_t x1 = ((x[i].qs[j] >> 4) | xh_1) - 16;

      y[i * qk + j + 0] = x0 * d;
      y[i * qk + j + qk / 2] = x1 * d;
    }
  }
}

static void dequantize_row_q5_1(const block_q5_1* x, float* y, int k) {
  static const int qk = QK5_1;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    const float d = NE_FP16_TO_FP32(x[i].d);
    const float m = NE_FP16_TO_FP32(x[i].m);

    uint32_t qh;
    memcpy(&qh, x[i].qh, sizeof(qh));

    for (int j = 0; j < qk / 2; ++j) {
      const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
      const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;

      const int x0 = (x[i].qs[j] & 0x0F) | xh_0;
      const int x1 = (x[i].qs[j] >> 4) | xh_1;

      y[i * qk + j + 0] = x0 * d + m;
      y[i * qk + j + qk / 2] = x1 * d + m;
    }
  }
}

static void dequantize_row_q8_0(const void* vx, float* y, int k) {
  static const int qk = QK8_0;

  assert(k % qk == 0);

  const int nb = k / qk;

  const block_q8_0* x = (const block_q8_0*)vx;

  for (int i = 0; i < nb; i++) {
    const float d = NE_FP16_TO_FP32(x[i].d);

    for (int j = 0; j < qk; ++j) {
      y[i * qk + j] = x[i].qs[j] * d;
    }
  }
}