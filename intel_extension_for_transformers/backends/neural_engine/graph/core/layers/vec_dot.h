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

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h>  // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>

#ifndef __STDC_VERSION__
#define restrict
#elif __STDC_VERSION__ < 199901L
#define restrict
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#include "vectors/cpu/simd.h"
#include "core/data_types.h"
#include "vectors/cpu/quantize.h"

#define NE_VEC_DOT_UNROLL 2

#ifdef __cplusplus
extern "C" {
#endif

static void ne_vec_dot_f32(const int n, float* restrict s, const float* restrict x, const float* restrict y) {
#ifdef NE_SIMD
  float sumf = 0.0f;
  const int np = (n & ~(NE_F32_STEP - 1));

  NE_F32_VEC sum[NE_F32_ARR] = {NE_F32_VEC_ZERO};

  NE_F32_VEC ax[NE_F32_ARR];
  NE_F32_VEC ay[NE_F32_ARR];

  for (int i = 0; i < np; i += NE_F32_STEP) {
    for (int j = 0; j < NE_F32_ARR; j++) {
      ax[j] = NE_F32_VEC_LOAD(x + i + j * NE_F32_EPR);
      ay[j] = NE_F32_VEC_LOAD(y + i + j * NE_F32_EPR);

      sum[j] = NE_F32_VEC_FMA(sum[j], ax[j], ay[j]);
    }
  }

  // reduce sum0..sum3 to sum0
  NE_F32_VEC_REDUCE(sumf, sum);

  // leftovers
  for (int i = np; i < n; ++i) {
    sumf += x[i] * y[i];
  }
#else
  // scalar
  ne_float sumf = 0.0;
  for (int i = 0; i < n; ++i) {
    sumf += (ne_float)(x[i] * y[i]);
  }
#endif

  *s = sumf;
}

static void ne_vec_dot_f16(const int n, float* restrict s, ne_fp16_t* restrict x, ne_fp16_t* restrict y) {
  ne_float sumf = 0.0;

#if defined(NE_SIMD)
  const int np = (n & ~(NE_F16_STEP - 1));

  NE_F16_VEC sum[NE_F16_ARR] = {NE_F16_VEC_ZERO};

  NE_F16_VEC ax[NE_F16_ARR];
  NE_F16_VEC ay[NE_F16_ARR];

  for (int i = 0; i < np; i += NE_F16_STEP) {
    for (int j = 0; j < NE_F16_ARR; j++) {
      ax[j] = NE_F16_VEC_LOAD(x + i + j * NE_F16_EPR, j);
      ay[j] = NE_F16_VEC_LOAD(y + i + j * NE_F16_EPR, j);

      sum[j] = NE_F16_VEC_FMA(sum[j], ax[j], ay[j]);
    }
  }

  // reduce sum0..sum3 to sum0
  NE_F16_VEC_REDUCE(sumf, sum);

  // leftovers
  for (int i = np; i < n; ++i) {
    sumf += (ne_float)(NE_FP16_TO_FP32(x[i]) * NE_FP16_TO_FP32(y[i]));
  }
#else
  for (int i = 0; i < n; ++i) {
    sumf += (ne_float)(NE_FP16_TO_FP32(x[i]) * NE_FP16_TO_FP32(y[i]));
  }
#endif

  *s = sumf;
}

static void ne_vec_dot_q4_0_q8_0(const int n, float* restrict s, const void* restrict vx, const void* restrict vy) {
  const int qk = QK8_0;
  const int nb = n / qk;

  assert(n % qk == 0);
  assert(nb % 2 == 0);

  const block_q4_0* restrict x = (const block_q4_0*)vx;
  const block_q8_0* restrict y = (const block_q8_0*)vy;

#if defined(__AVX2__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();

  // Main loop
  for (int i = 0; i < nb; ++i) {
    /* Compute combined scale for the block */
    const __m256 d = _mm256_set1_ps(NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d));

    __m256i bx = bytes_from_nibbles_32(x[i].qs);

    // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
    const __m256i off = _mm256_set1_epi8(8);
    bx = _mm256_sub_epi8(bx, off);

    __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);

    const __m256 q = mul_sum_i8_pairs_float(bx, by);

    /* Multiply q with scale and accumulate */
    acc = _mm256_fmadd_ps(d, q, acc);
  }

  *s = hsum_float_8(acc);
#elif defined(__AVX__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();

  // Main loop
  for (int i = 0; i < nb; ++i) {
    // Compute combined scale for the block
    const __m256 d = _mm256_set1_ps(NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d));

    const __m128i lowMask = _mm_set1_epi8(0xF);
    const __m128i off = _mm_set1_epi8(8);

    const __m128i tmp = _mm_loadu_si128((const __m128i*)x[i].qs);

    __m128i bx = _mm_and_si128(lowMask, tmp);
    __m128i by = _mm_loadu_si128((const __m128i*)y[i].qs);
    bx = _mm_sub_epi8(bx, off);
    const __m128i i32_0 = mul_sum_i8_pairs(bx, by);

    bx = _mm_and_si128(lowMask, _mm_srli_epi64(tmp, 4));
    by = _mm_loadu_si128((const __m128i*)(y[i].qs + 16));
    bx = _mm_sub_epi8(bx, off);
    const __m128i i32_1 = mul_sum_i8_pairs(bx, by);

    // Convert int32_t to float
    __m256 p = _mm256_cvtepi32_ps(_mm256_set_m128i(i32_0, i32_1));

    // Apply the scale, and accumulate
    acc = _mm256_add_ps(_mm256_mul_ps(d, p), acc);
  }

  *s = hsum_float_8(acc);
#elif defined(__SSSE3__)
  // set constants
  const __m128i lowMask = _mm_set1_epi8(0xF);
  const __m128i off = _mm_set1_epi8(8);

  // Initialize accumulator with zeros
  __m128 acc_0 = _mm_setzero_ps();
  __m128 acc_1 = _mm_setzero_ps();
  __m128 acc_2 = _mm_setzero_ps();
  __m128 acc_3 = _mm_setzero_ps();

  // First round without accumulation
  {
    _mm_prefetch(&x[0] + sizeof(block_q4_0), _MM_HINT_T0);
    _mm_prefetch(&y[0] + sizeof(block_q8_0), _MM_HINT_T0);

    // Compute combined scale for the block 0 and 1
    const __m128 d_0_1 = _mm_set1_ps(NE_FP16_TO_FP32(x[0].d) * NE_FP16_TO_FP32(y[0].d));

    const __m128i tmp_0_1 = _mm_loadu_si128((const __m128i*)x[0].qs);

    __m128i bx_0 = _mm_and_si128(lowMask, tmp_0_1);
    __m128i by_0 = _mm_loadu_si128((const __m128i*)y[0].qs);
    bx_0 = _mm_sub_epi8(bx_0, off);
    const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

    __m128i bx_1 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_0_1, 4));
    __m128i by_1 = _mm_loadu_si128((const __m128i*)(y[0].qs + 16));
    bx_1 = _mm_sub_epi8(bx_1, off);
    const __m128i i32_1 = mul_sum_i8_pairs(bx_1, by_1);

    _mm_prefetch(&x[1] + sizeof(block_q4_0), _MM_HINT_T0);
    _mm_prefetch(&y[1] + sizeof(block_q8_0), _MM_HINT_T0);

    // Compute combined scale for the block 2 and 3
    const __m128 d_2_3 = _mm_set1_ps(NE_FP16_TO_FP32(x[1].d) * NE_FP16_TO_FP32(y[1].d));

    const __m128i tmp_2_3 = _mm_loadu_si128((const __m128i*)x[1].qs);

    __m128i bx_2 = _mm_and_si128(lowMask, tmp_2_3);
    __m128i by_2 = _mm_loadu_si128((const __m128i*)y[1].qs);
    bx_2 = _mm_sub_epi8(bx_2, off);
    const __m128i i32_2 = mul_sum_i8_pairs(bx_2, by_2);

    __m128i bx_3 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_2_3, 4));
    __m128i by_3 = _mm_loadu_si128((const __m128i*)(y[1].qs + 16));
    bx_3 = _mm_sub_epi8(bx_3, off);
    const __m128i i32_3 = mul_sum_i8_pairs(bx_3, by_3);

    // Convert int32_t to float
    __m128 p0 = _mm_cvtepi32_ps(i32_0);
    __m128 p1 = _mm_cvtepi32_ps(i32_1);
    __m128 p2 = _mm_cvtepi32_ps(i32_2);
    __m128 p3 = _mm_cvtepi32_ps(i32_3);

    // Apply the scale
    acc_0 = _mm_mul_ps(d_0_1, p0);
    acc_1 = _mm_mul_ps(d_0_1, p1);
    acc_2 = _mm_mul_ps(d_2_3, p2);
    acc_3 = _mm_mul_ps(d_2_3, p3);
  }

  // Main loop
  for (int i = 2; i < nb; i += 2) {
    _mm_prefetch(&x[i] + sizeof(block_q4_0), _MM_HINT_T0);
    _mm_prefetch(&y[i] + sizeof(block_q8_0), _MM_HINT_T0);

    // Compute combined scale for the block 0 and 1
    const __m128 d_0_1 = _mm_set1_ps(NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d));

    const __m128i tmp_0_1 = _mm_loadu_si128((const __m128i*)x[i].qs);

    __m128i bx_0 = _mm_and_si128(lowMask, tmp_0_1);
    __m128i by_0 = _mm_loadu_si128((const __m128i*)y[i].qs);
    bx_0 = _mm_sub_epi8(bx_0, off);
    const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

    __m128i bx_1 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_0_1, 4));
    __m128i by_1 = _mm_loadu_si128((const __m128i*)(y[i].qs + 16));
    bx_1 = _mm_sub_epi8(bx_1, off);
    const __m128i i32_1 = mul_sum_i8_pairs(bx_1, by_1);

    _mm_prefetch(&x[i] + 2 * sizeof(block_q4_0), _MM_HINT_T0);
    _mm_prefetch(&y[i] + 2 * sizeof(block_q8_0), _MM_HINT_T0);

    // Compute combined scale for the block 2 and 3
    const __m128 d_2_3 = _mm_set1_ps(NE_FP16_TO_FP32(x[i + 1].d) * NE_FP16_TO_FP32(y[i + 1].d));

    const __m128i tmp_2_3 = _mm_loadu_si128((const __m128i*)x[i + 1].qs);

    __m128i bx_2 = _mm_and_si128(lowMask, tmp_2_3);
    __m128i by_2 = _mm_loadu_si128((const __m128i*)y[i + 1].qs);
    bx_2 = _mm_sub_epi8(bx_2, off);
    const __m128i i32_2 = mul_sum_i8_pairs(bx_2, by_2);

    __m128i bx_3 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_2_3, 4));
    __m128i by_3 = _mm_loadu_si128((const __m128i*)(y[i + 1].qs + 16));
    bx_3 = _mm_sub_epi8(bx_3, off);
    const __m128i i32_3 = mul_sum_i8_pairs(bx_3, by_3);

    // Convert int32_t to float
    __m128 p0 = _mm_cvtepi32_ps(i32_0);
    __m128 p1 = _mm_cvtepi32_ps(i32_1);
    __m128 p2 = _mm_cvtepi32_ps(i32_2);
    __m128 p3 = _mm_cvtepi32_ps(i32_3);

    // Apply the scale
    __m128 p0_d = _mm_mul_ps(d_0_1, p0);
    __m128 p1_d = _mm_mul_ps(d_0_1, p1);
    __m128 p2_d = _mm_mul_ps(d_2_3, p2);
    __m128 p3_d = _mm_mul_ps(d_2_3, p3);

    // Acummulate
    acc_0 = _mm_add_ps(p0_d, acc_0);
    acc_1 = _mm_add_ps(p1_d, acc_1);
    acc_2 = _mm_add_ps(p2_d, acc_2);
    acc_3 = _mm_add_ps(p3_d, acc_3);
  }

  *s = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);
#else
  // scalar
  float sumf = 0.0;

  for (int i = 0; i < nb; i++) {
    int sumi = 0;

    for (int j = 0; j < qk / 2; ++j) {
      const int v0 = (x[i].qs[j] & 0x0F) - 8;
      const int v1 = (x[i].qs[j] >> 4) - 8;

      sumi += (v0 * y[i].qs[j]) + (v1 * y[i].qs[j + qk / 2]);
    }

    sumf += sumi * NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d);
  }

  *s = sumf;
#endif
}

static void ne_vec_dot_q4_1_q8_1(const int n, float* restrict s, const void* restrict vx, const void* restrict vy) {
  const int qk = QK8_1;
  const int nb = n / qk;

  assert(n % qk == 0);
  assert(nb % 2 == 0);

  const block_q4_1* restrict x = (const block_q4_1*)vx;
  const block_q8_1* restrict y = (const block_q8_1*)vy;

  // TODO: add WASM SIMD
#if defined(__AVX2__) || defined(__AVX__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();

  float summs = 0;

  // Main loop
  for (int i = 0; i < nb; ++i) {
    const float d0 = NE_FP16_TO_FP32(x[i].d);
    const float d1 = y[i].d;

    summs += NE_FP16_TO_FP32(x[i].m) * y[i].s;

    const __m256 d0v = _mm256_set1_ps(d0);
    const __m256 d1v = _mm256_set1_ps(d1);

    // Compute combined scales
    const __m256 d0d1 = _mm256_mul_ps(d0v, d1v);

    // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
    const __m256i bx = bytes_from_nibbles_32(x[i].qs);
    const __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);

    const __m256 xy = mul_sum_us8_pairs_float(bx, by);

    // Accumulate d0*d1*x*y
#if defined(__AVX2__)
    acc = _mm256_fmadd_ps(d0d1, xy, acc);
#else
    acc = _mm256_add_ps(_mm256_mul_ps(d0d1, xy), acc);
#endif
  }

  *s = hsum_float_8(acc) + summs;
#else
  // scalar
  float sumf = 0.0;

  for (int i = 0; i < nb; i++) {
    int sumi = 0;

    for (int j = 0; j < qk / 2; ++j) {
      const int v0 = (x[i].qs[j] & 0x0F);
      const int v1 = (x[i].qs[j] >> 4);

      sumi += (v0 * y[i].qs[j]) + (v1 * y[i].qs[j + qk / 2]);
    }

    sumf += (NE_FP16_TO_FP32(x[i].d) * y[i].d) * sumi + NE_FP16_TO_FP32(x[i].m) * y[i].s;
  }

  *s = sumf;
#endif
}

static void ne_vec_dot_q5_0_q8_0(const int n, float* restrict s, const void* restrict vx, const void* restrict vy) {
  const int qk = QK8_0;
  const int nb = n / qk;

  assert(n % qk == 0);
  assert(nb % 2 == 0);
  assert(qk == QK5_0);

  const block_q5_0* restrict x = (const block_q5_0*)vx;
  const block_q8_0* restrict y = (const block_q8_0*)vy;

#if defined(__AVX2__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();

  // Main loop
  for (int i = 0; i < nb; i++) {
    /* Compute combined scale for the block */
    const __m256 d = _mm256_set1_ps(NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d));

    __m256i bx = bytes_from_nibbles_32(x[i].qs);
    __m256i bxhi = bytes_from_bits_32(x[i].qh);
    bxhi = _mm256_andnot_si256(bxhi, _mm256_set1_epi8((char)0xF0));
    bx = _mm256_or_si256(bx, bxhi);

    __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);

    const __m256 q = mul_sum_i8_pairs_float(bx, by);

    /* Multiply q with scale and accumulate */
    acc = _mm256_fmadd_ps(d, q, acc);
  }

  *s = hsum_float_8(acc);
#elif defined(__AVX__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();
  __m128i mask = _mm_set1_epi8((char)0xF0);

  // Main loop
  for (int i = 0; i < nb; i++) {
    /* Compute combined scale for the block */
    const __m256 d = _mm256_set1_ps(NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d));

    __m256i bx = bytes_from_nibbles_32(x[i].qs);
    const __m256i bxhi = bytes_from_bits_32(x[i].qh);
    __m128i bxhil = _mm256_castsi256_si128(bxhi);
    __m128i bxhih = _mm256_extractf128_si256(bxhi, 1);
    bxhil = _mm_andnot_si128(bxhil, mask);
    bxhih = _mm_andnot_si128(bxhih, mask);
    __m128i bxl = _mm256_castsi256_si128(bx);
    __m128i bxh = _mm256_extractf128_si256(bx, 1);
    bxl = _mm_or_si128(bxl, bxhil);
    bxh = _mm_or_si128(bxh, bxhih);
    bx = _mm256_set_m128i(bxh, bxl);

    const __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);

    const __m256 q = mul_sum_i8_pairs_float(bx, by);

    /* Multiply q with scale and accumulate */
    acc = _mm256_add_ps(_mm256_mul_ps(d, q), acc);
  }

  *s = hsum_float_8(acc);
#else
  // scalar
  float sumf = 0.0;

  for (int i = 0; i < nb; i++) {
    uint32_t qh;
    memcpy(&qh, x[i].qh, sizeof(qh));

    int sumi = 0;

    for (int j = 0; j < qk / 2; ++j) {
      const uint8_t xh_0 = ((qh & (1u << (j + 0))) >> (j + 0)) << 4;
      const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

      const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
      const int32_t x1 = ((x[i].qs[j] >> 4) | xh_1) - 16;

      sumi += (x0 * y[i].qs[j]) + (x1 * y[i].qs[j + qk / 2]);
    }

    sumf += (NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d)) * sumi;
  }

  *s = sumf;
#endif
}

static void ne_vec_dot_q5_1_q8_1(const int n, float* restrict s, const void* restrict vx, const void* restrict vy) {
  const int qk = QK8_1;
  const int nb = n / qk;

  assert(n % qk == 0);
  assert(nb % 2 == 0);
  assert(qk == QK5_1);

  const block_q5_1* restrict x = (const block_q5_1*)vx;
  const block_q8_1* restrict y = (const block_q8_1*)vy;

#if defined(__AVX2__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();

  float summs = 0.0f;

  // Main loop
  for (int i = 0; i < nb; i++) {
    const __m256 dx = _mm256_set1_ps(NE_FP16_TO_FP32(x[i].d));

    summs += NE_FP16_TO_FP32(x[i].m) * y[i].s;

    __m256i bx = bytes_from_nibbles_32(x[i].qs);
    __m256i bxhi = bytes_from_bits_32(x[i].qh);
    bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
    bx = _mm256_or_si256(bx, bxhi);

    const __m256 dy = _mm256_set1_ps(y[i].d);
    const __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);

    const __m256 q = mul_sum_us8_pairs_float(bx, by);

    acc = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc);
  }

  *s = hsum_float_8(acc) + summs;
#elif defined(__AVX__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();
  __m128i mask = _mm_set1_epi8(0x10);

  float summs = 0.0f;

  // Main loop
  for (int i = 0; i < nb; i++) {
    const __m256 dx = _mm256_set1_ps(NE_FP16_TO_FP32(x[i].d));

    summs += NE_FP16_TO_FP32(x[i].m) * y[i].s;

    __m256i bx = bytes_from_nibbles_32(x[i].qs);
    const __m256i bxhi = bytes_from_bits_32(x[i].qh);
    __m128i bxhil = _mm256_castsi256_si128(bxhi);
    __m128i bxhih = _mm256_extractf128_si256(bxhi, 1);
    bxhil = _mm_and_si128(bxhil, mask);
    bxhih = _mm_and_si128(bxhih, mask);
    __m128i bxl = _mm256_castsi256_si128(bx);
    __m128i bxh = _mm256_extractf128_si256(bx, 1);
    bxl = _mm_or_si128(bxl, bxhil);
    bxh = _mm_or_si128(bxh, bxhih);
    bx = _mm256_set_m128i(bxh, bxl);

    const __m256 dy = _mm256_set1_ps(y[i].d);
    const __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);

    const __m256 q = mul_sum_us8_pairs_float(bx, by);

    acc = _mm256_add_ps(_mm256_mul_ps(q, _mm256_mul_ps(dx, dy)), acc);
  }

  *s = hsum_float_8(acc) + summs;
#else
  // scalar
  float sumf = 0.0;

  for (int i = 0; i < nb; i++) {
    uint32_t qh;
    memcpy(&qh, x[i].qh, sizeof(qh));

    int sumi = 0;

    for (int j = 0; j < qk / 2; ++j) {
      const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
      const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;

      const int32_t x0 = (x[i].qs[j] & 0xF) | xh_0;
      const int32_t x1 = (x[i].qs[j] >> 4) | xh_1;

      sumi += (x0 * y[i].qs[j]) + (x1 * y[i].qs[j + qk / 2]);
    }

    sumf += (NE_FP16_TO_FP32(x[i].d) * y[i].d) * sumi + NE_FP16_TO_FP32(x[i].m) * y[i].s;
  }

  *s = sumf;
#endif
}

static void ne_vec_dot_q8_0_q8_0(const int n, float* restrict s, const void* restrict vx, const void* restrict vy) {
  const int qk = QK8_0;
  const int nb = n / qk;

  assert(n % qk == 0);
  assert(nb % 2 == 0);

  const block_q8_0* restrict x = (const block_q8_0*)vx;
  const block_q8_0* restrict y = (const block_q8_0*)vy;

#if defined(__AVX2__) || defined(__AVX__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();

  // Main loop
  for (int i = 0; i < nb; ++i) {
    // Compute combined scale for the block
    const __m256 d = _mm256_set1_ps(NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d));
    __m256i bx = _mm256_loadu_si256((const __m256i*)x[i].qs);
    __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);

    const __m256 q = mul_sum_i8_pairs_float(bx, by);

    // Multiply q with scale and accumulate
#if defined(__AVX2__)
    acc = _mm256_fmadd_ps(d, q, acc);
#else
    acc = _mm256_add_ps(_mm256_mul_ps(d, q), acc);
#endif
  }

  *s = hsum_float_8(acc);
#else
  // scalar
  float sumf = 0.0;

  for (int i = 0; i < nb; i++) {
    int sumi = 0;

    for (int j = 0; j < qk; j++) {
      sumi += x[i].qs[j] * y[i].qs[j];
    }

    sumf += sumi * (NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d));
  }

  *s = sumf;
#endif
}

// compute NE_VEC_DOT_UNROLL dot products at once
// xs - x row stride in bytes
static void ne_vec_dot_f16_unroll(const int n, const int xs, float* restrict s, void* restrict xv,
                                  ne_fp16_t* restrict y) {
  ne_float sumf[NE_VEC_DOT_UNROLL] = {0.0};

  ne_fp16_t* restrict x[NE_VEC_DOT_UNROLL];

  for (int i = 0; i < NE_VEC_DOT_UNROLL; ++i) {
    x[i] = (ne_fp16_t*)((char*)xv + i * xs);
  }

#if defined(NE_SIMD)
  const int np = (n & ~(NE_F16_STEP - 1));

  NE_F16_VEC sum[NE_VEC_DOT_UNROLL][NE_F16_ARR] = {{NE_F16_VEC_ZERO}};

  NE_F16_VEC ax[NE_F16_ARR];
  NE_F16_VEC ay[NE_F16_ARR];

  for (int i = 0; i < np; i += NE_F16_STEP) {
    for (int j = 0; j < NE_F16_ARR; j++) {
      ay[j] = NE_F16_VEC_LOAD(y + i + j * NE_F16_EPR, j);

      for (int k = 0; k < NE_VEC_DOT_UNROLL; ++k) {
        ax[j] = NE_F16_VEC_LOAD(x[k] + i + j * NE_F16_EPR, j);

        sum[k][j] = NE_F16_VEC_FMA(sum[k][j], ax[j], ay[j]);
      }
    }
  }

  // reduce sum0..sum3 to sum0
  for (int k = 0; k < NE_VEC_DOT_UNROLL; ++k) {
    NE_F16_VEC_REDUCE(sumf[k], sum[k]);
  }

  // leftovers
  for (int i = np; i < n; ++i) {
    for (int j = 0; j < NE_VEC_DOT_UNROLL; ++j) {
      sumf[j] += (ne_float)(NE_FP16_TO_FP32(x[j][i]) * NE_FP16_TO_FP32(y[i]));
    }
  }
#else
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < NE_VEC_DOT_UNROLL; ++j) {
      sumf[j] += (ne_float)(NE_FP16_TO_FP32(x[j][i]) * NE_FP16_TO_FP32(y[i]));
    }
  }
#endif

  for (int i = 0; i < NE_VEC_DOT_UNROLL; ++i) {
    s[i] = sumf[i];
  }
}

#ifdef __cplusplus
}
#endif
