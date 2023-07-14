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

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// floating point type used to accumulate sums
typedef double ne_float;
typedef uint16_t ne_fp16_t;

enum ne_type {
  NE_TYPE_F32 = 0,
  NE_TYPE_F16 = 1,
  NE_TYPE_Q4_0 = 2,
  NE_TYPE_Q4_1 = 3,
  // NE_TYPE_Q4_2 = 4, support has been removed
  // NE_TYPE_Q4_3 (5) support has been removed
  NE_TYPE_Q5_0 = 6,
  NE_TYPE_Q5_1 = 7,
  NE_TYPE_Q8_0 = 8,
  NE_TYPE_Q8_1 = 9,
  NE_TYPE_I8,
  NE_TYPE_I16,
  NE_TYPE_I32,
  NE_TYPE_Q4_JBLAS,
  NE_TYPE_COUNT,
};

// model file types
enum ne_ftype {
  NE_FTYPE_UNKNOWN = -1,
  NE_FTYPE_ALL_F32 = 0,
  NE_FTYPE_MOSTLY_F16 = 1,            // except 1d tensors
  NE_FTYPE_MOSTLY_Q4_0 = 2,           // except 1d tensors
  NE_FTYPE_MOSTLY_Q4_1 = 3,           // except 1d tensors
  NE_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
  NE_FTYPE_MOSTLY_Q8_0 = 7,           // except 1d tensors
  NE_FTYPE_MOSTLY_Q5_0 = 8,           // except 1d tensors
  NE_FTYPE_MOSTLY_Q5_1 = 9,           // except 1d tensors
  NE_FTYPE_MOSTLY_Q4_JBLAS_B32 = 10,            // except 1d tensors
  NE_FTYPE_MOSTLY_Q4_JBLAS_B128 = 11,           // except 1d tensors
  NE_FTYPE_MOSTLY_Q4_JBLAS_B1024 = 12,          // except 1d tensors
  NE_FTYPE_MOSTLY_Q4_JBLAS_BF16_B32 = 13,       // except 1d tensors
  NE_FTYPE_MOSTLY_Q4_JBLAS_VNNI_B32 = 14,       // except 1d tensors
  NE_FTYPE_MOSTLY_Q4_JBLAS_VNNI_BF16_B32 = 15,  // except 1d tensors
  NE_FTYPE_MOSTLY_Q4_JBLAS_VNNI_B128 = 16,      // except 1d tensors
};

#define QK4_0 32
typedef struct {
  ne_fp16_t d;            // delta
  uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;

#define QK4_1 32
typedef struct {
  ne_fp16_t d;            // delta
  ne_fp16_t m;            // min
  uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;

#define QK5_0 32
typedef struct {
  ne_fp16_t d;            // delta
  uint8_t qh[4];          // 5-th bit of quants
  uint8_t qs[QK5_0 / 2];  // nibbles / quants
} block_q5_0;

#define QK5_1 32
typedef struct {
  ne_fp16_t d;            // delta
  ne_fp16_t m;            // min
  uint8_t qh[4];          // 5-th bit of quants
  uint8_t qs[QK5_1 / 2];  // nibbles / quants
} block_q5_1;

#define QK8_0 32
typedef struct {
  ne_fp16_t d;       // delta
  int8_t qs[QK8_0];  // quants
} block_q8_0;

#define QK8_1 32
typedef struct {
  float d;           // delta
  float s;           // d * sum(qs[i])
  int8_t qs[QK8_1];  // quants
} block_q8_1;

// ne_fp16_t related

#ifdef __F16C__

#ifdef _MSC_VER
#define NE_COMPUTE_FP16_TO_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
#define NE_COMPUTE_FP32_TO_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)
#else
#define NE_COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
#define NE_COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)
#endif

#else

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

static inline float fp32_from_bits(uint32_t w) {
  union {
    uint32_t as_bits;
    float as_value;
  } fp32;
  fp32.as_bits = w;
  return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
  union {
    float as_value;
    uint32_t as_bits;
  } fp32;
  fp32.as_value = f;
  return fp32.as_bits;
}

static inline float ne_compute_fp16_to_fp32(ne_fp16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;

  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
  const float exp_scale = 0x1.0p-112f;
#else
  const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
  const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
      sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

static inline ne_fp16_t ne_compute_fp32_to_fp16(float f) {
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
  const float scale_to_inf = 0x1.0p+112f;
  const float scale_to_zero = 0x1.0p-110f;
#else
  const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
  const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
  float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

  const uint32_t w = fp32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define NE_COMPUTE_FP16_TO_FP32(x) ne_compute_fp16_to_fp32(x)
#define NE_COMPUTE_FP32_TO_FP16(x) ne_compute_fp32_to_fp16(x)

#endif  // __F16C__

//
// global data
//

// precomputed gelu table for f16 (128 KB)
static ne_fp16_t table_gelu_f16[1 << 16];

// precomputed silu table for f16 (128 KB)
static ne_fp16_t table_silu_f16[1 << 16];

// precomputed exp table for f16 (128 KB)
static ne_fp16_t table_exp_f16[1 << 16];

// precomputed f32 table for f16 (256 KB)
static float table_f32_f16[1 << 16];

#if !defined(NE_FP16_TO_FP32) || !defined(NE_FP32_TO_FP16)

inline static float ne_lookup_fp16_to_fp32(ne_fp16_t f) {
  uint16_t s;
  memcpy(&s, &f, sizeof(uint16_t));
  return table_f32_f16[s];
}

#define NE_FP16_TO_FP32(x) ne_lookup_fp16_to_fp32(x)
#define NE_FP32_TO_FP16(x) NE_COMPUTE_FP32_TO_FP16(x)

#endif

#ifdef __cplusplus
}
#endif
