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
#if defined(_MSC_VER) || defined(__MINGW32__)
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#include <string.h>
#include <math.h>
#include "core/data_types.h"

// precomputed gelu table for f16 (128 KB)
static ggml_fp16_t table_gelu_f16[1 << 16];

// precomputed silu table for f16 (128 KB)
static ggml_fp16_t table_silu_f16[1 << 16];

// precomputed exp table for f16 (128 KB)
static ggml_fp16_t table_exp_f16[1 << 16];

// precomputed f32 table for f16 (256 KB)
static float table_f32_f16[1 << 16];

#ifdef __F16C__

#ifdef _MSC_VER
#define GGML_COMPUTE_FP16_TO_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
#define GGML_COMPUTE_FP32_TO_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)
#else
#define GGML_COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)
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

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
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

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
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

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#endif  // __F16C__

#if !defined(GGML_FP16_TO_FP32) || !defined(GGML_FP32_TO_FP16)

inline static float ggml_lookup_fp16_to_fp32(ggml_fp16_t f) {
  uint16_t s;
  memcpy(&s, &f, sizeof(uint16_t));
  return table_f32_f16[s];
}
#define GGML_FP16_TO_FP32(x) ggml_lookup_fp16_to_fp32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

#endif

float ggml_fp16_to_fp32(ggml_fp16_t x) { return (float)GGML_FP16_TO_FP32(x); }

ggml_fp16_t ggml_fp32_to_fp16(float x) { return GGML_FP32_TO_FP16(x); }

void ggml_fp16_to_fp32_row(const ggml_fp16_t* x, float* y, size_t n) {
  for (size_t i = 0; i < n; i++) {
    y[i] = GGML_FP16_TO_FP32(x[i]);
  }
}

void ggml_fp32_to_fp16_row(const float* x, ggml_fp16_t* y, size_t n) {
  size_t i = 0;
#if defined(__F16C__)
  for (; i + 7 < n; i += 8) {
    __m256 x_vec = _mm256_loadu_ps(x + i);
    __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128((__m128i*)(y + i), y_vec);
  }
  for (; i + 3 < n; i += 4) {
    __m128 x_vec = _mm_loadu_ps(x + i);
    __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
    _mm_storel_epi64((__m128i*)(y + i), y_vec);
  }
#endif
  for (; i < n; i++) {
    y[i] = GGML_FP32_TO_FP16(x[i]);
  }
}
