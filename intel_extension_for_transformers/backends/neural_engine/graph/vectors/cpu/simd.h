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
#ifndef NE_GRAPH_SIMD_H
#define NE_GRAPH_SIMD_H
#if defined(_MSC_VER) || defined(__MINGW32__)
#include <intrin.h>
#else
#include <immintrin.h>
#endif

//
// simd mappings
//

// we define a common set of C macros which map to specific intrinsics based on the current architecture
// we then implement the fundamental computation operations below using only these macros
// adding support for new architectures requires to define the corresponding SIMD macros
//
// NE_F32_STEP / NE_F16_STEP
//   number of elements to process in a single step
//
// NE_F32_EPR / NE_F16_EPR
//   number of elements to fit in a single register
//

#if defined(__AVX__)

#define NE_SIMD

// F32 AVX

#define NE_F32_STEP 32
#define NE_F32_EPR 8

#define NE_F32x8 __m256
#define NE_F32x8_ZERO _mm256_setzero_ps()
#define NE_F32x8_SET1(x) _mm256_set1_ps(x)
#define NE_F32x8_LOAD _mm256_loadu_ps
#define NE_F32x8_STORE _mm256_storeu_ps
#if defined(__FMA__)
#define NE_F32x8_FMA(a, b, c) _mm256_fmadd_ps(b, c, a)
#else
#define NE_F32x8_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps(b, c), a)
#endif
#define NE_F32x8_ADD _mm256_add_ps
#define NE_F32x8_MUL _mm256_mul_ps
#define NE_F32x8_REDUCE(res, x)                                                                 \
  {                                                                                             \
    for (int i = 0; i < NE_F32_ARR / 2; ++i) {                                                  \
      x[2 * i] = _mm256_add_ps(x[2 * i], x[2 * i + 1]);                                         \
    }                                                                                           \
    for (int i = 0; i < NE_F32_ARR / 4; ++i) {                                                  \
      x[4 * i] = _mm256_add_ps(x[4 * i], x[4 * i + 2]);                                         \
    }                                                                                           \
    for (int i = 0; i < NE_F32_ARR / 8; ++i) {                                                  \
      x[8 * i] = _mm256_add_ps(x[8 * i], x[8 * i + 4]);                                         \
    }                                                                                           \
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]), _mm256_extractf128_ps(x[0], 1)); \
    const __m128 t1 = _mm_hadd_ps(t0, t0);                                                      \
    res = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));                                                   \
  }
// TODO: is this optimal ?

#define NE_F32_VEC NE_F32x8
#define NE_F32_VEC_ZERO NE_F32x8_ZERO
#define NE_F32_VEC_SET1 NE_F32x8_SET1
#define NE_F32_VEC_LOAD NE_F32x8_LOAD
#define NE_F32_VEC_STORE NE_F32x8_STORE
#define NE_F32_VEC_FMA NE_F32x8_FMA
#define NE_F32_VEC_ADD NE_F32x8_ADD
#define NE_F32_VEC_MUL NE_F32x8_MUL
#define NE_F32_VEC_REDUCE NE_F32x8_REDUCE

// F16 AVX

#define NE_F16_STEP 32
#define NE_F16_EPR 8

// F16 arithmetic is not supported by AVX, so we use F32 instead

#define NE_F32Cx8 __m256
#define NE_F32Cx8_ZERO _mm256_setzero_ps()
#define NE_F32Cx8_SET1(x) _mm256_set1_ps(x)

#if defined(__F16C__)
// the  _mm256_cvt intrinsics require F16C
#define NE_F32Cx8_LOAD(x) _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(x)))
#define NE_F32Cx8_STORE(x, y) _mm_storeu_si128((__m128i*)(x), _mm256_cvtps_ph(y, 0))
#else
static inline __m256 __avx_f32cx8_load(ne_fp16_t* x) {
  float tmp[8];

  for (int i = 0; i < 8; i++) {
    tmp[i] = NE_FP16_TO_FP32(x[i]);
  }

  return _mm256_loadu_ps(tmp);
}
static inline void __avx_f32cx8_store(ne_fp16_t* x, __m256 y) {
  float arr[8];

  _mm256_storeu_ps(arr, y);

  for (int i = 0; i < 8; i++) x[i] = NE_FP32_TO_FP16(arr[i]);
}
#define NE_F32Cx8_LOAD(x) __avx_f32cx8_load(x)
#define NE_F32Cx8_STORE(x, y) __avx_f32cx8_store(x, y)
#endif

#define NE_F32Cx8_FMA NE_F32x8_FMA
#define NE_F32Cx8_ADD _mm256_add_ps
#define NE_F32Cx8_MUL _mm256_mul_ps
#define NE_F32Cx8_REDUCE NE_F32x8_REDUCE

#define NE_F16_VEC NE_F32Cx8
#define NE_F16_VEC_ZERO NE_F32Cx8_ZERO
#define NE_F16_VEC_SET1 NE_F32Cx8_SET1
#define NE_F16_VEC_LOAD(p, i) NE_F32Cx8_LOAD(p)
#define NE_F16_VEC_STORE(p, r, i) NE_F32Cx8_STORE(p, r[i])
#define NE_F16_VEC_FMA NE_F32Cx8_FMA
#define NE_F16_VEC_ADD NE_F32Cx8_ADD
#define NE_F16_VEC_MUL NE_F32Cx8_MUL
#define NE_F16_VEC_REDUCE NE_F32Cx8_REDUCE

#elif defined(__SSE3__)

#define NE_SIMD

// F32 SSE

#define NE_F32_STEP 32
#define NE_F32_EPR 4

#define NE_F32x4 __m128
#define NE_F32x4_ZERO _mm_setzero_ps()
#define NE_F32x4_SET1(x) _mm_set1_ps(x)
#define NE_F32x4_LOAD _mm_loadu_ps
#define NE_F32x4_STORE _mm_storeu_ps
#if defined(__FMA__)
// TODO: Does this work?
#define NE_F32x4_FMA(a, b, c) _mm_fmadd_ps(b, c, a)
#else
#define NE_F32x4_FMA(a, b, c) _mm_add_ps(_mm_mul_ps(b, c), a)
#endif
#define NE_F32x4_ADD _mm_add_ps
#define NE_F32x4_MUL _mm_mul_ps
#define NE_F32x4_REDUCE(res, x)                      \
  {                                                  \
    for (int i = 0; i < NE_F32_ARR / 2; ++i) {       \
      x[2 * i] = _mm_add_ps(x[2 * i], x[2 * i + 1]); \
    }                                                \
    for (int i = 0; i < NE_F32_ARR / 4; ++i) {       \
      x[4 * i] = _mm_add_ps(x[4 * i], x[4 * i + 2]); \
    }                                                \
    for (int i = 0; i < NE_F32_ARR / 8; ++i) {       \
      x[8 * i] = _mm_add_ps(x[8 * i], x[8 * i + 4]); \
    }                                                \
    const __m128 t0 = _mm_hadd_ps(x[0], x[0]);       \
    res = _mm_cvtss_f32(_mm_hadd_ps(t0, t0));        \
  }
// TODO: is this optimal ?

#define NE_F32_VEC NE_F32x4
#define NE_F32_VEC_ZERO NE_F32x4_ZERO
#define NE_F32_VEC_SET1 NE_F32x4_SET1
#define NE_F32_VEC_LOAD NE_F32x4_LOAD
#define NE_F32_VEC_STORE NE_F32x4_STORE
#define NE_F32_VEC_FMA NE_F32x4_FMA
#define NE_F32_VEC_ADD NE_F32x4_ADD
#define NE_F32_VEC_MUL NE_F32x4_MUL
#define NE_F32_VEC_REDUCE NE_F32x4_REDUCE

// F16 SSE

#define NE_F16_STEP 32
#define NE_F16_EPR 4

static inline __m128 __sse_f16x4_load(ne_fp16_t* x) {
  float tmp[4];

  tmp[0] = NE_FP16_TO_FP32(x[0]);
  tmp[1] = NE_FP16_TO_FP32(x[1]);
  tmp[2] = NE_FP16_TO_FP32(x[2]);
  tmp[3] = NE_FP16_TO_FP32(x[3]);

  return _mm_loadu_ps(tmp);
}

static inline void __sse_f16x4_store(ne_fp16_t* x, __m128 y) {
  float arr[4];

  _mm_storeu_ps(arr, y);

  x[0] = NE_FP32_TO_FP16(arr[0]);
  x[1] = NE_FP32_TO_FP16(arr[1]);
  x[2] = NE_FP32_TO_FP16(arr[2]);
  x[3] = NE_FP32_TO_FP16(arr[3]);
}

#define NE_F32Cx4 __m128
#define NE_F32Cx4_ZERO _mm_setzero_ps()
#define NE_F32Cx4_SET1(x) _mm_set1_ps(x)
#define NE_F32Cx4_LOAD(x) __sse_f16x4_load(x)
#define NE_F32Cx4_STORE(x, y) __sse_f16x4_store(x, y)
#define NE_F32Cx4_FMA NE_F32x4_FMA
#define NE_F32Cx4_ADD _mm_add_ps
#define NE_F32Cx4_MUL _mm_mul_ps
#define NE_F32Cx4_REDUCE NE_F32x4_REDUCE

#define NE_F16_VEC NE_F32Cx4
#define NE_F16_VEC_ZERO NE_F32Cx4_ZERO
#define NE_F16_VEC_SET1 NE_F32Cx4_SET1
#define NE_F16_VEC_LOAD(p, i) NE_F32Cx4_LOAD(p)
#define NE_F16_VEC_STORE(p, r, i) NE_F32Cx4_STORE(p, r[i])
#define NE_F16_VEC_FMA NE_F32Cx4_FMA
#define NE_F16_VEC_ADD NE_F32Cx4_ADD
#define NE_F16_VEC_MUL NE_F32Cx4_MUL
#define NE_F16_VEC_REDUCE NE_F32Cx4_REDUCE

#endif

// NE_F32_ARR / NE_F16_ARR
//   number of registers to use per step
#ifdef NE_SIMD
#define NE_F32_ARR (NE_F32_STEP / NE_F32_EPR)
#define NE_F16_ARR (NE_F16_STEP / NE_F16_EPR)
#endif

#endif  // NE_GRAPH_SIMD_H
