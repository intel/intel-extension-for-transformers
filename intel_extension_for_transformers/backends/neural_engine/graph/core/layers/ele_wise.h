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

#include <math.h>
#include "core/data_types.h"
#include "vectors/cpu/simd.h"

#ifdef __cplusplus
extern "C" {
#endif
//
// fundamental operations
//

inline static void ne_vec_set_i8(const int n, int8_t* x, const int8_t v) {
  for (int i = 0; i < n; ++i) x[i] = v;
}

inline static void ne_vec_set_i16(const int n, int16_t* x, const int16_t v) {
  for (int i = 0; i < n; ++i) x[i] = v;
}

inline static void ne_vec_set_i32(const int n, int32_t* x, const int32_t v) {
  for (int i = 0; i < n; ++i) x[i] = v;
}

inline static void ne_vec_set_f16(const int n, ne_fp16_t* x, const int32_t v) {
  for (int i = 0; i < n; ++i) x[i] = v;
}

inline static void ne_vec_add_f32(const int n, float* z, const float* x, const float* y) {
  for (int i = 0; i < n; ++i) z[i] = x[i] + y[i];
}
inline static void ne_vec_add1_f32(const int n, float* z, const float* x, const float v) {
  for (int i = 0; i < n; ++i) z[i] = x[i] + v;
}
inline static void ne_vec_acc_f32(const int n, float* y, const float* x) {
  for (int i = 0; i < n; ++i) y[i] += x[i];
}
inline static void ne_vec_acc1_f32(const int n, float* y, const float v) {
  for (int i = 0; i < n; ++i) y[i] += v;
}
inline static void ne_vec_sub_f32(const int n, float* z, const float* x, const float* y) {
  for (int i = 0; i < n; ++i) z[i] = x[i] - y[i];
}

inline static void ne_vec_set_f32(const int n, float* x, const float v) {
  for (int i = 0; i < n; ++i) x[i] = v;
}

inline static void ne_vec_cpy_f32(const int n, float* y, const float* x) {
  for (int i = 0; i < n; ++i) y[i] = x[i];
}
inline static void ne_vec_neg_f32(const int n, float* y, const float* x) {
  for (int i = 0; i < n; ++i) y[i] = -x[i];
}
inline static void ne_vec_mul_f32(const int n, float* z, const float* x, const float* y) {
  for (int i = 0; i < n; ++i) z[i] = x[i] * y[i];
}
inline static void ne_vec_div_f32(const int n, float* z, const float* x, const float* y) {
  for (int i = 0; i < n; ++i) z[i] = x[i] / y[i];
}

inline static void ne_vec_mad_f32(const int n, float* __restrict y, const float* __restrict x, const float v) {
#if defined(NE_SIMD)
  const int np = (n & ~(NE_F32_STEP - 1));

  NE_F32_VEC vx = NE_F32_VEC_SET1(v);

  NE_F32_VEC ax[NE_F32_ARR];
  NE_F32_VEC ay[NE_F32_ARR];

  for (int i = 0; i < np; i += NE_F32_STEP) {
    for (int j = 0; j < NE_F32_ARR; j++) {
      ax[j] = NE_F32_VEC_LOAD(x + i + j * NE_F32_EPR);
      ay[j] = NE_F32_VEC_LOAD(y + i + j * NE_F32_EPR);
      ay[j] = NE_F32_VEC_FMA(ay[j], ax[j], vx);

      NE_F32_VEC_STORE(y + i + j * NE_F32_EPR, ay[j]);
    }
  }

  // leftovers
  for (int i = np; i < n; ++i) {
    y[i] += x[i] * v;
  }
#else
  // scalar
  for (int i = 0; i < n; ++i) {
    y[i] += x[i] * v;
  }
#endif
}

// inline static void ne_vec_scale_f32(const int n, float * y, const float   v) { for (int i = 0; i < n; ++i) y[i] *=
// v;          }
inline static void ne_vec_scale_f32(const int n, float* y, const float v) {
#if defined(NE_SIMD)
  const int np = (n & ~(NE_F32_STEP - 1));

  NE_F32_VEC vx = NE_F32_VEC_SET1(v);

  NE_F32_VEC ay[NE_F32_ARR];

  for (int i = 0; i < np; i += NE_F32_STEP) {
    for (int j = 0; j < NE_F32_ARR; j++) {
      ay[j] = NE_F32_VEC_LOAD(y + i + j * NE_F32_EPR);
      ay[j] = NE_F32_VEC_MUL(ay[j], vx);

      NE_F32_VEC_STORE(y + i + j * NE_F32_EPR, ay[j]);
    }
  }

  // leftovers
  for (int i = np; i < n; ++i) {
    y[i] *= v;
  }
#else
  // scalar
  for (int i = 0; i < n; ++i) {
    y[i] *= v;
  }
#endif
}

inline static void ne_vec_sqr_f32(const int n, float* y, const float* x) {
  for (int i = 0; i < n; ++i) y[i] = x[i] * x[i];
}
inline static void ne_vec_sqrt_f32(const int n, float* y, const float* x) {
  for (int i = 0; i < n; ++i) y[i] = sqrtf(x[i]);
}
inline static void ne_vec_log_f32(const int n, float* y, const float* x) {
  for (int i = 0; i < n; ++i) y[i] = logf(x[i]);
}
inline static void ne_vec_abs_f32(const int n, float* y, const float* x) {
  for (int i = 0; i < n; ++i) y[i] = fabsf(x[i]);
}
inline static void ne_vec_sgn_f32(const int n, float* y, const float* x) {
  for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : ((x[i] < 0.f) ? -1.f : 0.f);
}
inline static void ne_vec_step_f32(const int n, float* y, const float* x) {
  for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : 0.f;
}
inline static void ne_vec_relu_f32(const int n, float* y, const float* x) {
  for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : 0.f;
}

static const float GELU_COEF_A = 0.044715f;
static const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

inline static float ne_gelu_f32(float x) {
  return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

inline static void ne_vec_gelu_f16(const int n, ne_fp16_t* y, const ne_fp16_t* x) {
  const uint16_t* i16 = (const uint16_t*)x;
  for (int i = 0; i < n; ++i) {
    y[i] = table_gelu_f16[i16[i]];
  }
}

#ifdef NE_GELU_FP16
inline static void ne_vec_gelu_f32(const int n, float* y, const float* x) {
  uint16_t t;
  for (int i = 0; i < n; ++i) {
    ne_fp16_t fp16 = NE_FP32_TO_FP16(x[i]);
    memcpy(&t, &fp16, sizeof(uint16_t));
    y[i] = NE_FP16_TO_FP32(table_gelu_f16[t]);
  }
}
#else
inline static void ne_vec_gelu_f32(const int n, float* y, const float* x) {
  for (int i = 0; i < n; ++i) {
    y[i] = ne_gelu_f32(x[i]);
  }
}
#endif

// Sigmoid Linear Unit (SiLU) function
inline static float ne_silu_f32(float x) { return x / (1.0f + expf(-x)); }

// inline static void ne_vec_silu_f16(const int n, ne_fp16_t * y, const ne_fp16_t * x) {
//     const uint16_t * i16 = (const uint16_t *) x;
//     for (int i = 0; i < n; ++i) {
//         y[i] = table_silu_f16[i16[i]];
//     }
// }

#ifdef NE_SILU_FP16
inline static void ne_vec_silu_f32(const int n, float* y, const float* x) {
  uint16_t t;
  for (int i = 0; i < n; ++i) {
    ne_fp16_t fp16 = NE_FP32_TO_FP16(x[i]);
    memcpy(&t, &fp16, sizeof(uint16_t));
    y[i] = NE_FP16_TO_FP32(table_silu_f16[t]);
  }
}
#else
inline static void ne_vec_silu_f32(const int n, float* y, const float* x) {
  for (int i = 0; i < n; ++i) {
    y[i] = ne_silu_f32(x[i]);
  }
}
#endif

inline static float ne_silu_backward_f32(float x, float dy) {
  const float s = 1.0f / (1.0f + expf(-x));
  return dy * s * (1.0f + x * (1.0f - s));
}

#ifdef NE_SILU_FP16
inline static void ne_vec_silu_backward_f32(const int n, float* dx, const float* x, const float* dy) {
  for (int i = 0; i < n; ++i) {
    // we did not use x[i] to compute forward silu but its f16 equivalent
    // take derivative at f16 of x[i]:
    ne_fp16_t fp16 = NE_FP32_TO_FP16(x[i]);
    float usedx = NE_FP16_TO_FP32(fp16);
    dx[i] = ne_silu_backward_f32(usedx, dy[i]);
  }
}
#else
inline static void ne_vec_silu_backward_f32(const int n, float* dx, const float* x, const float* dy) {
  for (int i = 0; i < n; ++i) {
    dx[i] = ne_silu_backward_f32(x[i], dy[i]);
  }
}
#endif

#ifdef __cplusplus
}
#endif
