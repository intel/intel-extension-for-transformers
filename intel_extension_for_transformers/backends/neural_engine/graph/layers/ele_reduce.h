#pragma once

#include "core/data_types.h"
#include "vectors/cpu/simd.h"
#include "layers/vec_dot.h"

#ifdef  __cplusplus
extern "C" {
#endif

inline static void ne_vec_norm_f32 (const int n, float * s, const float * x) { ne_vec_dot_f32(n, s, x, x); *s = sqrtf(*s);   }

inline static void ne_vec_sum_f32(const int n, float * s, const float * x) {
    ne_float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (ne_float)x[i];
    }
    *s = sum;
}

inline static void ne_vec_sum_ggf(const int n, ne_float * s, const float * x) {
    ne_float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (ne_float)x[i];
    }
    *s = sum;
}

inline static void ne_vec_max_f32(const int n, float * s, const float * x) {
    float max = -INFINITY;
    for (int i = 0; i < n; ++i) {
        max = x[i] > max ? x[i] : max;
    }
    *s = max;
}

inline static void ne_vec_norm_inv_f32(const int n, float * s, const float * x) {
    ne_vec_norm_f32(n, s, x);
    *s = 1.f/(*s);
}

#ifdef  __cplusplus
}
#endif