#include "vectors/cpu/vec.hpp"
#include "vectors/ele_reduce.h"
#include "cmath"

void ne_vec_norm_f32_(const int n, float* s, const float* x) {
  float sum = 0.0;
  ne_dot_fp32x16_kernel_t k_t;
  for (int i = 0; i < n / 16; ++i) {
    float tmp;
    k_t(reinterpret_cast<void*>(&tmp), reinterpret_cast<const void*>(x + i * 16),
        reinterpret_cast<const void*>(x + i * 16));
    sum += tmp;
  }
  for (int i = n / 16 * 16; i < n; i++) sum += x[i] * x[i];
  *s = sqrtf(sum);
}

void ne_vec_sum_f32_(const int n, float* s, const float* x) {
  float sum = 0.0;
  ne_reduce_add_fp32x16_kernel_t k_t;
  for (int i = 0; i < n / 16; ++i) {
    float tmp;
    k_t(reinterpret_cast<void*>(&tmp), reinterpret_cast<const void*>(x + i * 16));
    sum += tmp;
  }
  for (int i = n / 16 * 16; i < n; i++) sum += x[i];
  *s = sum;
}

void ne_vec_max_f32_(const int n, float* s, const float* x) {
  float max = -INFINITY;
  ne_reduce_max_fp32x16_kernel_t k_t;
  for (int i = 0; i < n / 16; ++i) {
    float tmp;
    k_t(reinterpret_cast<void*>(&tmp), reinterpret_cast<const void*>(x + i * 16));
    max = max > tmp ? max : tmp;
  }
  for (int i = n / 16 * 16; i < n; i++) {
    max = x[i] > max ? x[i] : max;
  }
  *s = max;
}

void ne_vec_norm_inv_f32_(const int n, float* s, const float* x) {
  ne_vec_norm_f32_(n, s, x);
  *s = 1.f / (*s);
}
void ne_vec_sum_ggf_(const int n, double* s, const float* x) {
  float sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += (float)x[i];
  }
  *s = sum;
}
