#include "vectors/cpu/vec.hpp"
#include "vectors/ele_wise.h"
#include "cmath"
#ifdef __cplusplus
extern "C" {
#endif
void ne_vec_set_i8_(const int n, int8_t* x, const int8_t v) {
  ne_set1_s8x16_kernel_t k_t;
  for (int i = 0; i < n / 16; ++i) {
    k_t(reinterpret_cast<void*>(x), reinterpret_cast<const void*>(&v));
  }
  for (int i = n / 16 * 16; i < n; i++) x[i] = v;
}

void ne_vec_set_i16_(const int n, int16_t* x, const int16_t v) {
  ne_set1_s16x16_kernel_t k_t;
  for (int i = 0; i < n / 16; ++i) {
    k_t(reinterpret_cast<void*>(x), reinterpret_cast<const void*>(&v));
  }
  for (int i = n / 16 * 16; i < n; i++) x[i] = v;
}

void ne_vec_set_i32_(const int n, int32_t* x, const int32_t v) {
  ne_set1_s32x16_kernel_t k_t;
  for (int i = 0; i < n / 16; ++i) {
    k_t(reinterpret_cast<void*>(x), reinterpret_cast<const void*>(&v));
  }
  for (int i = n / 16 * 16; i < n; i++) x[i] = v;
}

void ne_vec_set_f16_(const int n, uint16_t* x, const int32_t v) {
  ne_set1_fp16x16_kernel_t k_t;
  for (int i = 0; i < n / 16; ++i) {
    k_t(reinterpret_cast<void*>(x), reinterpret_cast<const void*>(&v));
  }
  for (int i = n / 16 * 16; i < n; i++) x[i] = v;
}

void ne_vec_add_f32_(const int n, float* z, const float* x, const float* y) {
  ne_add_fp32x16_kernel_t k_t;
  for (int i = 0; i < n / 16; ++i) {
    k_t(reinterpret_cast<void*>(z), reinterpret_cast<const void*>(x), reinterpret_cast<const void*>(y));
  }
  for (int i = n / 16 * 16; i < n; i++) z[i] = x[i] + y[i];
}
void ne_vec_acc_f32_(const int n, float* y, const float* x) {
  ne_add_fp32x16_kernel_t k_t;
  for (int i = 0; i < n / 16; ++i) {
    k_t(reinterpret_cast<void*>(y), reinterpret_cast<const void*>(x), reinterpret_cast<const void*>(y));
  }
  for (int i = n / 16 * 16; i < n; i++) y[i] = x[i] + y[i];
}
void ne_vec_sub_f32_(const int n, float* z, const float* x, const float* y) {
  ne_sub_fp32x16_kernel_t k_t;
  for (int i = 0; i < n / 16; ++i) {
    k_t(reinterpret_cast<void*>(z), reinterpret_cast<const void*>(x), reinterpret_cast<const void*>(y));
  }
  for (int i = n / 16 * 16; i < n; i++) z[i] = x[i] - y[i];
}

void ne_vec_set_f32_(const int n, float* x, const float v) {
  ne_set1_fp32x16_kernel_t k_t;
  for (int i = 0; i < n / 16; ++i) {
    k_t(reinterpret_cast<void*>(x), reinterpret_cast<const void*>(&v));
  }
  for (int i = n / 16 * 16; i < n; i++) x[i] = v;
}

void ne_vec_mul_f32_(const int n, float* z, const float* x, const float* y) {
  ne_mul_fp32x16_kernel_t k_t;
  for (int i = 0; i < n / 16; ++i) {
    k_t(reinterpret_cast<void*>(z), reinterpret_cast<const void*>(x), reinterpret_cast<const void*>(y));
  }
  for (int i = n / 16 * 16; i < n; i++) z[i] = x[i] * y[i];
}
void ne_vec_div_f32_(const int n, float* z, const float* x, const float* y) {
  ne_div_fp32x16_kernel_t k_t;
  for (int i = 0; i < n / 16; ++i) {
    k_t(reinterpret_cast<void*>(z), reinterpret_cast<const void*>(x), reinterpret_cast<const void*>(y));
  }
  for (int i = n / 16 * 16; i < n; i++) z[i] = x[i] / y[i];
}

#ifdef __cplusplus
}
#endif
