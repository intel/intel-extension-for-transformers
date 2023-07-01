#pragma once
#include <inttypes.h>
#ifdef __cplusplus
extern "C" {
#endif

#ifdef VEC_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef VEC_BUILD
#define VEC_API __declspec(dllexport)
#else
#define VEC_API __declspec(dllimport)
#endif
#else
#define VEC_API __attribute__((visibility("default")))
#endif
#else
#define VEC_API
#endif

void ne_vec_set_i8_(const int n, int8_t* x, const int8_t v);

void ne_vec_set_i16_(const int n, int16_t* x, const int16_t v);

void ne_vec_set_i32_(const int n, int32_t* x, const int32_t v);

void ne_vec_set_f16_(const int n, uint16_t* x, const int32_t v);

void ne_vec_add_f32_(const int n, float* z, const float* x, const float* y);
void ne_vec_acc_f32_(const int n, float* y, const float* x);
void ne_vec_sub_f32_(const int n, float* z, const float* x, const float* y);

void ne_vec_set_f32_(const int n, float* x, const float v);

void ne_vec_mul_f32_(const int n, float* z, const float* x, const float* y);
void ne_vec_div_f32_(const int n, float* z, const float* x, const float* y);

#ifdef __cplusplus
}
#endif
