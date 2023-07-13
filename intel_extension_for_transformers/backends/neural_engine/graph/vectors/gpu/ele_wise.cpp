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
#include "vector_kernel.h"
#define GPU_BACKEND
#include "../parallel_for.h"
#include "../../core/data_types.h"
//
// fundamental operations
//
#define LOOP (n / VL * VL)
static sycl::queue q = sycl::queue();

inline static void ne_vec_set_i8(const int n, int8_t* x, const int8_t v) {
  constexpr size_t VL = 512 / 8 / sizeof(int8_t);
  vec_set_kernel_t kernel = vec_set_kernel_t<int8_t, VL>(x, v);
  vec_set_kernel_t kernel_tail = vec_set_kernel_t<int8_t, 1>(x + LOOP, v);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}

inline static void ne_vec_set_i16(const int n, int16_t* x, const int16_t v) {
  constexpr size_t VL = 512 / 8 / sizeof(int16_t);
  vec_set_kernel_t kernel = vec_set_kernel_t<int16_t, VL>(x, v);
  vec_set_kernel_t kernel_tail = vec_set_kernel_t<int16_t, 1>(x + LOOP, v);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}

inline static void ne_vec_set_i32(const int n, int32_t* x, const int32_t v) {
  constexpr size_t VL = 512 / 8 / sizeof(int32_t);
  vec_set_kernel_t kernel = vec_set_kernel_t<int32_t, VL>(x, v);
  vec_set_kernel_t kernel_tail = vec_set_kernel_t<int32_t, 1>(x + LOOP, v);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}

inline static void ne_vec_set_f16(const int n, ne_fp16_t* x, const int32_t v) {
  constexpr size_t VL = 512 / 8 / sizeof(ne_fp16_t);
  vec_set_kernel_t kernel = vec_set_kernel_t<ne_fp16_t, VL>(x, v);
  vec_set_kernel_t kernel_tail = vec_set_kernel_t<ne_fp16_t, 1>(x + LOOP, v);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}

inline static void ne_vec_add_f32(const int n, float* z, const float* x, const float* y) {
  constexpr size_t VL = 512 / 8 / sizeof(float);
  vec_add_kernel_t kernel = vec_add_kernel_t(z, const_cast<float*>(x), const_cast<float*>(y));
  vec_add_kernel_t kernel_tail =
      vec_add_kernel_t<float, float, 1>(z + LOOP, const_cast<float*>(x) + LOOP, const_cast<float*>(y) + LOOP);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}

inline static void ne_vec_add1_f32(const int n, float* z, const float* x, const float v) {
  constexpr size_t VL = 512 / 8 / sizeof(float);
  vec_scalar_add_kernel_t kernel = vec_scalar_add_kernel_t(z, const_cast<float*>(x), v);
  vec_scalar_add_kernel_t kernel_tail =
      vec_scalar_add_kernel_t<float, float, 1>(z + LOOP, const_cast<float*>(x) + LOOP, v);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}
inline static void ne_vec_acc_f32(const int n, float* y, const float* x) {
  constexpr size_t VL = 512 / 8 / sizeof(float);
  vec_add_kernel_t kernel = vec_add_kernel_t(y, const_cast<float*>(x), y);
  vec_add_kernel_t kernel_tail = vec_add_kernel_t<float, float, 1>(y + LOOP, const_cast<float*>(x) + LOOP, y + LOOP);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}
inline static void ne_vec_acc1_f32(const int n, float* y, const float v) {
  constexpr size_t VL = 512 / 8 / sizeof(float);
  vec_scalar_add_kernel_t kernel = vec_scalar_add_kernel_t(y, y, v);
  vec_scalar_add_kernel_t kernel_tail = vec_scalar_add_kernel_t<float, float, 1>(y + LOOP, y + LOOP, v);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}
inline static void ne_vec_sub_f32(const int n, float* z, const float* x, const float* y) {
  constexpr size_t VL = 512 / 8 / sizeof(float);
  vec_sub_kernel_t kernel = vec_sub_kernel_t(z, const_cast<float*>(x), const_cast<float*>(y));
  vec_sub_kernel_t kernel_tail = vec_sub_kernel_t<float, float, 1>(
      const_cast<float*>(z) + LOOP, const_cast<float*>(x) + LOOP, const_cast<float*>(y) + LOOP);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}

inline static void ne_vec_set_f32(const int n, float* x, const float v) {
  constexpr size_t VL = 512 / 8 / sizeof(float);
  vec_set_kernel_t kernel = vec_set_kernel_t<float, VL>(x, v);
  vec_set_kernel_t kernel_tail = vec_set_kernel_t<float, 1>(x + LOOP, v);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}

inline static void ne_vec_cpy_f32(const int n, float* y, const float* x) {
  constexpr size_t VL = 512 / 8 / sizeof(float);
  vec_cpy_kernel_t kernel = vec_cpy_kernel_t<float, VL>(const_cast<float*>(x), y);
  vec_cpy_kernel_t kernel_tail = vec_cpy_kernel_t<float, 1>(const_cast<float*>(x) + LOOP, y + LOOP);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}

inline static void ne_vec_neg_f32(const int n, float* y, const float* x) {
  constexpr size_t VL = 512 / 8 / sizeof(float);
  vec_scalar_mul_kernel_t kernel = vec_scalar_mul_kernel_t<float, float, VL>(y, const_cast<float*>(x), -1.f);
  vec_scalar_mul_kernel_t kernel_tail =
      vec_scalar_mul_kernel_t<float, float, 1>(y + LOOP, const_cast<float*>(x) + LOOP, -1.f);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}

inline static void ne_vec_mul_f32(const int n, float* z, const float* x, const float* y) {
  constexpr size_t VL = 512 / 8 / sizeof(float);
  vec_mul_kernel_t kernel = vec_mul_kernel_t<float, float, VL>(z, const_cast<float*>(x), const_cast<float*>(y));
  vec_mul_kernel_t kernel_tail =
      vec_mul_kernel_t<float, float, 1>(z + LOOP, const_cast<float*>(x) + LOOP, const_cast<float*>(y) + LOOP);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}
inline static void ne_vec_div_f32(const int n, float* z, const float* x, const float* y) {
  constexpr size_t VL = 512 / 8 / sizeof(float);
  vec_div_kernel_t kernel = vec_div_kernel_t<float, float, VL>(z, const_cast<float*>(x), const_cast<float*>(y));
  vec_div_kernel_t kernel_tail =
      vec_div_kernel_t<float, float, 1>(z + LOOP, const_cast<float*>(x) + LOOP, const_cast<float*>(y) + LOOP);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}

inline static void ne_vec_mad_f32(const int n, float* __restrict y, const float* __restrict x, const float v) {
  constexpr size_t VL = 512 / 8 / sizeof(float);
  vec_scalar_fma_kernel_t kernel = vec_scalar_fma_kernel_t<float, float, VL>(y, const_cast<float*>(x), v);
  vec_scalar_fma_kernel_t kernel_tail =
      vec_scalar_fma_kernel_t<float, float, 1>(y + LOOP, const_cast<float*>(x) + LOOP, v);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}

inline static void ne_vec_scale_f32(const int n, float* y, const float v) {
  constexpr size_t VL = 512 / 8 / sizeof(float);
  vec_scalar_mul_kernel_t kernel = vec_scalar_mul_kernel_t<float, float, VL>(y, y, v);
  vec_scalar_mul_kernel_t kernel_tail = vec_scalar_mul_kernel_t<float, float, 1>(y + LOOP, y + LOOP, v);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}

inline static void ne_vec_sqr_f32(const int n, float* y, const float* x) {
  constexpr size_t VL = 512 / 8 / sizeof(float);
  vec_mul_kernel_t kernel = vec_mul_kernel_t<float, float, VL>(y, y, const_cast<float*>(x));
  vec_mul_kernel_t kernel_tail = vec_mul_kernel_t<float, float, 1>(y + LOOP, y + LOOP, const_cast<float*>(x) + LOOP);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}
inline static void ne_vec_sqrt_f32(const int n, float* y, const float* x) {
  constexpr size_t VL = 512 / 8 / sizeof(float);
  vec_sqrt_kernel_t kernel = vec_sqrt_kernel_t<float, float, VL>(y, const_cast<float*>(x));
  vec_sqrt_kernel_t kernel_tail = vec_sqrt_kernel_t<float, float, 1>(y + LOOP, const_cast<float*>(x) + LOOP);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}
inline static void ne_vec_log_f32(const int n, float* y, const float* x) {
  constexpr size_t VL = 512 / 8 / sizeof(float);
  vec_log_kernel_t kernel = vec_log_kernel_t<float, float, VL>(y, const_cast<float*>(x));
  vec_log_kernel_t kernel_tail = vec_log_kernel_t<float, float, 1>(y + LOOP, const_cast<float*>(x) + LOOP);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}
inline static void ne_vec_abs_f32(const int n, float* y, const float* x) {
  constexpr size_t VL = 512 / 8 / sizeof(float);
  vec_abs_kernel_t kernel = vec_abs_kernel_t<float, float, VL>(y, const_cast<float*>(x));
  vec_abs_kernel_t kernel_tail = vec_abs_kernel_t<float, float, 1>(y + LOOP, const_cast<float*>(x) + LOOP);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}

// inline static void ne_vec_sgn_f32(const int n, float* y, const float* x) {
//   for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : ((x[i] < 0.f) ? -1.f : 0.f);
// }
// inline static void ne_vec_step_f32(const int n, float* y, const float* x) {
//   for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : 0.f;
// }
// inline static void ne_vec_relu_f32(const int n, float* y, const float* x) {
//   for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : 0.f;
// }

// static const float GELU_COEF_A = 0.044715f;
// static const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

// float ne_gelu_f32(float x) { return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x))); }

inline static void ne_vec_gelu_f32(const int n, float* y, const float* x) {
  constexpr size_t VL = 512 / 8 / sizeof(float);
  vec_gelu_kernel_t kernel = vec_gelu_kernel_t<float, float, VL>(y, const_cast<float*>(x));
  vec_gelu_kernel_t kernel_tail = vec_gelu_kernel_t<float, float, 1>(y + LOOP, const_cast<float*>(x) + LOOP);
  parallel_for<VL>(q, n, kernel, kernel_tail);
}

// inline static void ne_vec_gelu_f32_ref(const int n, float* y, const float* x) {
//   for (int i = 0; i < n; ++i) {
//     y[i] = ne_gelu_f32(x[i]);
//   }
// }

// inline static void ne_vec_gelu_f16(const int n, ne_fp16_t* y, const ne_fp16_t* x) {
//   const uint16_t* i16 = (const uint16_t*)x;
//   for (int i = 0; i < n; ++i) {
//     y[i] = table_gelu_f16[i16[i]];
//   }
// }

// #ifdef NE_GELU_FP16
// inline static void ne_vec_gelu_f32(const int n, float* y, const float* x) {
//   uint16_t t;
//   for (int i = 0; i < n; ++i) {
//     ne_fp16_t fp16 = NE_FP32_TO_FP16(x[i]);
//     memcpy(&t, &fp16, sizeof(uint16_t));
//     y[i] = NE_FP16_TO_FP32(table_gelu_f16[t]);
//   }
// }

// // Sigmoid Linear Unit (SiLU) function
// float ne_silu_f32(float x) { return x / (1.0f + expf(-x)); }

// //  inline static void ne_vec_silu_f16(const int n, ne_fp16_t * y, const ne_fp16_t * x) {
// //     const uint16_t * i16 = (const uint16_t *) x;
// //     for (int i = 0; i < n; ++i) {
// //         y[i] = table_silu_f16[i16[i]];
// //     }
// // }

// #ifdef NE_SILU_FP16
// inline static void ne_vec_silu_f32(const int n, float* y, const float* x) {
//   uint16_t t;
//   for (int i = 0; i < n; ++i) {
//     ne_fp16_t fp16 = NE_FP32_TO_FP16(x[i]);
//     memcpy(&t, &fp16, sizeof(uint16_t));
//     y[i] = NE_FP16_TO_FP32(table_silu_f16[t]);
//   }
// }

// inline static void ne_vec_silu_f32(const int n, float* y, const float* x) {
//   for (int i = 0; i < n; ++i) {
//     y[i] = ne_silu_f32(x[i]);
//   }
// }

// float ne_silu_backward_f32(float x, float dy) {
//   const float s = 1.0f / (1.0f + expf(-x));
//   return dy * s * (1.0f + x * (1.0f - s));
// }

// #ifdef NE_SILU_FP16
// inline static void ne_vec_silu_backward_f32(const int n, float* dx, const float* x, const float* dy) {
//   for (int i = 0; i < n; ++i) {
//     // we did not use x[i] to compute forward silu but its f16 equivalent
//     // take derivative at f16 of x[i]:
//     ne_fp16_t fp16 = NE_FP32_TO_FP16(x[i]);
//     float usedx = NE_FP16_TO_FP32(fp16);
//     dx[i] = ne_silu_backward_f32(usedx, dy[i]);
//   }
// }

// inline static void ne_vec_silu_backward_f32(const int n, float* dx, const float* x, const float* dy) {
//   for (int i = 0; i < n; ++i) {
//     dx[i] = ne_silu_backward_f32(x[i], dy[i]);
//   }
// }

// }
#include <iostream>
int main() {
  constexpr unsigned size = 17;
  sycl::queue q = sycl::queue();
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>() << "\n";

  char* src = malloc_shared<char>(size, q);
  float* src0 = malloc_shared<float>(size, q);
  float* src1 = malloc_shared<float>(size, q);
  float* dst = malloc_shared<float>(size, q);
  float* dst1 = malloc_shared<float>(size, q);
  //   float scale = 0.5f;
  for (unsigned i = 0; i < size; ++i) {
    src0[i] = 4 * i;
  }

  ne_vec_gelu_f32(size, dst, src0);

  for (unsigned i = 0; i < size; ++i) {
    // std::cout << (float)dst[i] << " ";
    printf("%f ", dst[i]);
  }
  std::cout << std::endl;
  for (unsigned i = 0; i < size; ++i) {
    std::cout << (float)dst1[i] << " ";
  }
  std::cout << std::endl;

  free(src0, q);
  free(src1, q);
  free(dst, q);
  return 0;
}
