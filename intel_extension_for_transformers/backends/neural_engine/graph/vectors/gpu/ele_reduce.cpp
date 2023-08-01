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

#include "reduce.h"

static sycl::queue q = sycl::queue();
// inline static void ne_vec_norm_f32 (const int n, float * s, const float * x) { ne_vec_dot_f32(n, s, x, x); *s =
// sqrtf(*s);   }

void ne_vec_sum_f32(const int n, float* s, const float* x) { reduce<float, sycl::plus<>, 16>(n, s, x, q); }

// inline static void ne_vec_sum_ggf(const int n, ne_float * s, const float * x) {
//     ne_float sum = 0.0;
//     for (int i = 0; i < n; ++i) {
//         sum += (ne_float)x[i];
//     }
//     *s = sum;
// }

void ne_vec_max_f32(const int n, float* s, const float* x) { reduce<float, sycl::maximum<float>, 16>(n, s, x, q); }

// inline static void ne_vec_norm_inv_f32(const int n, float * s, const float * x) {
//     ne_vec_norm_f32(n, s, x);
//     *s = 1.f/(*s);
// }
#include <iostream>
int main() {
  size_t n = 32 * 10;
  std::vector<float> h_src(n);
  std::vector<float> h_dst(n);
  for (size_t i = 0; i < n; i++) {
    h_src[i] = 1.f;
  }
  h_src[1] = 5.f;
  ne_vec_max_f32(n, h_dst.data(), h_src.data());
  std::cout << h_dst[0] << std::endl;
  return 0;
}
