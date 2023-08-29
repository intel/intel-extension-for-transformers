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

#include <iostream>
#include <memory>
#include <random>

#include "jblas/jit_blas_utils.h"

#ifndef NE_TESTS
static_assert(false, "Only include this header file for testing!");
#endif

template <typename T>
void init_vector(std::vector<T>* v, float range1 = -10, float range2 = 10, int seed = 5489u) {
  float low_value = std::max(range1, static_cast<float>(std::numeric_limits<T>::lowest()) + 1);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> u(low_value, range2);
  for (size_t i = 0; i < v->size(); ++i) (*v)[i] = u(gen);
}

template <>
void init_vector<jblas::utils::bf16>(std::vector<jblas::utils::bf16>* v, float range1, float range2, int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> u(range1, range2);
  for (size_t i = 0; i < v->size(); ++i) (*v)[i] = jblas::utils::bf16(u(gen));
}

template <>
void init_vector<jblas::utils::fp16>(std::vector<jblas::utils::fp16>* v, float range1, float range2, int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> u(range1, range2);
  for (size_t i = 0; i < v->size(); ++i) (*v)[i] = jblas::utils::fp16(u(gen));
}

template <typename T>
struct s_is_u8s8 {
  enum { value = false };
};

template <>
struct s_is_u8s8<int8_t> {
  enum { value = true };
};

template <>
struct s_is_u8s8<uint8_t> {
  enum { value = true };
};

template <typename T>
inline typename std::enable_if<!s_is_u8s8<T>::value, float>::type get_err(const T& a, const T& b) {
  // we compare float relative error ratio here
  return fabs(static_cast<float>(a) - static_cast<float>(b)) /
         std::max(static_cast<float>(fabs(static_cast<float>(b))), 1.0f);
}
template <typename T>
inline typename std::enable_if<s_is_u8s8<T>::value, float>::type get_err(const T& a, const T& b) {
  // for quantized value, error ratio was calcualted with its data range
  return fabs(static_cast<float>(a) - static_cast<float>(b)) / UINT8_MAX;
}

template <typename T>
bool compare_data(const T* buf1, const T* buf2, size_t size, float eps = 1e-6) {
  if (buf1 == buf2) return false;

  for (size_t i = 0; i < size; ++i) {
    if (get_err(buf1[i], buf2[i]) > eps) {
      std::cerr << static_cast<float>(buf1[i]) << "vs" << static_cast<float>(buf2[i]) << " idx=" << i << std::endl;
      return false;
    }
  }
  return true;
}
