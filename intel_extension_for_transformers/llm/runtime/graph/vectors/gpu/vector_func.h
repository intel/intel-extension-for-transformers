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
#include <sycl/ext/intel/esimd.hpp>

template <typename T, size_t VL>
SYCL_EXTERNAL void usm_copy_from(T* src, sycl::ext::intel::esimd::simd<T, VL> vec, int i) SYCL_ESIMD_FUNCTION {
  vec.copy_from(src + i * VL);
}

template <typename T, size_t VL>
SYCL_EXTERNAL sycl::ext::intel::esimd::simd<T, VL> usm_copy_from(T* src, int i) SYCL_ESIMD_FUNCTION {
  sycl::ext::intel::esimd::simd<T, VL> vec;
  vec.copy_from(src + i * VL);
  return vec;
}

template <typename T, size_t VL>
SYCL_EXTERNAL void usm_copy_to(T* dst, sycl::ext::intel::esimd::simd<T, VL> vec, int i) SYCL_ESIMD_FUNCTION {
  vec.copy_to(dst + i * VL);
}

template <typename T, size_t VL>
SYCL_EXTERNAL void set_value(sycl::ext::intel::esimd::simd<T, VL>& vec, T value) SYCL_ESIMD_FUNCTION {
  vec = sycl::ext::intel::esimd::simd<T, VL>(value, 0);
}

template <typename T, size_t VL>
SYCL_EXTERNAL sycl::ext::intel::esimd::simd<T, VL> set_value(T value) SYCL_ESIMD_FUNCTION {
  return sycl::ext::intel::esimd::simd<T, VL>(value, 0);
}
template <typename T, size_t VL>
SYCL_EXTERNAL sycl::ext::intel::esimd::simd<T, VL> vec_tanh(sycl::ext::intel::esimd::simd<T, VL> src)
    SYCL_ESIMD_FUNCTION {
  auto exp2x = sycl::ext::intel::esimd::exp(src * 2.f);
  return (exp2x - 1.f) / (exp2x + 1.f);
}
