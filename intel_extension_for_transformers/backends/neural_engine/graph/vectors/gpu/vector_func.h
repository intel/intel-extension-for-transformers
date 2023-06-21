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
