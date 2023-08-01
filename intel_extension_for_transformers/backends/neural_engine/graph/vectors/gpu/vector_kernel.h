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
#include "vector_func.h"
#include <sycl/sycl.hpp>

template <typename T, size_t VL = 1>
struct vec_set_kernel_t {
  T* x;
  T value;
  vec_set_kernel_t(T* ptr, T v) : x(ptr), value(v) {}
  void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
    auto vec = set_value<T, VL>(value);
    usm_copy_to<T, VL>(x, vec, ndi.get_global_id(0));
  }
};

template <typename T, size_t VL = 16>
struct vec_cpy_kernel_t {
  T* x;
  T* y;
  vec_cpy_kernel_t(T* x, T* y) : x(x), y(y) {}
  void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
    auto v_x = usm_copy_from<T, VL>(x, ndi.get_global_id(0));
    usm_copy_to<T, VL>(y, v_x, ndi.get_global_id(0));
  }
};

template <typename srcT, typename dstT = srcT, size_t VL = 16>
struct vec_add_kernel_t {
  srcT* add1;
  srcT* add2;
  dstT* ret;
  vec_add_kernel_t(dstT* z, srcT* x, srcT* y) : ret(z), add1(x), add2(y) {}
  void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
    auto vadd1 = usm_copy_from<srcT, VL>(add1, ndi.get_global_id(0));
    auto vadd2 = usm_copy_from<srcT, VL>(add2, ndi.get_global_id(0));
    auto vsum = vadd1 + vadd2;
    usm_copy_to<dstT, VL>(ret, vsum, ndi.get_global_id(0));
  }
};
template <typename srcT, typename dstT = srcT, size_t VL = 16>
struct vec_sub_kernel_t {
  srcT* num;
  srcT* minuend;
  dstT* ret;
  vec_sub_kernel_t(dstT* z, srcT* x, srcT* y) : ret(z), num(x), minuend(y) {}
  void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
    auto vnum = usm_copy_from<srcT, VL>(num, ndi.get_global_id(0));
    auto vminuend = usm_copy_from<srcT, VL>(minuend, ndi.get_global_id(0));
    auto vret = vnum - vminuend;
    usm_copy_to<dstT, VL>(ret, vret, ndi.get_global_id(0));
  }
};
template <typename srcT, typename dstT = srcT, size_t VL = 16>
struct vec_scalar_add_kernel_t {
  srcT* add1;
  srcT add2;
  dstT* ret;
  vec_scalar_add_kernel_t(dstT* z, srcT* x, srcT y) : ret(z), add1(x), add2(y) {}
  void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
    auto vadd1 = usm_copy_from<srcT, VL>(add1, ndi.get_global_id(0));
    auto vsum = vadd1 + add2;
    usm_copy_to<dstT, VL>(ret, vsum, ndi.get_global_id(0));
  }
};
template <typename srcT, typename dstT = srcT, size_t VL = 16>
struct vec_scalar_mul_kernel_t {
  srcT* num;
  srcT mul;
  dstT* ret;
  vec_scalar_mul_kernel_t(dstT* z, srcT* x, srcT y) : ret(z), num(x), mul(y) {}
  void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
    auto vnum = usm_copy_from<srcT, VL>(num, ndi.get_global_id(0));
    auto vret = vnum * mul;
    usm_copy_to<dstT, VL>(ret, vret, ndi.get_global_id(0));
  }
};
template <typename srcT, typename dstT = srcT, size_t VL = 16>
struct vec_mul_kernel_t {
  srcT* num1;
  srcT* num2;
  dstT* ret;
  vec_mul_kernel_t(dstT* z, srcT* x, srcT* y) : ret(z), num1(x), num2(y) {}
  void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
    auto vnum1 = usm_copy_from<srcT, VL>(num1, ndi.get_global_id(0));
    auto vnum2 = usm_copy_from<srcT, VL>(num2, ndi.get_global_id(0));
    auto vret = vnum1 * vnum2;
    usm_copy_to<dstT, VL>(ret, vret, ndi.get_global_id(0));
  }
};
template <typename srcT, typename dstT = srcT, size_t VL = 16>
struct vec_div_kernel_t {
  srcT* num1;
  srcT* num2;
  dstT* ret;
  vec_div_kernel_t(dstT* z, srcT* x, srcT* y) : ret(z), num1(x), num2(y) {}
  void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
    auto vnum1 = usm_copy_from<srcT, VL>(num1, ndi.get_global_id(0));
    auto vnum2 = usm_copy_from<srcT, VL>(num2, ndi.get_global_id(0));
    auto vret = vnum1 / vnum2;
    usm_copy_to<dstT, VL>(ret, vret, ndi.get_global_id(0));
  }
};

template <typename srcT, typename dstT = srcT, size_t VL = 16>
struct vec_scalar_fma_kernel_t {
  srcT* num1;
  srcT num2;
  dstT* ret;
  vec_scalar_fma_kernel_t(dstT* z, srcT* x, srcT y) : ret(z), num1(x), num2(y) {}
  void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
    auto vnum1 = usm_copy_from<srcT, VL>(num1, ndi.get_global_id(0));
    auto vret = usm_copy_from<srcT, VL>(num1, ndi.get_global_id(0));
    vret += (vnum1 * num2);
    usm_copy_to<dstT, VL>(ret, vret, ndi.get_global_id(0));
  }
};

template <typename srcT, typename dstT = srcT, size_t VL = 16>
struct vec_sqrt_kernel_t {
  srcT* num1;
  dstT* ret;
  vec_sqrt_kernel_t(dstT* z, srcT* x) : ret(z), num1(x) {}
  void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
    auto vnum1 = usm_copy_from<srcT, VL>(num1, ndi.get_global_id(0));
    auto vret = sycl::ext::intel::esimd::sqrt(vnum1);
    usm_copy_to<dstT, VL>(ret, vret, ndi.get_global_id(0));
  }
};

template <typename srcT, typename dstT = srcT, size_t VL = 16>
struct vec_log_kernel_t {
  srcT* num1;
  dstT* ret;
  vec_log_kernel_t(dstT* z, srcT* x) : ret(z), num1(x) {}
  void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
    auto vnum1 = usm_copy_from<srcT, VL>(num1, ndi.get_global_id(0));
    auto vret = sycl::ext::intel::esimd::log(vnum1);
    usm_copy_to<dstT, VL>(ret, vret, ndi.get_global_id(0));
  }
};
template <typename srcT, typename dstT = srcT, size_t VL = 16>
struct vec_abs_kernel_t {
  srcT* num1;
  dstT* ret;
  vec_abs_kernel_t(dstT* z, srcT* x) : ret(z), num1(x) {}
  void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
    auto vnum1 = usm_copy_from<srcT, VL>(num1, ndi.get_global_id(0));
    auto vret = sycl::ext::intel::esimd::abs(vnum1);
    usm_copy_to<dstT, VL>(ret, vret, ndi.get_global_id(0));
  }
};
template <typename srcT, typename dstT = srcT, size_t VL = 16>
struct vec_gelu_kernel_t {
  srcT* num1;
  dstT* ret;
  static constexpr float C0 = 0.044715f;
  static constexpr float sqrt_two_over_pi = 0.79788458347320556640625f;

  vec_gelu_kernel_t(dstT* z, srcT* x) : ret(z), num1(x) {}
  void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
    auto vnum1 = usm_copy_from<srcT, VL>(num1, ndi.get_global_id(0));
    sycl::ext::intel::esimd::simd<srcT, VL> tmp = (sqrt_two_over_pi * vnum1 * (1.f + C0 * vnum1 * vnum1));
    // auto vret = 0.5f * vnum1 * (1.f + vec_tanh<srcT, VL>(tmp));
    auto vtanh_value = vec_tanh<srcT, VL>(tmp);
    auto vret = 0.5f * vnum1 * (1.f + vtanh_value);
    usm_copy_to<dstT, VL>(ret, vret, ndi.get_global_id(0));
  }
};
