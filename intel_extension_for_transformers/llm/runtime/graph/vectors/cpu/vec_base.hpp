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

#ifndef ENGINE_EXECUTOR_INCLUDE_VEC_BASE_HPP_
#define ENGINE_EXECUTOR_INCLUDE_VEC_BASE_HPP_

#include <immintrin.h>
#include <cstdint>

#if __AVX512F__
struct fp32x16 {
  __m512 first;
};

struct s32x16 {
  __m512i first;
};
struct u32x16 {
  __m512i first;
};
#else
struct fp32x16 {
  __m256 first, second;
};

struct s32x16 {
  __m256i first, second;
};
struct u32x16 {
  __m256i first, second;
};
#define MASK_DECORATOR(blend_func, a, b, mask, res) \
  switch ((mask)) {                                 \
    case 1:                                         \
      (res) = blend_func((a), (b), 1);              \
      break;                                        \
    case 3:                                         \
      (res) = blend_func((a), (b), 3);              \
      break;                                        \
    case 7:                                         \
      (res) = blend_func((a), (b), 7);              \
      break;                                        \
    case 15:                                        \
      (res) = blend_func((a), (b), 15);             \
      break;                                        \
    case 31:                                        \
      (res) = blend_func((a), (b), 31);             \
      break;                                        \
    case 63:                                        \
      (res) = blend_func((a), (b), 63);             \
      break;                                        \
    case 127:                                       \
      (res) = blend_func((a), (b), 127);            \
      break;                                        \
    default:                                        \
      break;                                        \
  }

#endif

struct bf16x16 {
  __m256i first;
};

struct fp16x16 {
  __m256i first;
};

struct s16x16 {
  __m256i first;
};
struct s8x16 {
  __m128i first;
};
struct u8x16 {
  __m128i first;
};

#define CPU_VEC_STEP 16

template <typename T>
T load_kernel_t(const void* src) {
  return *reinterpret_cast<const T*>(src);
}

template <>
fp32x16 load_kernel_t<fp32x16>(const void* src);
template <>
bf16x16 load_kernel_t<bf16x16>(const void* src);

template <typename T>
void store_kernel_t(void* dst, T src) {
  T* dst_T = reinterpret_cast<T*>(dst);
  *dst_T = src;
}

template <>
void store_kernel_t<s8x16>(void* dst, s8x16 src);
template <>
void store_kernel_t<fp32x16>(void* dst, fp32x16 src);
template <>
void store_kernel_t<bf16x16>(void* dst, bf16x16 src);

template <typename dstT, typename src0T = void, typename src1T = void, typename src2T = void>
struct kernel_t {
  dstT (*func_)(src0T, src1T, src2T);
  void operator()(void* dst, const void* src0, const void* src1, const void* src2) {
    store_kernel_t<dstT>(dst,
                         func_(load_kernel_t<src0T>(src0), load_kernel_t<src1T>(src1), load_kernel_t<src2T>(src2)));
  }
};

template <typename dstT, typename src0T, typename src1T>
struct kernel_t<dstT, src0T, src1T, void> {
  dstT (*func_)(src0T, src1T);
  void operator()(void* dst, const void* src0, const void* src1) {
    store_kernel_t<dstT>(dst, func_(load_kernel_t<src0T>(src0), load_kernel_t<src1T>(src1)));
  }
};

template <typename dstT, typename src0T>
struct kernel_t<dstT, src0T, void, void> {
  dstT (*func_)(src0T);
  void operator()(void* dst, const void* src0) { store_kernel_t<dstT>(dst, func_(load_kernel_t<src0T>(src0))); }
};

template <typename dstT>
struct kernel_t<dstT, void, void, void> {
  dstT (*func_)();
  void operator()(void* dst) { store_kernel_t<dstT>(dst, func_()); }
};

#define REGISTER_KERNEL_T(func, ...)                           \
  struct ne_##func##_kernel_t : public kernel_t<__VA_ARGS__> { \
    ne_##func##_kernel_t() { func_ = func; }                   \
  };

#endif  // ENGINE_EXECUTOR_INCLUDE_VEC_BASE_HPP_
