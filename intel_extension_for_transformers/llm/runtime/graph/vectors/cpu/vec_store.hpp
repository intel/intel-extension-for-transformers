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

#ifndef ENGINE_EXECUTOR_INCLUDE_VEC_STORE_HPP_
#define ENGINE_EXECUTOR_INCLUDE_VEC_STORE_HPP_

#include "vec_base.hpp"

inline void store_s8x16(void* mem_addr, s8x16 a) { _mm_storeu_si128(reinterpret_cast<__m128i*>(mem_addr), a.first); }
inline void store_u8x16(void* mem_addr, u8x16 a) { _mm_storeu_si128(reinterpret_cast<__m128i*>(mem_addr), a.first); }
template <>
inline void store_kernel_t<s8x16>(void* dst, s8x16 src) {
  store_s8x16(dst, src);
}

inline void mask_store_s8x16(void* mem_addr, const int mask, s8x16 a) {
#ifdef __AVX512F__
  _mm_mask_storeu_epi8(mem_addr, mask, a.first);
#else
  __m128i mask_reg =
      _mm_set_epi8(mask & 32768, mask & 16384, mask & 8192, mask & 4096, mask & 2048, mask & 1024, mask & 512,
                   mask & 256, mask & 128, mask & 64, mask & 32, mask & 16, mask & 8, mask & 4, mask & 2, mask & 1);
  _mm_maskmoveu_si128(a.first, mask_reg, reinterpret_cast<char*>(mem_addr));
#endif
}

inline void mask_store_u8x16(void* mem_addr, const int mask, u8x16 a) {
#ifdef __AVX512F__
  _mm_mask_storeu_epi8(mem_addr, mask, a.first);
#else
  __m128i mask_reg =
      _mm_set_epi8(mask & 32768, mask & 16384, mask & 8192, mask & 4096, mask & 2048, mask & 1024, mask & 512,
                   mask & 256, mask & 128, mask & 64, mask & 32, mask & 16, mask & 8, mask & 4, mask & 2, mask & 1);
  _mm_maskmoveu_si128(a.first, mask_reg, reinterpret_cast<char*>(mem_addr));
#endif
}

inline void store_fp32x16(void* mem_addr, fp32x16 a) {
#ifdef __AVX512F__
  _mm512_storeu_ps(mem_addr, a.first);
#else
  float* mem_addr_fp32 = reinterpret_cast<float*>(mem_addr);
  _mm256_storeu_ps(mem_addr_fp32, a.first);
  _mm256_storeu_ps(mem_addr_fp32 + 8, a.second);
#endif
}

template <>
inline void store_kernel_t<fp32x16>(void* dst, fp32x16 src) {
  store_fp32x16(dst, src);
}

inline void store_bf16x16(void* mem_addr, bf16x16 a) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem_addr), a.first);
}

template <>
inline void store_kernel_t<bf16x16>(void* dst, bf16x16 src) {
  store_bf16x16(dst, src);
}

#endif  // ENGINE_EXECUTOR_INCLUDE_VEC_STORE_HPP_
