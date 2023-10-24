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

#ifndef ENGINE_EXECUTOR_INCLUDE_VEC_LOAD_HPP_
#define ENGINE_EXECUTOR_INCLUDE_VEC_LOAD_HPP_

#include "vec_base.hpp"

inline fp32x16 load_fp32x16(void const* mem_addr) {
#if __AVX512F__
  return {_mm512_loadu_ps(mem_addr)};
#else
  float const* mem_addr_fp32 = reinterpret_cast<float const*>(mem_addr);
  return {_mm256_loadu_ps(mem_addr_fp32), _mm256_loadu_ps(mem_addr_fp32 + 8)};
#endif
}
template <>
inline fp32x16 load_kernel_t<fp32x16>(const void* src) {
  return load_fp32x16(src);
}
inline fp32x16 mask_load_fp32x16(fp32x16 src, int mask, void const* mem_addr) {
#if __AVX512F__
  return {_mm512_mask_loadu_ps(src.first, mask, mem_addr)};
#else
  float const* mem_addr_fp32 = reinterpret_cast<float const*>(mem_addr);
  return {_mm256_loadu_ps(mem_addr_fp32), _mm256_loadu_ps(mem_addr_fp32 + 8)};
#endif
}

inline bf16x16 load_bf16x16(void const* mem_addr) {
  __m256i const* mem_addr_bf16 = reinterpret_cast<__m256i const*>(mem_addr);
  return {_mm256_loadu_si256(mem_addr_bf16)};
}
template <>
inline bf16x16 load_kernel_t<bf16x16>(const void* src) {
  return load_bf16x16(src);
}

inline bf16x16 maskz_load_bf16x16(int mask, void const* mem_addr);

#endif  // ENGINE_EXECUTOR_INCLUDE_VEC_LOAD_HPP_
