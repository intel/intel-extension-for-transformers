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

#include "vec_load.hpp"

inline fp32x16 load_fp32x16(void const* mem_addr) {
#if __AVX512F__
  return _mm512_loadu_ps(mem_addr);
#else
  float const* mem_addr_fp32 = reinterpret_cast<float const*>(mem_addr);
  return {_mm256_loadu_ps(mem_addr_fp32), _mm256_loadu_ps(mem_addr_fp32 + 8)};
#endif
}

inline fp32x16 mask_load_fp32x16(fp32x16 src, int mask, void const* mem_addr) {
#if __AVX512F__
  return _mm512_mask_loadu_ps(src, mask, mem_addr);
#else
  float const* mem_addr_fp32 = reinterpret_cast<float const*>(mem_addr);
  return {_mm256_loadu_ps(mem_addr_fp32), _mm256_loadu_ps(mem_addr_fp32 + 8)};
#endif
}

inline bf16x16 load_bf16x16(void const* mem_addr) {
  __m256i const* mem_addr_bf16 = reinterpret_cast<__m256i const*>(mem_addr);
  return _mm256_loadu_si256(mem_addr_bf16);
}

inline bf16x16 maskz_load_bf16x16(int mask, void const* mem_addr) {
#if __AVX512F__
  __m256i const* mem_addr_bf16 = reinterpret_cast<__m256i const*>(mem_addr);
  return _mm256_maskz_loadu_epi16(mask, mem_addr_bf16);
#else
  bf16x16 res;
  MASK_DECORATOR(_mm256_blend_epi16, _mm256_setzero_si256(),
                 _mm256_loadu_si256(reinterpret_cast<__m256i const*>(mem_addr)), mask, res);
  return res;
#endif
}
