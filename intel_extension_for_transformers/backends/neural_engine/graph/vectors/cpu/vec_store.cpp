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

#include "vec_store.hpp"

inline void store_int8x16(void* mem_addr, int8x16 a) { _mm_storeu_si128(reinterpret_cast<__m128i*>(mem_addr), a); }
inline void mask_store_int8x16(void* mem_addr, const int mask, int8x16 a) {
#ifdef __AVX512F__
  _mm_mask_storeu_epi8(mem_addr, mask, a);
#else
  __m128i mask_reg =
      _mm_set_epi8(mask & 32768, mask & 16384, mask & 8192, mask & 4096, mask & 2048, mask & 1024, mask & 512,
                   mask & 256, mask & 128, mask & 64, mask & 32, mask & 16, mask & 8, mask & 4, mask & 2, mask & 1);
  _mm_maskmoveu_si128(a, mask_reg, reinterpret_cast<char*>(mem_addr));
#endif
}

inline void store_fp32x16(void* mem_addr, fp32x16 a) {
#ifdef __AVX512F__
  _mm512_storeu_ps(mem_addr, a);
#else
  float* mem_addr_fp32 = reinterpret_cast<float*>(mem_addr);
  _mm256_storeu_ps(mem_addr_fp32, a.first);
  _mm256_storeu_ps(mem_addr_fp32 + 8, a.second);
#endif
}

inline void store_bf16x16(void* mem_addr, bf16x16 a) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem_addr), a); }

inline void cvtuint32x16_store_int8x16(void* base_addr, int32x16 a) {
#ifdef __AVX512F__
  _mm512_mask_cvtusepi32_storeu_epi8(base_addr, 0xffff, a);
#else
  store_int8x16(base_addr, cvt_uint32x16_uint8x16(a));
#endif
}

inline void mask_cvtuint32x16_store_int8x16(void* base_addr, int mask, int32x16 a) {
#ifdef __AVX512F__
  _mm512_mask_cvtusepi32_storeu_epi8(base_addr, mask, a);
#else
  mask_store_int8x16(base_addr, mask, maskz_cvt_uint32x16_uint8x16(mask, a));
#endif
}
