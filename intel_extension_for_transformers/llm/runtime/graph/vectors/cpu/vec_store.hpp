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

inline void store_int8x16(void* mem_addr, int8x16 a);
template <>
void store_kernel_t<int8x16>(void* dst, int8x16 src) {
  store_int8x16(dst, src);
}

inline void mask_store_int8x16(void* mem_addr, const int mask, int8x16 a);

inline void store_fp32x16(void* mem_addr, fp32x16 a);
template <>
void store_kernel_t<float>(void* dst, float src) {
  float* dst_fp32 = reinterpret_cast<float*>(dst);
  *dst_fp32 = src;
}

template <>
void store_kernel_t<fp32x16>(void* dst, fp32x16 src) {
  store_fp32x16(dst, src);
}

inline void store_bf16x16(void* mem_addr, bf16x16 a);
template <>
void store_kernel_t<bf16x16>(void* dst, bf16x16 src) {
  store_bf16x16(dst, src);
}

inline void cvtuint32x16_store_int8x16(void* base_addr, int32x16 a);

inline void mask_cvtuint32x16_store_int8x16(void* base_addr, int mask, int32x16 a);
#endif  // ENGINE_EXECUTOR_INCLUDE_VEC_STORE_HPP_
