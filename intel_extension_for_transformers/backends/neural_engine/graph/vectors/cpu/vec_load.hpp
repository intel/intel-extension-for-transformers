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

template <>
float load_kernel_t<float>(const void* src) {
  return *reinterpret_cast<const float*>(src);
}

inline fp32x16 load_fp32x16(void const* mem_addr);
template <>
fp32x16 load_kernel_t<fp32x16>(const void* src) {
  return load_fp32x16(src);
}
inline fp32x16 mask_load_fp32x16(fp32x16 src, int mask, void const* mem_addr);

inline bf16x16 load_bf16x16(void const* mem_addr);
template <>
bf16x16 load_kernel_t<bf16x16>(const void* src) {
  return load_bf16x16(src);
}

inline bf16x16 maskz_load_bf16x16(int mask, void const* mem_addr);

#endif  // ENGINE_EXECUTOR_INCLUDE_VEC_LOAD_HPP_
