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

#ifndef ENGINE_EXECUTOR_INCLUDE_VEC_SET_HPP_
#define ENGINE_EXECUTOR_INCLUDE_VEC_SET_HPP_

#include "vec_base.hpp"

fp32x16 set1_fp32x16(const float x);
REGISTER_KERNEL_T(set1_fp32x16, fp32x16, float);

s8x16 set1_s8x16(const int8_t x);
REGISTER_KERNEL_T(set1_s8x16, s8x16, int8_t);

s16x16 set1_s16x16(const int16_t x);
REGISTER_KERNEL_T(set1_s16x16, s16x16, int16_t);

fp16x16 set1_fp16x16(const uint16_t x);
REGISTER_KERNEL_T(set1_fp16x16, fp16x16, uint16_t);

s32x16 set1_s32x16(const int32_t x);
REGISTER_KERNEL_T(set1_s32x16, s32x16, int32_t);

s32x16 setzero_s32x16();

fp32x16 setzero_fp32x16();

#endif  // ENGINE_EXECUTOR_INCLUDE_VEC_SET_HPP_
