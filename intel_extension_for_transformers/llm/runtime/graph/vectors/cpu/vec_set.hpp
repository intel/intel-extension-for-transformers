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

inline fp32x16 set1_fp32x16(const float x);

inline int32x16 set1_int8x16(const int8_t x);

inline int32x16 set1_int16x16(const int16_t x);

inline int32x16 set1_fp16x16(const uint16_t x);

inline int32x16 set1_int32x16(const int16_t x);

inline int32x16 setzero_int32x16();

inline fp32x16 setzero_fp32x16();

#endif  // ENGINE_EXECUTOR_INCLUDE_VEC_SET_HPP_
