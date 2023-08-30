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

#ifndef ENGINE_EXECUTOR_INCLUDE_VEC_ARITHMETIC_HPP_
#define ENGINE_EXECUTOR_INCLUDE_VEC_ARITHMETIC_HPP_

#include "vec_base.hpp"

inline fp32x16 sub_fp32x16(fp32x16 x, fp32x16 y);
REGISTER_KERNEL_T(sub_fp32x16, fp32x16, fp32x16, fp32x16);

inline fp32x16 fmsub_fp32x16(fp32x16 x, fp32x16 y, fp32x16 z);
REGISTER_KERNEL_T(fmsub_fp32x16, fp32x16, fp32x16, fp32x16, fp32x16);

inline fp32x16 maskz_fmsub_fp32x16(int mask, fp32x16 x, fp32x16 y, fp32x16 z);

inline fp32x16 add_fp32x16(fp32x16 x, fp32x16 y);
REGISTER_KERNEL_T(add_fp32x16, fp32x16, fp32x16, fp32x16);

inline fp32x16 fmadd_fp32x16(fp32x16 x, fp32x16 y, fp32x16 z);
REGISTER_KERNEL_T(fmadd_fp32x16, fp32x16, fp32x16, fp32x16, fp32x16);

inline fp32x16 mul_fp32x16(fp32x16 x, fp32x16 y);
REGISTER_KERNEL_T(mul_fp32x16, fp32x16, fp32x16, fp32x16);

inline fp32x16 maskz_mul_fp32x16(int mask, fp32x16 x, fp32x16 y);

template <int rounding>
inline fp32x16 mul_round_fp32x16(fp32x16 x, fp32x16 y);

inline fp32x16 div_fp32x16(fp32x16 x, fp32x16 y);
REGISTER_KERNEL_T(div_fp32x16, fp32x16, fp32x16, fp32x16);

inline float reduce_add_fp32x16(fp32x16 x);
REGISTER_KERNEL_T(reduce_add_fp32x16, float, fp32x16);

inline fp32x16 sqrt_fp32x16(fp32x16 x);
REGISTER_KERNEL_T(sqrt_fp32x16, fp32x16, fp32x16);

inline fp32x16 rsqrt14_fp32x16(fp32x16 x);
REGISTER_KERNEL_T(rsqrt14_fp32x16, fp32x16, fp32x16);

inline fp32x16 ceil_fp32x16(fp32x16 x);
REGISTER_KERNEL_T(ceil_fp32x16, fp32x16, fp32x16);

inline fp32x16 scale_fp32x16(fp32x16 x, fp32x16 y);
REGISTER_KERNEL_T(scale_fp32x16, fp32x16, fp32x16, fp32x16);

inline float dot_fp32x16(fp32x16 x, fp32x16 y);
REGISTER_KERNEL_T(dot_fp32x16, float, fp32x16, fp32x16);

inline fp32x16 abs_fp32x16(fp32x16 x);
REGISTER_KERNEL_T(abs_fp32x16, fp32x16, fp32x16);

#endif  // ENGINE_EXECUTOR_INCLUDE_VEC_SET_HPP_
