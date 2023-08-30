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

#ifndef ENGINE_EXECUTOR_INCLUDE_VEC_CONVERT_HPP_
#define ENGINE_EXECUTOR_INCLUDE_VEC_CONVERT_HPP_

#include "vec_base.hpp"

template <int rounding = (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)>
inline int32x16 cvt_roundfp32x16_int32x16(fp32x16 a);
template <int rounding>
struct ne_cvt_roundfp32x16_int32x16_kernel_t : public kernel_t<int32x16, fp32x16> {
  ne_cvt_roundfp32x16_int32x16_kernel_t() { func_ = cvt_roundfp32x16_int32x16<rounding>; }
};

template <int rounding = (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)>
inline int32x16 maskz_cvt_roundfp32x16_int32x16(int mask, fp32x16 a);
inline bf16x16 cvt_fp32x16_bf16x16(fp32x16 a);

inline fp32x16 cvt_bf16x16_fp32x16(bf16x16 a);

inline fp32x16 maskz_cvt_bf16x16_fp32x16(int mask, bf16x16 a);

inline int8x16 cvt_uint32x16_uint8x16(int32x16 a);

inline int8x16 maskz_cvt_uint32x16_uint8x16(int mask, int32x16 a);

inline int8x16 cvt_int32x16_int8x16(int32x16 a);
inline int8x16 maskz_cvt_int32x16_int8x16(const int mask, int32x16 a);

#endif  // ENGINE_EXECUTOR_INCLUDE_VEC_CONVERT_HPP_
