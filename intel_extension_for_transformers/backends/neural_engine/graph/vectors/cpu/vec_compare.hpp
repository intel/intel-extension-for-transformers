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

#ifndef ENGINE_EXECUTOR_INCLUDE_VEC_COMPARE_HPP_
#define ENGINE_EXECUTOR_INCLUDE_VEC_COMPARE_HPP_

#include "vec_base.hpp"

inline fp32x16 min_fp32x16(fp32x16 a, fp32x16 b);

inline int32x16 max_int32x16(int32x16 a, int32x16 b);

inline fp32x16 max_fp32x16(fp32x16 a, fp32x16 b);

inline float reduce_max_fp32x16(fp32x16 x);
REGISTER_KERNEL_T(reduce_max_fp32x16, float, fp32x16);

#endif  // ENGINE_EXECUTOR_INCLUDE_VEC_COMPARE_HPP_
