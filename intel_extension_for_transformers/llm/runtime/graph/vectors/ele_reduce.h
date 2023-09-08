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
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifdef VEC_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef VEC_BUILD
#define VEC_API __declspec(dllexport)
#else
#define VEC_API __declspec(dllimport)
#endif
#else
#define VEC_API __attribute__((visibility("default")))
#endif
#else
#define VEC_API
#endif
VEC_API void ne_vec_norm_f32_(const int n, float* s, const float* x);
VEC_API void ne_vec_sum_f32_(const int n, float* s, const float* x);

VEC_API void ne_vec_sum_ggf_(const int n, double* s, const float* x);

VEC_API void ne_vec_max_f32_(const int n, float* s, const float* x);

VEC_API void ne_vec_norm_inv_f32_(const int n, float* s, const float* x);

#ifdef __cplusplus
}
#endif
