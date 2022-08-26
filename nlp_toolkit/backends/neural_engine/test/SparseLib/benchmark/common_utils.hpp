//  Copyright (c) 2022 Intel Corporation
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

#ifndef ENGINE_SPARSELIB_BENCH_INCLUDE_COMMON_UTILS_HPP_
#define ENGINE_SPARSELIB_BENCH_INCLUDE_COMMON_UTILS_HPP_

#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <sstream>
#include <set>

#include "interface.hpp"

#define exp_ln_flt_max_f 0x42b17218
#define exp_ln_flt_min_f 0xc2aeac50

enum memo_mode { MALLOC, MEMSET };

inline int uint8_2_int32(uint8_t a) { return static_cast<int>(a); }

inline float rand_float_postfix() { return rand() / static_cast<float>(RAND_MAX); }

inline int8_t fp32_2_s8(float val, float scale) {
  int32_t res = nearbyint(val * scale);
  return static_cast<int8_t>(res < -128 ? -128 : res > 127 ? 127 : res);
}

float get_exp(float x);

float get_gelu(float x);

inline float get_relu(float x, float alpha) { return x > 0 ? x : alpha * x; }

int get_quantize(float x, float alpha, float scale);

float get_dequantize(float x, float alpha, float scale);

int get_data_width(jd::data_type dtype);

float apply_postop_list(float value, const std::vector<jd::postop_attr>& attrs);

void assign_val(void* ptr, jd::data_type dtype, float val, int idx);

void* memo_op(void* ptr, int num, jd::data_type dtype, memo_mode mode);

int get_element_num(const jd::operator_desc& op_desc, int idx = 0);

namespace jd {

std::vector<postop_attr> get_postop_attr(const char* postop_str, data_type* dt_ptr);

}  // namespace jd

#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_COMMON_UTILS_HPP_
