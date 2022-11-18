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

#include <glog/logging.h>
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

inline int8_t fp32_2_s8(float val, float scale = 1.f) {
  int32_t res = nearbyint(val * scale);
  return static_cast<int8_t>(res < -128 ? -128 : res > 127 ? 127 : res);
}

inline float bf16_2_fp32(jd::bfloat16_t bf16_val) {
  unsigned int ans = bf16_val << 16;
  return *reinterpret_cast<float*>(&ans);
}

inline jd::data_type str_2_dt(std::string str) {
  if (str == "fp32") return jd::data_type::fp32;
  if (str == "fp16") return jd::data_type::fp16;
  if (str == "bf16") return jd::data_type::bf16;
  if (str == "s8") return jd::data_type::s8;
  if (str == "u8") return jd::data_type::u8;
  LOG(ERROR) << "sparselib do not support " + str + " dt now." << std::endl;
  return jd::data_type::undef;
}

int get_data_width(jd::data_type dtype);

void assign_val(void* ptr, jd::data_type dtype, float val, int idx);

int get_element_num(const jd::operator_desc& op_desc, int idx = 0);

namespace jd {

std::vector<postop_attr> get_postop_attr(const char* postop_str, data_type* dt_ptr);

}  // namespace jd

#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_COMMON_UTILS_HPP_
