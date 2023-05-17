//  Copyright (c) 2021 Intel Corporation
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

#ifndef ENGINE_EXECUTOR_INCLUDE_WEIGHT_COMPRESSION_HPP_
#define ENGINE_EXECUTOR_INCLUDE_WEIGHT_COMPRESSION_HPP_

#include <stdint.h>
#include <type_traits>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <vector>
#include "param_types.hpp"
#include "data_type/data_types.hpp"

struct weight_compression {
  jd::data_type type_ = jd::data_type::undef;
  std::vector<uint8_t> mem_;
  bool enabled() { return type_ != jd::data_type::undef; }
  bool valid = false;
  float scale_ = 1.f;
  std::vector<float> scales_;
  static int constexpr PreferedM = 4;
};

template <typename T>
struct float8_auto_scale {
  static constexpr bool is_e4m3 = std::is_same<T, jd::float8_e4m3_t>::value;
  static constexpr float EnlargeTarget = is_e4m3 ? 256.f : 65536.f;

  static void auto_scale_T_bf16(const jd::bfloat16_t* _src, T* _dst, size_t _n, size_t _k, float* _scale) {
    std::vector<float> minvals(_n, std::numeric_limits<float>::max()), maxvals(_n, std::numeric_limits<float>::min());
#pragma omp parallel for
    for (int i = 0; i < _n; i++) {
      for (size_t j = 0; j < _k; j++) {
        float fval = _src[i * _k + j];
        if (fval < minvals[i]) {
          minvals[i] = fval;
        }
        if (fval > maxvals[i]) {
          maxvals[i] = fval;
        }
      }
    }
    float maxabsval = -1.f;
    for (size_t i = 0; i < _n; i++) {
      maxabsval = std::max(maxabsval, std::abs(minvals[i]));
      maxabsval = std::max(maxabsval, std::abs(maxvals[i]));
    }

    float SCALE = EnlargeTarget / maxabsval;
#pragma omp parallel for
    for (int i = 0; i < _n * _k; i++) {
      _dst[i] = static_cast<float>(_src[i]) * SCALE;
    }
    *_scale = 1.f / SCALE;
  }
};

struct int8_quantize {
  static void quantize_T_bf16(const jd::bfloat16_t* _src, int8_t* _dst, size_t _size, float* _scale) {
    float minvalue = std::numeric_limits<float>::max(), maxvalue = std::numeric_limits<float>::min();
    for (size_t i = 0; i < _size; i++) {
      float fval = _src[i];
      if (fval < minvalue) {
        minvalue = fval;
      }
      if (fval > maxvalue) {
        maxvalue = fval;
      }
    }
    float maxabsval = std::abs(minvalue);
    maxabsval = std::max(maxabsval, std::abs(maxvalue));
    float SCALE = 127 / maxabsval;

#pragma omp parallel for
    for (int i = 0; i < _size; i++) {
      float tmp = static_cast<float>(_src[i]) * SCALE + 0.5f;
      tmp = tmp > 127 ? 127 : tmp;
      tmp = tmp < -128 ? -128 : tmp;
      _dst[i] = int8_t(tmp);
    }

    *_scale = 1 / SCALE;
  }

  static void quantize_T_bf16_percn(const jd::bfloat16_t* _src, int8_t* _dst, size_t _n, size_t _k, float* _scales) {
    std::vector<float> minvals(_n, std::numeric_limits<float>::max()), maxvals(_n, std::numeric_limits<float>::min());
    std::vector<float> maxabsvals(_n);
#pragma omp parallel for
    for (int i = 0; i < _n; i++) {
      for (size_t j = 0; j < _k; j++) {
        float fval = static_cast<float>(_src[i * _k + j]);
        if (fval < minvals[i]) {
          minvals[i] = fval;
        }
        if (fval > maxvals[i]) {
          maxvals[i] = fval;
        }
      }
      maxabsvals[i] = std::max(std::abs(minvals[i]), std::abs(maxvals[i]));
      float SCALE = 127 / maxabsvals[i];
      for (size_t j = 0; j < _k; j++) {
        float tmp = static_cast<float>(_src[i * _k + j]) * SCALE + 0.5f;
        tmp = tmp > 127 ? 127 : tmp;
        tmp = tmp < -128 ? -128 : tmp;
        _dst[i * _k + j] = int8_t(tmp);
      }
      _scales[i] = 1 / SCALE;
    }
  }
};

struct weight_compression_context {
  jd::data_type global_type_;
  weight_compression_context() {
    global_type_ = jd::data_type::undef;
    auto envstr = std::getenv("NE_WEIGHT_FP8_4E3M");
    if (envstr != NULL && atoi(envstr) > 0) {
      global_type_ = jd::data_type::f8_e4m3;
    }
    envstr = std::getenv("NE_WEIGHT_FP8_5E2M");
    if (envstr != NULL && atoi(envstr) > 0) {
      global_type_ = jd::data_type::f8_e5m2;
    }
    envstr = std::getenv("NE_WEIGHT_INT8");
    if (envstr != NULL && atoi(envstr) > 0) {
      global_type_ = jd::data_type::s8;
    }
    envstr = std::getenv("NE_WEIGHT_INT4");
    if (envstr != NULL && atoi(envstr) > 0) {
      global_type_ = jd::data_type::s4;
    }
  }
  static weight_compression_context* get_instance();
};
#endif  // ENGINE_EXECUTOR_INCLUDE_WEIGHT_COMPRESSION_HPP_
