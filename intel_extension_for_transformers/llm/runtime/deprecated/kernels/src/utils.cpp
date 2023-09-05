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

#include "utils.hpp"

#include <glog/logging.h>

#include "data_type/data_types.hpp"

namespace jd {
int8_t fp32_to_int8(const float fp32, const float scale, const float zp) {
  int32_t int32 = nearbyint(fp32 / scale + zp);
  int32 = int32 < -128 ? -128 : int32;
  int32 = int32 > 127 ? 127 : int32;
  return static_cast<int8_t>(int32);
}

float int8_to_fp32(const int8_t int8, const float scale, const float zp) {
  float fp32 = static_cast<float>(static_cast<int>(int8));
  return fp32 * scale - zp;
}

float time(const std::string& state) {
  static auto time_axis = std::chrono::microseconds();
  std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
  std::chrono::system_clock::duration dur = tp.time_since_epoch();
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(dur);
  if (state == "start") {
    time_axis = micros;
    return 0;
  } else if (state == "end") {
    return (micros.count() - time_axis.count()) / 1e3;  // Displayed in milliseconds.
  }
  return -1;
}

template <typename T>
T str_to_num(const std::string& s) {
  return static_cast<T>(atof(s.c_str()));
}
template float str_to_num<float>(const std::string&);
template int str_to_num<int>(const std::string&);
template int64_t str_to_num<int64_t>(const std::string&);
template uint64_t str_to_num<uint64_t>(const std::string&);

template <typename T>
std::vector<T> split_str(const std::string& s, const char& delim) {
  std::stringstream ss(s);
  std::string temp;
  std::vector<T> ans;
  while (std::getline(ss, temp, delim)) {
    if (!temp.empty()) {
      ans.push_back(str_to_num<T>(temp));
    }
  }
  return ans;
}

template <>
std::vector<std::string> split_str<std::string>(const std::string& s, const char& delim) {
  std::stringstream ss(s);
  std::string temp;
  std::vector<std::string> ans;
  while (std::getline(ss, temp, delim))
    if (!temp.empty()) ans.push_back(temp);
  return ans;
}

template std::vector<int64_t> split_str<int64_t>(const std::string&, const char&);
template std::vector<int> split_str<int>(const std::string&, const char&);

template <typename T>
inline bool all_zeros(const T* data, dim_t ld, dim_t nd1, dim_t nd2) {
  for (dim_t i = 0; i < nd1; i++) {
    for (dim_t j = 0; j < nd2; j++) {
      if (data[i * ld + j] != 0) return false;
    }
  }
  return true;
}
template bool all_zeros<float>(const float*, dim_t, dim_t, dim_t);

int get_data_size(data_type dt) {
  if (type_size.find(dt) != type_size.end()) {
    return type_size.at(dt);
  } else {
    SPARSE_LOG(ERROR) << "unsupported data type.";
    return 1;
  }
}

float get_exp(float x) {
  static const auto fmax = bit_cast<float>(0x42b17218);
  static const auto fmin = bit_cast<float>(0xc2aeac50);
  if (x < fmin) x = fmin;
  return x > fmax ? INFINITY : expf(x);
}

// todo:add a erf_gelu version.
float get_gelu(float x) {
  // an approximate fitting function of GELU(x)
  // GELU(x)≈0.5x(1+tanh[(2/pi)^0.5)*(x+0.044715x^3)]
  // for more details,pls refer this paper:https://arxiv.org/abs/1606.08415
  return 0.5 * x * (1 + tanhf(0.7978845834732056 * (x + 0.044714998453855515 * x * x * x)));
}

float get_relu(float x, float alpha) { return x > 0 ? x : alpha * x; }

int get_quantize(float x, float alpha, float scale, data_type dt) {
  x /= scale;
  x += alpha;
  int ans = std::nearbyint(x);

  if (dt == data_type::s8) {
    ans = ans > 127 ? 127 : ans;
    ans = ans < -128 ? -128 : ans;
  }

  if (dt == data_type::u8) {
    ans = ans > 255 ? 255 : ans;
    ans = ans < 0 ? 0 : ans;
  }
  return ans;
}

float get_dequantize(float x, float alpha, float scale) {
  x -= alpha;
  x *= scale;
  return x;
}

float get_linear(float x, float aplha, float beta) { return x * aplha + beta; }

float get_swish(float x, float alpha) { return x / (1.f + get_exp(-1 * alpha * x)); }

float apply_postop_list(float value, const std::vector<postop_attr>& attrs) {
  for (auto&& i : attrs) {
    if (i.op_type == postop_type::eltwise) {
      if (i.op_alg == postop_alg::exp) value = get_exp(value);
      if (i.op_alg == postop_alg::gelu) value = get_gelu(value);
      if (i.op_alg == postop_alg::relu) value = get_relu(value, i.alpha);
      if (i.op_alg == postop_alg::quantize) value = get_quantize(value, i.alpha, i.scale, i.dt);
      if (i.op_alg == postop_alg::dequantize) value = get_dequantize(value, i.alpha, i.scale);
      if (i.op_alg == postop_alg::tanh) value = tanh(value);
      if (i.op_alg == postop_alg::linear) value = get_linear(value, i.alpha, i.beta);
      if (i.op_alg == postop_alg::swish) value = get_swish(value, i.alpha);
      if (i.op_alg == postop_alg::eltop_int_lut) continue;
    } else {
      SPARSE_LOG(ERROR) << "unsupported postop type.";
    }
  }
  return value;
}

template <typename T>
bool set_once_before_first_get_setting_t<T>::set(T new_value) {
  if (state_.load() == locked) return false;

  while (true) {
    unsigned expected = idle;
    if (state_.compare_exchange_weak(expected, busy_setting)) break;
    if (expected == locked) return false;
  }

  value_ = new_value;
  state_.store(locked);
  return true;
}

template <typename T>
void cast_to_float_array(const void* src, std::vector<float>* dst, int size) {
  T* src_typed = reinterpret_cast<T*>(const_cast<void*>(src));
  for (int i = 0; i < size; ++i) {
    (*dst)[i] = static_cast<float>(src_typed[i]);
  }
}

template void cast_to_float_array<float>(const void*, std::vector<float>*, int);
template void cast_to_float_array<int>(const void*, std::vector<float>*, int);
template void cast_to_float_array<int8_t>(const void*, std::vector<float>*, int);
template void cast_to_float_array<uint8_t>(const void*, std::vector<float>*, int);
template void cast_to_float_array<bfloat16_t>(const void*, std::vector<float>*, int);

template <typename T>
void cast_from_float_array(const std::vector<float>& src, void* dst, int size) {
  T* dst_typed = reinterpret_cast<T*>(dst);
  for (int i = 0; i < size; ++i) {
    dst_typed[i] = cast_to<T>(src[i]);
  }
}

template void cast_from_float_array<float>(const std::vector<float>&, void*, int);
template void cast_from_float_array<int>(const std::vector<float>&, void*, int);
template void cast_from_float_array<int8_t>(const std::vector<float>&, void*, int);
template void cast_from_float_array<uint8_t>(const std::vector<float>&, void*, int);
template void cast_from_float_array<bfloat16_t>(const std::vector<float>&, void*, int);

}  // namespace jd
