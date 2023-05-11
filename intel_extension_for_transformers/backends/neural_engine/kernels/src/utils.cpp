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
#include <iostream>
#include "fp8.hpp"

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

float fp16_to_fp32(const uint16_t x) {    // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15,
                                          // +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
  const uint32_t e = (x & 0x7C00) >> 10;  // exponent
  const uint32_t m = (x & 0x03FF) << 13;  // mantissa
  // evil log2 bit hack to count leading zeros in denormalized format
  const uint32_t v = bit_cast<uint32_t>(static_cast<float>(m)) >> 23;
  // sign : normalized : denormalized
  return bit_cast<float>((x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
                         ((e == 0) & (m != 0)) * ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000)));
}
uint16_t fp32_to_fp16(const float x) {  // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15,
                                        // +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
  // round-to-nearest-even: add last bit after truncated mantissa
  const uint32_t b = bit_cast<uint32_t>(x) + 0x00001000;
  const uint32_t e = (b & 0x7F800000) >> 23;  // exponent
  // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
  const uint32_t m = b & 0x007FFFFF;
  // sign : normalized : denormalized : saturate
  return (b & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
         ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) | (e > 143) * 0x7FFF;
}

template <typename dst_t, typename src_t>
dst_t SPARSE_API_ cast_to(src_t x, data_type) {
  return static_cast<dst_t>(x);
}

template <>
uint16_t SPARSE_API_ cast_to<uint16_t>(const float x, data_type type) {
  if (type == data_type::fp16) {
    return fp32_to_fp16(x);
  } else {
    return fp32_to_bf16(x);
  }
}

template <>
float SPARSE_API_ cast_to<float>(const uint16_t x, data_type type) {
  if (type == data_type::fp16) {
    return fp16_to_fp32(x);
  } else {
    return bf16_to_fp32(x);
  }
}

template <>
uint8_t SPARSE_API_ cast_to<uint8_t>(const float x, data_type type) {
  if (type == data_type::f8_e5m2) {
    return float8_base<FloatEncoding::E5M2>::convert_float_to_fp8(x);
  } else if (type == data_type::f8_e4m3) {
    return float8_base<FloatEncoding::E4M3>::convert_float_to_fp8(x);
  } else {
    return static_cast<uint8_t>(x);
  }
}

template <>
float SPARSE_API_ cast_to<float>(const uint8_t x, data_type type) {
  if (type == data_type::f8_e5m2) {
    return float8_base<FloatEncoding::E5M2>::convert_fp8_to_float(x);
  } else if (type == data_type::f8_e4m3) {
    return float8_base<FloatEncoding::E4M3>::convert_fp8_to_float(x);
  } else {
    return static_cast<float>(x);
  }
}

// undefined behaviour, https://stackoverflow.com/questions/98650/what-is-the-strict-aliasing-rule
// typedef union
// {
//   float f;
//   unsigned int u;
//   bfloat16_t b[2];
// } union_b;
//
//
// inline float bf16_to_fp32(bfloat16_t x) {
//   union_b tmp;
//   tmp.u = 0;
//   tmp.b[1] = x;
//   return tmp.f;
// }
//
// inline bfloat16_t fp32_to_bf16(float x) {
//   union_b tmp;
//   tmp.f = x;
//   return tmp.b[1];
// }

#ifdef WITH_GCC_FLAGS
#pragma GCC push_options
#pragma GCC optimize "no-strict-aliasing"
#endif
inline float bf16_to_fp32(bfloat16_t x) {
  unsigned int y = static_cast<unsigned int>(x);
  y = y << 16;
  float* res = reinterpret_cast<float*>(&y);
  return *res;
}

inline bfloat16_t fp32_to_bf16(float x) {
  int* res = reinterpret_cast<int*>(&x);
  *res = *res >> 16;
  return (bfloat16_t)*res;
}
#ifdef WITH_GCC_FLAGS
#pragma GCC pop_options
#endif

template <typename T>
void init_vector(T* v, int num_size, float range1, float range2, int seed) {
  float low_value = std::max(range1, static_cast<float>(std::numeric_limits<T>::lowest()) + 1);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> u(low_value, range2);
  for (int i = 0; i < num_size; ++i) {
    v[i] = cast_to<T>(u(gen));
  }
}

#define DECLARE_INIT_VECTOR(type) template void init_vector<type>(type*, int, float, float, int);

DECLARE_INIT_VECTOR(float)
DECLARE_INIT_VECTOR(int32_t)
DECLARE_INIT_VECTOR(uint8_t)
DECLARE_INIT_VECTOR(int8_t)

// Range of bf16 should be almost the same as that of fp32
template <>
void init_vector<bfloat16_t>(bfloat16_t* v, int num_size, float range1, float range2, int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> u(range1, range2);
  for (int i = 0; i < num_size; ++i) {
    v[i] = cast_to<bfloat16_t>(u(gen), data_type::bf16);
  }
}

template <typename T>
struct s_is_u8s8 {
  enum { value = false };
};

template <>
struct s_is_u8s8<int8_t> {
  enum { value = true };
};

template <>
struct s_is_u8s8<uint8_t> {
  enum { value = true };
};

template <typename T>
inline typename std::enable_if<!s_is_u8s8<T>::value, float>::type get_err(const T& a, const T& b) {
  // we compare float relative error ratio here
  return fabs(cast_to<float>(a) - cast_to<float>(b)) / std::max(static_cast<float>(fabs(cast_to<float>(b))), 1.0f);
}
template <typename T>
inline typename std::enable_if<s_is_u8s8<T>::value, float>::type get_err(const T& a, const T& b) {
  // for quantized value, error ratio was calcualted with its data range
  return fabs(cast_to<float>(a) - cast_to<float>(b)) / UINT8_MAX;
}

template <typename T>
bool compare_data(const void* buf1, int64_t size1, const void* buf2, int64_t size2, float eps) {
  if (buf1 == buf2 || size1 != size2) return false;
  const auto& buf1_data = static_cast<const T*>(buf1);
  const auto& buf2_data = static_cast<const T*>(buf2);

  for (int64_t i = 0; i < size1; ++i) {
    if (get_err(buf1_data[i], buf2_data[i]) > eps) {
      SPARSE_LOG(ERROR) << cast_to<float>(buf1_data[i]) << "vs" << cast_to<float>(buf2_data[i]) << " idx=" << i;
      return false;
    }
  }
  return true;
}

#define DECLARE_COMPARE_DATA(type) template bool compare_data<type>(const void*, int64_t, const void*, int64_t, float);

DECLARE_COMPARE_DATA(float)
DECLARE_COMPARE_DATA(int32_t)
DECLARE_COMPARE_DATA(uint8_t)
DECLARE_COMPARE_DATA(uint16_t)
DECLARE_COMPARE_DATA(int8_t)

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
SPARSE_API_ std::vector<T> split_str(const std::string& s, const char& delim) {
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
SPARSE_API_ std::vector<std::string> split_str<std::string>(const std::string& s, const char& delim) {
  std::stringstream ss(s);
  std::string temp;
  std::vector<std::string> ans;
  while (std::getline(ss, temp, delim))
    if (!temp.empty()) ans.push_back(temp);
  return ans;
}

template SPARSE_API_ std::vector<int64_t> split_str<int64_t>(const std::string&, const char&);
template SPARSE_API_ std::vector<int> split_str<int>(const std::string&, const char&);

std::string join_str(const std::vector<std::string>& ss, const std::string& delim) {
  std::string ans;
  for (size_t i = 0; i < ss.size(); ++i) {
    if (i != 0) ans += delim;
    ans += ss[i];
  }
  return ans;
}

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

int get_data_size(jd::data_type dt) {
  // use at() instead of operator[] to raise an exception for invalid type
  return jd::type_size.at(dt);
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

float apply_postop_list(float value, const std::vector<jd::postop_attr>& attrs) {
  for (auto&& i : attrs) {
    if (i.op_type == jd::postop_type::eltwise) {
      if (i.op_alg == jd::postop_alg::exp) value = get_exp(value);
      if (i.op_alg == jd::postop_alg::gelu) value = get_gelu(value);
      if (i.op_alg == jd::postop_alg::relu) value = get_relu(value, i.alpha);
      if (i.op_alg == jd::postop_alg::quantize) value = get_quantize(value, i.alpha, i.scale, i.dt);
      if (i.op_alg == jd::postop_alg::dequantize) value = get_dequantize(value, i.alpha, i.scale);
      if (i.op_alg == jd::postop_alg::tanh) value = tanh(value);
      if (i.op_alg == jd::postop_alg::linear) value = get_linear(value, i.alpha, i.beta);
      if (i.op_alg == jd::postop_alg::swish) value = get_swish(value, i.alpha);
      if (i.op_alg == jd::postop_alg::eltop_int_lut) continue;
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
    (*dst)[i] = cast_to<float>(src_typed[i]);
  }
}
template void SPARSE_API_ cast_to_float_array<float>(const void*, std::vector<float>*, int);
template void SPARSE_API_ cast_to_float_array<int>(const void*, std::vector<float>*, int);
template void SPARSE_API_ cast_to_float_array<int8_t>(const void*, std::vector<float>*, int);
template void SPARSE_API_ cast_to_float_array<uint8_t>(const void*, std::vector<float>*, int);
template void SPARSE_API_ cast_to_float_array<bfloat16_t>(const void*, std::vector<float>*, int);

template <typename T>
void cast_from_float_array(const std::vector<float>& src, void* dst, int size) {
  T* dst_typed = reinterpret_cast<T*>(dst);
  for (int i = 0; i < size; ++i) {
    dst_typed[i] = cast_to<T>(src[i]);
  }
}
template void SPARSE_API_ cast_from_float_array<float>(const std::vector<float>&, void*, int);
template void SPARSE_API_ cast_from_float_array<int>(const std::vector<float>&, void*, int);
template void SPARSE_API_ cast_from_float_array<int8_t>(const std::vector<float>&, void*, int);
template void SPARSE_API_ cast_from_float_array<uint8_t>(const std::vector<float>&, void*, int);
template void SPARSE_API_ cast_from_float_array<bfloat16_t>(const std::vector<float>&, void*, int);

}  // namespace jd
