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

#include <sys/mman.h>
#include <sys/signal.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <glog/logging.h>
#include <iostream>

namespace jd {
template <typename T>
T cast_to(float x) {
  return static_cast<T>(x);
}

template <>
bfloat16_t cast_to(float x) {
  return make_bf16(x);
}

float make_fp32(bfloat16_t x) {
  unsigned int y = x;
  y = y << 16;
  float* res = reinterpret_cast<float*>(&y);
  return *res;
}

bfloat16_t make_bf16(float x) {
  int* res = reinterpret_cast<int*>(&x);
  *res = *res >> 16;
  return (bfloat16_t)*res;
}

template <typename T>
void init_vector(T* v, int num_size, float range1, float range2, int seed) {
  float low_value = std::max(range1, static_cast<float>(std::numeric_limits<T>::lowest()) + 1);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> u(low_value, range2);
  for (int i = 0; i < num_size; ++i) {
    v[i] = cast_to<T>(u(gen));
  }
}
template void init_vector<float>(float*, int, float, float, int);
template void init_vector<int32_t>(int32_t*, int, float, float, int);
template void init_vector<uint8_t>(uint8_t*, int, float, float, int);
template void init_vector<int8_t>(int8_t*, int, float, float, int);
template void init_vector<bfloat16_t>(bfloat16_t*, int, float, float, int);

template <typename T>
bool compare_data(const void* buf1, int64_t size1, const void* buf2, int64_t size2, T eps) {
  if (buf1 == buf2) {
    return false;
  }
  if (size1 != size2) {
    return false;
  }
  const auto& buf1_data = static_cast<const T*>(buf1);
  const auto& buf2_data = static_cast<const T*>(buf2);
  for (int64_t i = 0; i < size1; ++i) {
    auto err = fabs(cast_to<float>(buf1_data[i]) - cast_to<float>(buf2_data[i])) /
               std::max(fabs(cast_to<float>(buf2_data[i])), 1.0);
    if (err > eps) {
      LOG(ERROR) << cast_to<float>(buf1_data[i]) << "vs" << cast_to<float>(buf2_data[i]) << " idx=" << i << std::endl;
      return false;
    }
  }
  return true;
}
template bool compare_data<float>(const void*, int64_t, const void*, int64_t, float);
template bool compare_data<int32_t>(const void*, int64_t, const void*, int64_t, int32_t);
template bool compare_data<uint8_t>(const void*, int64_t, const void*, int64_t, uint8_t);
template bool compare_data<int8_t>(const void*, int64_t, const void*, int64_t, int8_t);
template bool compare_data<bfloat16_t>(const void*, int64_t, const void*, int64_t, bfloat16_t);

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
template std::vector<std::string> split_str<std::string>(const std::string&, const char&);

std::string join_str(const std::vector<std::string>& ss, const std::string& delim) {
  std::string ans;
  for (size_t i = 0; i < ss.size(); ++i) {
    if (i != 0) ans += delim;
    ans += ss[i];
  }
  return ans;
}

bool init_amx() {
#ifdef SPARSE_LIB_USE_AMX

#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

  unsigned long bitmask = 0;                                             // NOLINT
  long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);  // NOLINT
  if (0 != status) return false;
  if (bitmask & XFEATURE_MASK_XTILEDATA) return true;

  status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  if (0 != status)
    return false;  // XFEATURE_XTILEDATA setup is failed, TMUL usage is not
                   // allowed
  status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);

  // XFEATURE_XTILEDATA setup is failed, can't use TMUL
  if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA)) return false;

  // XFEATURE_XTILEDATA set successfully, TMUL usage is allowed

  return true;
#else
  return false;
#endif
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
  switch (dt) {
    case data_type::fp32:
    case data_type::s32:
      return 4;
    case data_type::bf16:
    case data_type::fp16:
      return 2;
    case data_type::s8:
    case data_type::u8:
      return 1;
    default:
      std::runtime_error("unsupported data type.");
  }
}

float get_exp(float x) {
  unsigned int max = 0x42b17218;
  unsigned int min = 0xc2aeac50;
  float fmax = *reinterpret_cast<float*>(&max);
  float fmin = *reinterpret_cast<float*>(&min);
  if (x > fmax) {
    return INFINITY;
  } else if (x < fmin) {
    return 0;
  } else {
    return expf(x);
  }
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
      if (i.op_alg == jd::postop_alg::eltop_int_lut) continue;
    } else {
      std::runtime_error("do not support postop type.");
    }
  }
  return value;
}

}  // namespace jd
