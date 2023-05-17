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
#include "common_utils.hpp"
#include <algorithm>
#include <limits>

namespace bench {
template <typename T>
void init_vector(T* v, int num_size, float range1, float range2, int seed) {
  float low_value = std::max(range1, static_cast<float>(std::numeric_limits<T>::lowest()) + 1);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> u(low_value, range2);
  for (int i = 0; i < num_size; ++i) {
    v[i] = u(gen);
  }
}
#define DECLARE_INIT_VECTOR(type) template void init_vector<type>(type*, int, float, float, int);

DECLARE_INIT_VECTOR(float)
DECLARE_INIT_VECTOR(int32_t)
DECLARE_INIT_VECTOR(uint8_t)
DECLARE_INIT_VECTOR(int8_t)
DECLARE_INIT_VECTOR(jd::float8_e4m3_t)
DECLARE_INIT_VECTOR(jd::float8_e5m2_t)

template <>
void init_vector<jd::bfloat16_t>(jd::bfloat16_t* v, int num_size, float range1, float range2, int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> u(range1, range2);
  for (int i = 0; i < num_size; ++i) {
    v[i] = u(gen);
  }
}

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

int get_data_width(jd::data_type dtype) {
  int data_width = 0;
  switch (dtype) {
    case jd::data_type::fp32:
      data_width = 4;
      break;
    case jd::data_type::bf16:
      data_width = 2;
      break;
    case jd::data_type::u8:
    case jd::data_type::s8:
      data_width = 1;
      break;
    default:
      throw std::runtime_error(std::string("sparselib_ut_malloc error:unsupport data type."));
      break;
  }
  return data_width;
}
void assign_val(void* ptr, jd::data_type dtype, float val, int idx) {
  switch (dtype) {
    case jd::data_type::fp32:
      *(reinterpret_cast<float*>(ptr) + idx) = val;
      break;
    case jd::data_type::bf16:
      *(reinterpret_cast<jd::bfloat16_t*>(ptr) + idx) = val;
      break;
    case jd::data_type::u8:
      *(reinterpret_cast<uint8_t*>(ptr) + idx) = static_cast<uint8_t>(val);
      break;
    case jd::data_type::s8:
      *(reinterpret_cast<int8_t*>(ptr) + idx) = static_cast<int8_t>(val);
      break;
    default:
      throw std::runtime_error(std::string("assign_val:unsupport this dtype."));
  }
}
int get_element_num(const jd::operator_desc& op_desc, int idx) {
  auto ts_descs = op_desc.tensor_descs();
  if (static_cast<size_t>(idx) >= ts_descs.size()) {
    LOG(ERROR) << "idx out of range";
    return 0;
  }
  int num = 1;
  for (auto&& i : ts_descs[idx].shape()) num *= i;
  return num;
}
std::vector<jd::postop_attr> get_postop_attr(const char* postop_str, jd::data_type* dt_ptr) {
  std::istringstream postop_stream(postop_str);
  std::vector<jd::postop_attr> postop_attrs(0);
  while (!postop_stream.eof()) {
    std::string postop_str;
    std::getline(postop_stream, postop_str, '+');
    size_t sep_idx = postop_str.find("_");
    if (sep_idx == postop_str.npos) {  // quantize or dequantize
      if (!strcmp(postop_str.c_str(), "quantize")) {
        postop_attrs.emplace_back(jd::data_type::u8, jd::postop_type::eltwise, jd::postop_alg::quantize, 0, 0, 2);
      } else if (!strcmp(postop_str.c_str(), "dequantize")) {
        postop_attrs.emplace_back(jd::data_type::u8, jd::postop_type::eltwise, jd::postop_alg::dequantize, 0, 0, 2);
      } else {
        LOG(ERROR) << postop_str << " is not supported";
      }
    } else {  // other algorithms
      // get jd::data_type
      const std::string data_type_str = postop_str.substr(0, sep_idx);
      if (data_type_str == "fp32") {
        *dt_ptr = jd::data_type::fp32;
      } else if (data_type_str == "bf16") {
        *dt_ptr = jd::data_type::bf16;
      } else {
        LOG(ERROR) << "Unsupported data type: " << data_type_str;
        continue;
      }
      // get algorithm
      jd::postop_alg alg;
      const std::string alg_str = postop_str.substr(sep_idx + 1);
      if (alg_str == "exp") {
        alg = jd::postop_alg::exp;
      } else if (alg_str == "gelu") {
        alg = jd::postop_alg::gelu;
      } else if (alg_str == "relu") {
        alg = jd::postop_alg::relu;
      } else if (alg_str == "tanh") {
        alg = jd::postop_alg::tanh;
      } else {
        LOG(ERROR) << "Unsupported algorithm: " << alg_str;
        continue;
      }
      postop_attrs.emplace_back(*dt_ptr, jd::postop_type::eltwise, alg);
    }
  }
  return postop_attrs;
}

template <typename T2, typename T1>
inline const T2 bit_cast(T1 i) {
  static_assert(sizeof(T1) == sizeof(T2), "Bit-casting must preserve size.");
  T2 o;
  memcpy(&o, &i, sizeof(T2));
  return o;
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

int get_quantize(float x, float alpha, float scale, const jd::data_type& data_type) {
  x /= scale;
  x += alpha;
  int ans = std::nearbyint(x);

  if (data_type == jd::data_type::s8) {
    ans = ans > 127 ? 127 : ans;
    ans = ans < -128 ? -128 : ans;
  }

  if (data_type == jd::data_type::u8) {
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

}  // namespace bench
