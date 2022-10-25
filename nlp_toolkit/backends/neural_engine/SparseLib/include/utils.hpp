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

#ifndef ENGINE_SPARSELIB_INCLUDE_UTILS_HPP_
#define ENGINE_SPARSELIB_INCLUDE_UTILS_HPP_
#include <glog/logging.h>
#include <stdlib.h>

#include <atomic>
#include <algorithm>
#include <chrono>  // NOLINT
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "param_types.hpp"

#ifdef SPARSE_LIB_USE_VTUNE
#include <ittnotify.h>
#endif

#define SPARSE_LOG(level) LOG(level) << "Log from Sparselib \n"
#define SPARSE_LOG_IF(level, f) LOG_IF(level, f) << "Log from Sparselib\n"
namespace jd {

typedef uint16_t bfloat16_t;  // NOLINT
typedef int64_t dim_t;

template <typename src_t, typename dst_t>
dst_t cast_to(src_t x);

float make_fp32(bfloat16_t x);

bfloat16_t make_bf16(float x);

template <typename T>
void init_vector(T* v, int num_size, float bound1 = -10, float bound2 = 10, int seed = 5489u);

template <typename T>
bool compare_data(const void* buf1, int64_t size1, const void* buf2, int64_t size2, float eps = 1e-6);

float time(const std::string& state);

template <typename value_type, typename predicate_type>
inline bool is_any_of(std::initializer_list<value_type> il, predicate_type pred) {
  return std::any_of(il.begin(), il.end(), pred);
}

#define ceil_div(x, y) (((x) + (y)-1) / (y))

#define is_nonzero(x) (fabs((x)) > (1e-3))

template <typename T>
T str_to_num(const std::string& s);

template <typename T>
std::vector<T> split_str(const std::string& s, const char& delim = ',');

std::string join_str(const std::vector<std::string>& ss, const std::string& delim = ",");

/**
 * @brief Check if every element in a sub matrix is zero
 *
 * @tparam T Element type of the matrix data
 * @param data pointer to the start element of th sub matrix to check
 * @param ld leading dim of the data
 * @param nd1 size in the major dimension
 * @param nd2 size in another dimension
 */
template <typename T>
bool all_zeros(const T* data, dim_t ld, dim_t nd1, dim_t nd2);

int get_data_size(data_type dt);

float get_exp(float x);
float get_linear(float x);
float get_gelu(float x);
float get_relu(float x, float alpha);
int get_quantize(float x, float alpha, float scale, data_type dt);
float get_dequantize(float x, float alpha, float scale);
float apply_postop_list(float value, const std::vector<jd::postop_attr>& attrs);

// A setting (basically a value) that can be set() multiple times until the
// time first time the get() method is called. The set() method is expected to
// be as expensive as a busy-waiting spinlock. The get() method is expected to
// be asymptotically as expensive as a single lock-prefixed memory read. The
// get() method also has a 'soft' mode when the setting is not locked for
// re-setting. This is used for testing purposes.
template <typename T>
struct set_once_before_first_get_setting_t {
 private:
  T value_;
  std::atomic<unsigned> state_;
  enum : unsigned { idle = 0, busy_setting = 1, locked = 2 };

 public:
  explicit set_once_before_first_get_setting_t(T init) : value_{init}, state_{idle} {}

  bool set(T new_value);

  T get(bool soft = false) {
    if (!soft && state_.load() != locked) {
      while (true) {
        unsigned expected = idle;
        if (state_.compare_exchange_weak(expected, locked)) break;
        if (expected == locked) break;
      }
    }
    return value_;
  }
};

template <typename T>
void cast_to_float_array(const void* src, std::vector<float>* dst, int size);

template <typename T>
void cast_from_float_array(std::vector<float> src, void* dst, int size);

}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_UTILS_HPP_
