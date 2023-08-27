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
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <random>
#include "interface.hpp"
#include "data_type/data_types.hpp"

#define exp_ln_flt_max_f 0x42b17218
#define exp_ln_flt_min_f 0xc2aeac50
#define SPARSE_LOG(level) LOG(level) << "Sparselib] "
#define SPARSE_LOG_IF(level, f) LOG_IF(level, f) << "Sparselib] "
#define SPARSE_DLOG(level) DLOG(level) << "Sparselib] "
#define SPARSE_DLOG_IF(level, f) DLOG_IF(level, f) << "Sparselib] "

#define ceil_div(x, y) (((x) + (y)-1) / (y))
#define pad_to(x, n) (ceil_div(x, n) * (n))

#ifdef _WIN32
#include <malloc.h>
#ifdef aligned_free
#undef aligned_free
#endif

#define aligned_alloc(align, size) _aligned_malloc(size, align)
#define aligned_free _aligned_free
#else
#define aligned_free free

#endif  // _WIN32

namespace bench {
enum memo_mode { MALLOC, MEMSET };
inline float rand_float_postfix() { return rand() / static_cast<float>(RAND_MAX); }
inline jd::data_type str_2_dt(std::string str) {
  for (auto& it : jd::data_type_name) {
    if (it.second == str) {
      return it.first;
    }
  }
  LOG(ERROR) << "sparselib do not support " + str + " dt now." << std::endl;
  return jd::data_type::undef;
}
int get_data_width(jd::data_type dtype);
void assign_val(void* ptr, jd::data_type dtype, float val, int idx);
int get_element_num(const jd::operator_desc& op_desc, int idx = 0);
std::vector<jd::postop_attr> get_postop_attr(const char* postop_str, jd::data_type* dt_ptr);

template <typename T>
std::vector<T> split_str(const std::string& s, const char& delim);

template <typename T>
T str_to_num(const std::string& s) {
  return static_cast<T>(atof(s.c_str()));
}

template <typename T, std::size_t N = 64>
class aligned_allocator_t {
 public:
  typedef T value_type;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;

 public:
  static inline pointer allocate(size_type n, bool zero = false) {
    n = pad_to(n * sizeof(value_type), N);
#ifdef _WIN32
    auto ptr = static_cast<pointer>(_aligned_malloc(n, N));
#else
    auto ptr = static_cast<pointer>(::aligned_alloc(N, n));
#endif

    if (zero) memset(reinterpret_cast<void*>(ptr), 0, n);
    return ptr;
  }
  static inline void deallocate(void* p, size_type = 0) {
#ifdef _WIN32
    _aligned_free(p);
#else
    ::free(p);
#endif
  }
  inline aligned_allocator_t() throw() {}
  template <typename T2>
  inline aligned_allocator_t(const aligned_allocator_t<T2, N>&) throw() {}
  inline ~aligned_allocator_t() throw() {}
  inline pointer adress(reference r) { return &r; }
  inline const_pointer adress(const_reference r) const { return &r; }
  inline void construct(pointer p, const value_type& wert) { new (p) value_type(wert); }
  inline void destroy(pointer p) { p->~value_type(); }
  inline size_type max_size() const throw() { return size_type(-1) / sizeof(value_type); }
  template <typename T2>
  struct rebind {
    typedef aligned_allocator_t<T2, N> other;
  };
  bool operator!=(const aligned_allocator_t<T, N>& other) const { return !(*this == other); }
  // Returns true if and only if storage allocated from *this can be deallocated from other, and vice versa.
  // Always returns true for stateless allocators.
  bool operator==(const aligned_allocator_t<T, N>& other) const { return true; }
};
template <typename T>
void init_vector(T* v, int num_size, float range1 = -10, float range2 = 10, int seed = 5489u);

template <typename T>
void cast_from_float_array(const float* src, void* dst, int size) {
  T* dst_typed = reinterpret_cast<T*>(dst);
  for (int i = 0; i < size; ++i) {
    dst_typed[i] = static_cast<T>(src[i]);
  }
}

template <typename T>
void cast_to_float_array(const void* src, float* dst, int size) {
  T* src_typed = reinterpret_cast<T*>(const_cast<void*>(src));
  for (int i = 0; i < size; ++i) {
    dst[i] = static_cast<float>(src_typed[i]);
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
  return fabs(static_cast<float>(a) - static_cast<float>(b)) /
         std::max(static_cast<float>(fabs(static_cast<float>(b))), 1.0f);
}
template <typename T>
inline typename std::enable_if<s_is_u8s8<T>::value, float>::type get_err(const T& a, const T& b) {
  // for quantized value, error ratio was calcualted with its data range
  return fabs(static_cast<float>(a) - static_cast<float>(b)) / UINT8_MAX;
}

template <typename T>
bool compare_data(const void* buf1, int64_t size1, const void* buf2, int64_t size2, float eps = 1e-6) {
  if (buf1 == buf2 || size1 != size2) return false;
  const auto& buf1_data = static_cast<const T*>(buf1);
  const auto& buf2_data = static_cast<const T*>(buf2);

  for (int64_t i = 0; i < size1; ++i) {
    if (get_err(buf1_data[i], buf2_data[i]) > eps) {
      SPARSE_LOG(ERROR) << static_cast<float>(buf1_data[i]) << "vs" << static_cast<float>(buf2_data[i]) << " idx=" << i;
      return false;
    }
  }
  return true;
}

float get_exp(float x);
// todo:add a erf_gelu version.
float get_gelu(float x);

float get_relu(float x, float alpha);

int get_quantize(float x, float alpha, float scale, const jd::data_type& data_type);
float get_dequantize(float x, float alpha, float scale);

float get_linear(float x, float aplha, float beta);

float get_swish(float x, float alpha);

float apply_postop_list(float value, const std::vector<jd::postop_attr>& attrs);

}  // namespace bench
#endif  // ENGINE_SPARSELIB_BENCH_INCLUDE_COMMON_UTILS_HPP_
