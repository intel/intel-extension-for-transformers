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
#include <omp.h>
#include <stdlib.h>

#include <algorithm>
#include <atomic>
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

#if __GNUC__ >= 4 || defined(__clang__)
#define WITH_GCC_FLAGS
#endif

#ifndef SPARSE_API_
#ifdef _MSC_VER
#if SPARSE_KERNEL_BUILD
#define SPARSE_API_ __declspec(dllexport)
#else
#define SPARSE_API_ __declspec(dllimport)
#endif
#elif __GNUC__ >= 4 || defined(__clang__)
#define SPARSE_API_ __attribute__((visibility("default")))
#endif  // _MSC_VER

#endif  // SPARSE_API_

#ifndef SPARSE_API_
#define SPARSE_API_
#endif  // SPARSE_API_

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

#if __has_cpp_attribute(fallthrough)
#elif defined(WITH_GCC_FLAGS)
#define fallthrough gnu::fallthrough
#endif

#define SPARSE_LOG(level) LOG(level) << "Sparselib] "
#define SPARSE_LOG_IF(level, f) LOG_IF(level, f) << "Sparselib] "
#define SPARSE_DLOG(level) DLOG(level) << "Sparselib] "
#define SPARSE_DLOG_IF(level, f) DLOG_IF(level, f) << "Sparselib] "
namespace jd {

typedef uint16_t bfloat16_t;  // NOLINT
typedef uint8_t float8_t;     // NOLINT

typedef int64_t dim_t;

SPARSE_API_ int8_t fp32_to_int8(const float fp32, const float scale = 1.f, const float zp = 0.f);
SPARSE_API_ float int8_to_fp32(const int8_t int8, const float scale = 1.f, const float zp = 0.f);
SPARSE_API_ uint16_t fp32_to_fp16(const float x);
SPARSE_API_ float fp16_to_fp32(const uint16_t x);
SPARSE_API_ bfloat16_t fp32_to_bf16(const float fp32);
SPARSE_API_ float bf16_to_fp32(const bfloat16_t bf16);

template <typename dst_t, typename src_t>
SPARSE_API_ dst_t cast_to(const src_t x, data_type = data_type::undef);

template <typename T>
SPARSE_API_ void init_vector(T* v, int num_size, float bound1 = -10, float bound2 = 10, int seed = 5489u);

#ifdef _WIN32
#define DECLARE_INIT_VECTOR(type) \
  template SPARSE_API_ void init_vector<type>(type * v, int num_size, float bound1, float bound2, int seed);

DECLARE_INIT_VECTOR(float)
DECLARE_INIT_VECTOR(int)
DECLARE_INIT_VECTOR(uint8_t)
DECLARE_INIT_VECTOR(int8_t)

#undef DECLARE_INIT_VECTOR
#endif

template <>
SPARSE_API_ void init_vector<bfloat16_t>(bfloat16_t* v, int num_size, float range1, float range2, int seed);

template <typename T>
bool SPARSE_API_ compare_data(const void* buf1, int64_t size1, const void* buf2, int64_t size2, float eps = 1e-6);

#ifdef _WIN32
#define DECLARE_COMPARE_DATA(type)                                                                               \
  template SPARSE_API_ bool compare_data<type>(const void* buf1, int64_t size1, const void* buf2, int64_t size2, \
                                               float eps);

DECLARE_COMPARE_DATA(float)
DECLARE_COMPARE_DATA(int)
DECLARE_COMPARE_DATA(uint8_t)
DECLARE_COMPARE_DATA(int8_t)
DECLARE_COMPARE_DATA(uint16_t)

#undef DECLARE_COMPARE_DATA
#endif

template <typename T2, typename T1>
inline const T2 bit_cast(T1 i) {
  static_assert(sizeof(T1) == sizeof(T2), "Bit-casting must preserve size.");
  T2 o;
  memcpy(&o, &i, sizeof(T2));
  return o;
}

float time(const std::string& state);

template <typename value_type, typename predicate_type>
inline bool is_any_of(std::initializer_list<value_type> il, predicate_type pred) {
  return std::any_of(il.begin(), il.end(), pred);
}

template <typename value_type, typename predicate_type>
inline bool is_all_of(std::initializer_list<value_type> il, predicate_type pred) {
  return std::all_of(il.begin(), il.end(), pred);
}

#define ceil_div(x, y) (((x) + (y)-1) / (y))
#define pad_to(x, n) (ceil_div(x, n) * (n))
#define pad_to_le(x, n) (x / n * (n))
#define is_nonzero(x) (fabs((x)) > (1e-3))
#define remainsize(x, size, n) (((x) + (n)) <= (size) ? (n) : ((size) - (x)))
template <typename T>
SPARSE_API_ T str_to_num(const std::string& s);

#ifdef _WIN32
#define DECLARE_STR_TO_NUM(type) template SPARSE_API_ type str_to_num<type>(const std::string& s);

DECLARE_STR_TO_NUM(uint64_t)
DECLARE_STR_TO_NUM(int64_t)
DECLARE_STR_TO_NUM(int)
DECLARE_STR_TO_NUM(float)

#undef DECLARE_STR_TO_NUM
#endif

template <typename T>
SPARSE_API_ std::vector<T> split_str(const std::string& s, const char& delim = ',');

#ifdef _WIN32
#define DECLARE_SPLIT_STR(type) \
  template SPARSE_API_ std::vector<type> split_str<type>(const std::string& s, const char& delim);

DECLARE_SPLIT_STR(int)
// DECLARE_SPLIT_STR(std::string)

#undef DECLARE_SPLIT_STR
#endif

SPARSE_API_ std::string join_str(const std::vector<std::string>& ss, const std::string& delim = ",");

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

int SPARSE_API_ get_data_size(data_type dt);

float SPARSE_API_ get_exp(float x);
float SPARSE_API_ get_linear(float x);
float SPARSE_API_ get_gelu(float x);
float SPARSE_API_ get_relu(float x, float alpha);
int SPARSE_API_ get_quantize(float x, float alpha, float scale, data_type dt);
float SPARSE_API_ get_dequantize(float x, float alpha, float scale);
float SPARSE_API_ apply_postop_list(float value, const std::vector<jd::postop_attr>& attrs);

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
void SPARSE_API_ cast_to_float_array(const void* src, std::vector<float>* dst, int size);

template <typename T>
void SPARSE_API_ cast_from_float_array(const std::vector<float>& src, void* dst, int size);

template <class T>
inline void safe_delete(T*& ptr) {  // NOLINT(runtime/references)
  if (ptr != nullptr) {
    delete ptr;
    ptr = nullptr;
  }
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
#ifdef _WIN32
    auto ptr = static_cast<pointer>(_aligned_malloc(n * sizeof(value_type), N));
#else
    auto ptr = static_cast<pointer>(::aligned_alloc(N, n * sizeof(value_type)));
#endif
    if (zero) memset(ptr, 0, n * sizeof(value_type));
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

/* Prepend ones to vec until the vector length reaches `n`. */
template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
inline std::vector<T> pre_pad1(const size_t n, const std::vector<T> vec) {
  SPARSE_LOG_IF(FATAL, n < vec.size()) << "Vector size to large!";
  std::vector<T> out(n, T{1});
  std::copy(vec.cbegin(), vec.cend(), out.begin() + n - vec.size());
  return out;
}

template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
inline std::vector<T> dim2stride(const std::vector<T> dim) {
  std::vector<T> out(dim.size());
  out[dim.size() - 1] = 1;
  for (int i = dim.size() - 2; i >= 0; --i) out[i] = out[i + 1] * dim[i + 1];
  return out;
}

template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
inline std::vector<T> dim2step(const std::vector<T> dim) {
  auto out = dim2stride(dim);
  for (size_t i = 0; i < dim.size(); ++i)
    if (dim[i] <= 1) out[i] = 0;
  return out;
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
inline std::vector<T> perm_inv(const std::vector<T>& perm) {
  std::vector<T> ret(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) ret[perm[i]] = i;
  return ret;
}

template <typename T, typename U, typename = typename std::enable_if<std::is_integral<U>::value>::type>
std::vector<T> apply_perm(const std::vector<T>& vec, const std::vector<U>& perm) {
  SPARSE_LOG_IF(FATAL, vec.size() != perm.size()) << "Vector must be the same size as perm!";
  std::vector<T> ret(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) ret[perm[i]] = vec[i];
  return ret;
}

inline std::vector<dim_t> dim2stride(const std::vector<dim_t>& dim) {
  std::vector<dim_t> stride(dim.size());
  stride[stride.size() - 1] = 1;
  for (auto i = stride.size() - 1; i > 0; --i) {
    stride[i - 1] = stride[i] * dim[i];
  }
  return stride;
}

class n_thread_t {
 public:
  explicit n_thread_t(int nthr, bool cap = false) : prev_nthr(omp_get_max_threads()) {
    if (nthr > 0 && nthr != prev_nthr && (!cap || nthr < prev_nthr)) omp_set_num_threads(nthr);
  }
  ~n_thread_t() { omp_set_num_threads(prev_nthr); }

 private:
  int prev_nthr;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_UTILS_HPP_
