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

#ifndef ENGINE_TEST_GTEST_SPARSELIB_UNIT_TEST_UTILS_HPP_
#define ENGINE_TEST_GTEST_SPARSELIB_UNIT_TEST_UTILS_HPP_

#include <glog/logging.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "data_type/data_types.hpp"
#include "operator_desc.hpp"

#define SPARSE_LOG(level) LOG(level) << "Sparselib] "
#define SPARSE_LOG_IF(level, f) LOG_IF(level, f) << "Sparselib] "
#define SPARSE_DLOG(level) DLOG(level) << "Sparselib] "
#define SPARSE_DLOG_IF(level, f) DLOG_IF(level, f) << "Sparselib] "

#define ceil_div(x, y) (((x) + (y)-1) / (y))

namespace test {
template <typename T>
T str_to_num(const std::string& s) {
  return static_cast<T>(atof(s.c_str()));
}

std::string join_str(const std::vector<std::string>& ss, const std::string& delim) {
  std::string ans;
  for (size_t i = 0; i < ss.size(); ++i) {
    if (i != 0) ans += delim;
    ans += ss[i];
  }
  return ans;
}
enum memo_mode { MALLOC, MEMSET };

float rand_float_postfix() {
  return rand() / static_cast<float>(RAND_MAX);  // NOLINT
}

void assign_val(void* ptr, jd::data_type dtype, float val, int idx) {
  switch (dtype) {
    case jd::data_type::fp32:
      reinterpret_cast<float*>(ptr)[idx] = val;
      break;
    case jd::data_type::bf16:
      reinterpret_cast<jd::bfloat16_t*>(ptr)[idx] = jd::bfloat16_t(val);
      break;
    case jd::data_type::u8:
      reinterpret_cast<uint8_t*>(ptr)[idx] = (uint8_t)val;
      break;
    case jd::data_type::s8:
      reinterpret_cast<int8_t*>(ptr)[idx] = (int8_t)val;
      break;
    default:
      throw std::runtime_error("assign_val: unsupport this dtype.");
  }
}

void* sparselib_ut_memo(void* ptr, int num, jd::data_type dtype, memo_mode mode) {
  int data_width = jd::type_size.at(dtype);
  switch (mode) {
    case MALLOC:
      ptr = malloc(num * data_width);
      break;
    case MEMSET:
      std::memset(ptr, 0, num * data_width);
      break;
    default:
      break;
  }
  return ptr;
}

int get_element_num(const jd::operator_desc& op_desc) {
  auto ts_descs = op_desc.tensor_descs();
  int num = 1;
  for (auto&& i : ts_descs[0].shape()) num *= i;
  return num;
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
bool compare_data(const void* buf1, dim_t size1, const void* buf2, dim_t size2, float eps = 1e-6) {
  if (buf1 == buf2 || size1 != size2) return false;
  const auto& buf1_data = static_cast<const T*>(buf1);
  const auto& buf2_data = static_cast<const T*>(buf2);

  for (dim_t i = 0; i < size1; ++i) {
    if (get_err(buf1_data[i], buf2_data[i]) > eps) {
      SPARSE_LOG(ERROR) << static_cast<float>(buf1_data[i]) << "vs" << static_cast<float>(buf2_data[i]) << " idx=" << i;
      return false;
    }
  }
  return true;
}

template <typename T>
void init_vector(T* v, int num_size, float range1 = -10, float range2 = 10, int seed = 5489u) {
  float low_value = std::max(range1, static_cast<float>(std::numeric_limits<T>::lowest()) + 1);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> u(low_value, range2);
  for (int i = 0; i < num_size; ++i) {
    v[i] = u(gen);
  }
}

template <>
void init_vector<jd::bfloat16_t>(jd::bfloat16_t* v, int num_size, float range1, float range2, int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> u(range1, range2);
  for (int i = 0; i < num_size; ++i) {
    v[i] = u(gen);
  }
}

template <typename T>
void prepare_blocked_sparse_data(T* data, const std::vector<dim_t>& a_shape, const std::vector<dim_t>& block_shape,

                                 float sparsity, unsigned int* seed) {
  dim_t K = a_shape[0], N = a_shape[1], BK = block_shape[0], BN = block_shape[1];
  LOG_IF(FATAL, (K % BK | N % BN) != 0) << "Matrix dim must be a multiple of block dim.";
  LOG_IF(FATAL, sparsity < 0 || sparsity > 1) << "Sparsity should be a value between 0 and 1.";
  dim_t nb_k = K / BK;
  dim_t nb_n = N / BN;
  std::srand(*seed);
  for (int ibk = 0; ibk < nb_k; ++ibk) {
    for (int ibn = 0; ibn < nb_n; ++ibn) {
      bool fill_zero = std::rand() % 100 <= (sparsity * 100);
      if (fill_zero) {
        dim_t i_start = ibk * BK;
        dim_t j_start = ibn * BN;
        for (dim_t i = i_start; i < i_start + BK; ++i) {
          for (dim_t j = j_start; j < j_start + BN; ++j) {
            data[i * N + j] = 0.f;
          }
        }
      }
    }
  }
}

template <typename T>
void prepare_sparse_data(T* vector_data, dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col, float sparsity,
                         uint32_t* seed = nullptr) {
  uint32_t default_seed = 123;
  if (seed == nullptr) seed = &default_seed;
  std::srand(default_seed);
  for (int i = 0; i < rows; i += blk_row) {
    for (int j = 0; j < cols; j += blk_col) {
      bool fill_zero = std::rand() % 100 <= (sparsity * 100);
      if (fill_zero) {
        for (int bi = i; bi < i + blk_row; ++bi) {
          for (int bj = j; bj < j + blk_col; ++bj) {
            vector_data[bi * cols + bj] = 0;
          }
        }
      }
    }
  }
}

std::pair<const void*, const void*> make_data_obj(const std::vector<dim_t>& a_shape, const jd::data_type& a_dt,
                                                  bool is_clear = false, const std::vector<float>& ranges = {-10, 10},
                                                  float sparsity = 0.f, jd::format_type a_ft = jd::format_type::uncoded,
                                                  const void* src_data = nullptr, const int seed = 5489u) {
  if (a_shape.empty()) {
    return {nullptr, nullptr};
  }
  int elem_num = std::accumulate(a_shape.begin(), a_shape.end(), size_t{1}, std::multiplies<size_t>());
  int bytes_size = elem_num * jd::type_size[a_dt];
  void* data_ptr = nullptr;
  if (is_clear) {
    data_ptr = new uint8_t[bytes_size];
    memset(data_ptr, 0, bytes_size);
  } else if (src_data != nullptr) {
    data_ptr = new uint8_t[bytes_size];
    memcpy(data_ptr, src_data, bytes_size);
  } else if (sparsity != 0.f) {
    std::vector<dim_t> block_shape = {1, 16};
    switch (a_ft) {
      case jd::format_type::bsc: {
        switch (a_dt) {
          case jd::data_type::fp32: {
            data_ptr = new float[elem_num];
            init_vector(reinterpret_cast<float*>(data_ptr), elem_num, ranges[0], ranges[1], seed);
            unsigned int seed_tmp = seed;
            prepare_blocked_sparse_data(reinterpret_cast<float*>(data_ptr), a_shape, block_shape, sparsity, &seed_tmp);
            break;
          }
          default:
            break;
        }
        break;
      }
      default: {
        switch (a_dt) {
          case jd::data_type::s8: {
            data_ptr = new int8_t[elem_num];
            init_vector(reinterpret_cast<int8_t*>(data_ptr), elem_num, ranges[0], ranges[1], seed);
            prepare_sparse_data(reinterpret_cast<int8_t*>(data_ptr), a_shape[0], a_shape[1], 4, 1, sparsity);
            break;
          }

          case jd::data_type::bf16: {
            data_ptr = new jd::bfloat16_t[elem_num];
            init_vector(reinterpret_cast<jd::bfloat16_t*>(data_ptr), elem_num, ranges[0], ranges[1], seed);
            prepare_sparse_data(reinterpret_cast<jd::bfloat16_t*>(data_ptr), a_shape[0], a_shape[1], 16, 1, sparsity);
            break;
          }
          default:
            break;
        }
      }
    }
  } else {
    switch (a_dt) {
      case jd::data_type::fp32:
        data_ptr = new float[elem_num];
        init_vector(reinterpret_cast<float*>(data_ptr), elem_num, ranges[0], ranges[1], seed);
        break;
      case jd::data_type::s32:
        data_ptr = new int32_t[elem_num];
        init_vector(reinterpret_cast<int32_t*>(data_ptr), elem_num, ranges[0], ranges[1], seed);
        break;
      case jd::data_type::u8:
        data_ptr = new uint8_t[elem_num];
        init_vector(reinterpret_cast<uint8_t*>(data_ptr), elem_num, ranges[0], ranges[1], seed);
        break;
      case jd::data_type::s8:
        data_ptr = new int8_t[elem_num];
        init_vector(reinterpret_cast<int8_t*>(data_ptr), elem_num, ranges[0], ranges[1], seed);
        break;
      case jd::data_type::bf16: {
        data_ptr = new jd::float16_t[elem_num];
        init_vector(reinterpret_cast<jd::bfloat16_t*>(data_ptr), elem_num, ranges[0], ranges[1], seed);
        break;
      }
      case jd::data_type::f8_e4m3: {
        data_ptr = new jd::float8_e4m3_t[elem_num];
        init_vector(reinterpret_cast<jd::float8_e4m3_t*>(data_ptr), elem_num, ranges[0], ranges[1], seed);
        break;
      }
      case jd::data_type::f8_e5m2: {
        data_ptr = new jd::float8_e5m2_t[elem_num];
        init_vector(reinterpret_cast<jd::float8_e5m2_t*>(data_ptr), elem_num, ranges[0], ranges[1], seed);
        break;
      }
      default:
        assert(false);
        break;
    }
  }
  void* data_ptr_copy = new uint8_t[bytes_size];
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}

std::pair<const void*, const void*> make_data_obj(const jd::tensor_desc& tensor_desc, bool is_clear = false,
                                                  const std::vector<float>& ranges = {-10, 10}, float sparsity = 0.f,
                                                  jd::format_type a_ft = jd::format_type::uncoded,
                                                  const void* src_data = nullptr, const int seed = 5489u) {
  return make_data_obj(tensor_desc.shape(), tensor_desc.dtype(), is_clear, ranges, sparsity, a_ft, src_data, seed);
}

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

/**
 * @brief Convert to string in the form of a valid c++ identifier (but may start with a number)
 */
inline std::string num2id(std::string s) {
  std::replace(s.begin(), s.end(), '.', 'D');  // replace all decimal sep
  std::replace(s.begin(), s.end(), '-', 'M');  // replace all minus sign
  return s;
}
inline std::string num2id(float n) { return num2id(std::to_string(n)); }

class n_thread_t {
 public:
  explicit n_thread_t(int nthr, bool cap = false) : prev_nthr(omp_get_max_threads()) {
    if (nthr > 0 && nthr != prev_nthr && (!cap || nthr < prev_nthr)) omp_set_num_threads(nthr);
  }
  ~n_thread_t() { omp_set_num_threads(prev_nthr); }

 private:
  int prev_nthr;
};
}  // namespace test
#endif  // ENGINE_TEST_GTEST_SPARSELIB_UNIT_TEST_UTILS_HPP_
