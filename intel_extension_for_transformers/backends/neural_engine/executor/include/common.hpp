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

#ifndef ENGINE_EXECUTOR_INCLUDE_COMMON_HPP_
#define ENGINE_EXECUTOR_INCLUDE_COMMON_HPP_

#include <float.h>
#include <glog/logging.h>
#include <limits.h>
#include <omp.h>

#include <chrono>  // NOLINT
#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <functional>
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>  // pair
#include <vector>

#include "memory_allocator.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "interface.hpp"

#if __AVX512F__
#include <immintrin.h>
#endif

#ifdef NEURALENGINE_SHARED_LIB_
#ifndef NEURALENGINE_API_
#ifdef _MSC_VER
#if NEURALENGINE_BUILD
#define NEURALENGINE_API_ __declspec(dllexport)
#else
#define NEURALENGINE_API_ __declspec(dllimport)
#endif
#elif __GNUC__ >= 4 || defined(__clang__)
#define NEURALENGINE_API_ __attribute__((visibility("default")))
#endif  // _MSC_VER

#endif  // SPARSE_API_
#endif  // NEURALENGINE_SHARED_LIB_

#ifndef NEURALENGINE_API_
#define NEURALENGINE_API_
#endif  // GTEST_API_


namespace executor {

using std::max;
using std::min;
using std::set;
using std::unordered_map;
using std::vector;

NEURALENGINE_API_ void GlobalInit(const char* pname);

extern unordered_map<string, int> type2bytes;
// some kernel may need config to execute when be dispatched
// write the specific config into dispatch table
extern unordered_map<string, vector<string>> dispatch_kernel_config;

// read weight file to data
void* read_file_to_type(const string& root, const string& type, const vector<int64_t>& shape,
                        const vector<int64_t>& location);

template <typename T>
void InitVector(T* v, int buffer_size);

int64_t Product(const vector<int64_t>& shape);

// Get the shapes vector with the absolute perm. Default or empty perm is (0, 1, 2, 3, ...).
// e.g.: shape_before = (64, 384, 16, 64), perm = (0, 2, 1, 3), return (64, 16, 384, 64)
vector<int64_t> GetShapes(const vector<int64_t>& origin_shape, const vector<int64_t>& absolute_perm = {});

// Get the strides vector with the absolute perm. Default or empty perm is (0, 1, 2, 3, ...).
// Tensor stride is a product of its higher dimensions, Stride[0] = Shape[1]*Shape[2]*...*Shape[n].
// e.g.: axis = (0, 1, 2, 3), shape = (64, 16, 384, 64), return stride = (16*384*64, 384*64, 64, 1)
vector<int64_t> GetStrides(const vector<int64_t>& origin_shape, const vector<int64_t>& absolute_perm = {});

vector<int64_t> GetDstShape(const vector<int64_t>& dst_shape, size_t dst_size,
                            const vector<int64_t>& ref_shape,
                            const vector<int64_t>& reshape_dims);

template <typename T>
T StringToNum(const string& str);

// Compare two buffer
template <typename T>
bool CompareData(const void* buf1, int64_t elem_num1, const void* buf2, int64_t elem_num2, float eps = 1e-6);

bool CompareShape(const vector<int64_t>& shape1, const vector<int64_t>& shape2);

vector<float> GetScales(const void* mins, const void* maxs, const int64_t size, const string& dtype);

vector<float> GetRescales(const vector<float>& src0_scales, const vector<float>& src1_scales,
                          const vector<float>& dst_scales, const string& dst_dtype, const bool append_eltwise = false);

vector<int> GetZeroPoints(const void* mins, const vector<float>& scales, const string& dtype);

void AddZeroPoints(const int size, const string& dtype, const float* src_data, const float* range_mins,
                   const vector<float>& scales, float* dst_data);

#if __AVX512F__
void Quantize_avx512(const int size, const string& dtype, const void* src_data, const float* range_mins,
                     const vector<float>& scales, void* dst_data);
#else
void Quantize(const int size, const string& dtype, const void* src_data, const float* range_mins,
              const vector<float>& scales, void* dst_data);
#endif

template <typename T>
float GetSparseRatio(const T* data, const vector<int64_t>& shape, const vector<int64_t>& blocksize);

vector<int64_t> ReversePerm(const vector<int64_t>& perm_to);

NEURALENGINE_API_ int64_t Time();

NEURALENGINE_API_ float Duration(int64_t start, int64_t end);

template <typename T>
void PrintToFile(const T* data, const std::string& name, size_t size = 1000);

template <typename T>
void StringSplit(vector<T>* split_list, const string& str_list, const string& split_op);

void InitSparse(int K, int N, int N_BLKSIZE, int K_BLKSIZE, int N_SPARSE, float* A);

/************* ref ************/
template <typename dst_type, typename src_type>
void ref_mov_ker(dst_type* inout, const src_type* in, size_t len);
template <typename dst_type, typename src_type>
void ref_add_ker(dst_type* inout, src_type* in, size_t len);
/************* fp32 ************/
void zero_ker(float* out, size_t len);
void move_ker(float* out, const float* in, size_t len);
void add_ker(float* inout, float* in, size_t len);
/************* bf16 ************/
#if __AVX512F__
// Conversion from BF16 to FP32
__m512 cvt_bf16_to_fp32(const __m256i src);
// Conversion from FP32 to BF16
__m256i trunc_fp32_to_bf16(const __m512 src);
__m256i cvt_fp32_to_bf16(const __m512 src);
#endif
void zero_ker(uint16_t* out, size_t len);
void move_ker(uint16_t* out, const uint16_t* in, size_t len);
void add_ker(uint16_t* inout, uint16_t* in, size_t len);
/************* int8 ************/
void zero_ker(uint8_t* out, size_t len);
void move_ker(uint8_t* out, const uint8_t* in, size_t len);
void add_ker(uint8_t* inout, uint8_t* in, size_t len);

void runtime_minmax(float* data, size_t len, float* min_num, float* max_num);
#if __AVX512F__
void block_minmax_avx512(float* Input, size_t N, float* Min, float* Max);
#else
void block_minmax(float* Input, size_t N, float* Min, float* Max);
#endif

/************ hash funtion for primitive cache ************/
template <typename T>
inline size_t hash_combine(size_t seed, const T& v) {
  return seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T>
inline size_t hash_val(size_t seed, const T& val) {
  return hash_combine(seed, val);
}

template <typename T, typename... Types>
inline size_t hash_val(size_t seed, const T& val, const Types&... args) {
  seed = hash_combine(seed, val);
  return hash_val(seed, args...);
}

template <typename... Types>
inline size_t hash_val(const Types&... args) {
  size_t seed = 0;
  return hash_val(seed, args...);
}

template <typename T>
inline size_t get_array_hash(size_t seed, const T& v, int size) {
  for (int i = 0; i < size; i++) {
    seed = hash_combine(seed, v[i]);
  }
  return seed;
}

template <typename K, typename V>
inline size_t get_map_hash(size_t seed, const std::map<K, V>& map) {
  if (!map.empty()) {
    for (auto iter = map.begin(); iter != map.end(); ++iter) {
      seed = hash_val(seed, iter->first, iter->second);
    }
    return seed;
  }
}

// Base class for dnnl primitive cache map.
// It will cache some important dnnl primitive in order to
// increase the probability of cache hit under dynamic input shape.
// Put all dnnl primitive into a single pool.
template <typename T>
class PrimitiveCachePool {
 public:
  PrimitiveCachePool() { Clear(); }
  ~PrimitiveCachePool() {}

  bool IsInCache(const size_t& key) {
    auto it = cache_map_.find(key);
    if (it != cache_map_.end()) {
      return true;
    } else {
      return false;
    }
  }

  // call in_cache function first
  T& GetContext(const size_t& key) noexcept { return cache_map_[key].primitive; }

  void SetContext(const size_t& key, T primitive) {
    Entry entry(primitive);
    cache_map_.emplace(std::make_pair(key, primitive));
  }

  // Clean up the cache map
  void Clear() {
    if (cache_map_.empty()) {
      return;
    } else {
      cache_map_.clear();
    }
  }

 private:
  // a struct for caching
  struct Entry {
    // dnnl primitive, it's mostly forward class
    T primitive;
    Entry() {}
    explicit Entry(T prim) { primitive = prim; }
    ~Entry() {}
  };

  // cache map
  unordered_map<size_t, Entry> cache_map_;
};

// Base class for each dnnl primitive to get the cached object
template <typename T>
class DnnlPrimitiveFactory {
 public:
  DnnlPrimitiveFactory() {
    do_not_cache_ = (getenv("ENGINE_PRIMITIVE_CACHE_OFF") != NULL);
  }
  ~DnnlPrimitiveFactory() {}

  bool IsInCache(const size_t& key) {
    auto& cache_pool = DnnlPrimitiveFactory<T>::GetCachePool();
    return cache_pool.IsInCache(key);
  }

  // call in_cache function first
  dnnl::primitive& GetPrimitive(const size_t& key) {
    auto& cache_pool = DnnlPrimitiveFactory<T>::GetCachePool();
    return cache_pool.GetContext(key);
  }

  void SetPrimitive(const size_t& key, dnnl::primitive primitive) {
    auto& cache_pool = DnnlPrimitiveFactory<T>::GetCachePool();
    cache_pool.SetContext(key, primitive);
  }

  void Clear() {
    auto& cache_pool = DnnlPrimitiveFactory<T>::GetCachePool();
    cache_pool.Clear();
  }

  // If set this env var, all dnnl primitive cache will be disabled
  bool do_not_cache_ = false;

 private:
  static inline PrimitiveCachePool<dnnl::primitive>& GetCachePool() {
    static thread_local PrimitiveCachePool<dnnl::primitive> cache_pool_;
    return cache_pool_;
  }
};

// Singleton, cache inner product forward class
class InnerProductPrimitiveFwdFactory : public DnnlPrimitiveFactory<dnnl::inner_product_forward> {
 public:
  static size_t Key(const string& src0_dtype, const string& src1_dtype, const string& dst_dtype,
                    const vector<int64_t>& src0_shape, const vector<int64_t>& src1_shape,
                    const vector<int64_t>& dst_perm, const string& append_op, const vector<int64_t>& post_op_shape,
                    const float& output_scale, const dnnl::engine* eng);
  static bool IsInFactory(const size_t& key);
  // call IsInFactory function first
  static dnnl::inner_product_forward& Get(const size_t& key);
  static void Set(const size_t& key, dnnl::primitive primitive);
  static void ClearFactory();
  static bool DoNotCache();

 private:
  InnerProductPrimitiveFwdFactory() {}
  ~InnerProductPrimitiveFwdFactory() {}
  static InnerProductPrimitiveFwdFactory& GetInstance();
  static size_t GenKey(const string& src0_dtype, const string& src1_dtype, const string& dst_dtype,
                       const vector<int64_t>& src0_shape, const vector<int64_t>& src1_shape,
                       const vector<int64_t>& dst_perm, const string& append_op, const vector<int64_t>& post_op_shape,
                       const float& output_scale, const dnnl::engine* eng);
};

// Singleton, cache matmul forward class
class MatMulPrimitiveFwdFactory : public DnnlPrimitiveFactory<dnnl::matmul> {
 public:
  static size_t Key(const string& src0_dtype, const string& src1_dtype, const string& dst_dtype,
                    const vector<int64_t>& src0_shape, const vector<int64_t>& src1_shape,
                    const vector<int64_t>& dst_perm, const string& append_op, const vector<int64_t>& post_op_shape,
                    const float& output_scale, const dnnl::engine* eng);
  static bool IsInFactory(const size_t& key);
  // call IsInFactory function first
  static dnnl::matmul& Get(const size_t& key);
  static void Set(const size_t& key, dnnl::primitive primitive);
  static void ClearFactory();
  static bool DoNotCache();

 private:
  MatMulPrimitiveFwdFactory() {}
  ~MatMulPrimitiveFwdFactory() {}
  static MatMulPrimitiveFwdFactory& GetInstance();
  static size_t GenKey(const string& src0_dtype, const string& src1_dtype, const string& dst_dtype,
                       const vector<int64_t>& src0_shape, const vector<int64_t>& src1_shape,
                       const vector<int64_t>& dst_perm, const string& append_op, const vector<int64_t>& post_op_shape,
                       const float& output_scale, const dnnl::engine* eng);
};

// Singleton, cache inner product forward class
class ConvolutionPrimitiveFwdFactory : public DnnlPrimitiveFactory<dnnl::convolution_forward> {
 public:
  static size_t Key(const string& src0_dtype, const string& src1_dtype, const string& dst_dtype,
                    const vector<int64_t>& src0_shape, const vector<int64_t>& src1_shape,
                    const vector<int64_t>& dst_perm, const string& append_op, const vector<int64_t>& post_op_shape,
                    const float& output_scale, const int64_t& group, const vector<int64_t>& pads,
                    const vector<int64_t>& strides, const dnnl::engine* eng);
  static bool IsInFactory(const size_t& key);
  // call IsInFactory function first
  static dnnl::convolution_forward& Get(const size_t& key);
  static void Set(const size_t& key, dnnl::primitive primitive);
  static void ClearFactory();
  static bool DoNotCache();

 private:
  ConvolutionPrimitiveFwdFactory() {}
  ~ConvolutionPrimitiveFwdFactory() {}
  static ConvolutionPrimitiveFwdFactory& GetInstance();
  static size_t GenKey(const string& src0_dtype, const string& src1_dtype, const string& dst_dtype,
                       const vector<int64_t>& src0_shape, const vector<int64_t>& src1_shape,
                       const vector<int64_t>& dst_perm, const string& append_op, const vector<int64_t>& post_op_shape,
                       const float& output_scale, const int64_t& group, const vector<int64_t>& pads,
                       const vector<int64_t>& strides, const dnnl::engine* eng);
};

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_COMMON_HPP_
