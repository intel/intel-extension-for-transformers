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
#include "common.hpp"

#include "cmath"

namespace executor {

unordered_map<string, int> type2bytes = {
    {"fp32", sizeof(float)}, {"int8", sizeof(char)}, {"int32", sizeof(int)},     {"u8", sizeof(unsigned char)},
    {"s8", sizeof(char)},    {"s32", sizeof(int)},   {"bf16", sizeof(uint16_t)}, {"int64", sizeof(int64_t)}};
unordered_map<string, vector<string>> dispatch_kernel_config = {
    {"InnerProduct_to_Convolution", {"input_shape"}},
    {"InnerProduct_to_SparseLib", {"input_shape", "micro_oc", "sub_func"}},
};
const int CPU_COUNT = omp_get_max_threads();
#if __AVX512F__
const int ALIGN_NUM = 64;
#else
const int ALIGN_NUM = 32;
#endif

void GlobalInit(const char* pname) {
  // Google logging.
  ::google::InitGoogleLogging(pname);
  FLAGS_logtostderr = 1;
  // Provide a backtrace on segfault.
  ::google::InstallFailureSignalHandler();
}
// read weight file to data
/*
Args:
    root is the model const tensors like weights, bias .bin file path.
    type is for setting the tensor datatype, like 'float32'
    shape is the data shape
    location gives the info of tensor data location in .bin file
      location[0] is the start idx when sotre the data bytes
      location[1] is the data bytes length
Return:
    void* ptr, points a consecutive memory that sotres the data
*/
void* read_file_to_type(const string& root, const string& type, const vector<int64_t>& shape,
                        const vector<int64_t>& location) {
  int b = type2bytes[type];
  if (b == 0) {
    DLOG(INFO) << type << " not implemented yet...";
  }

  int64_t size = Product(shape);
  // from file tensor will directly malloc memory
  void* p = reinterpret_cast<void*>(aligned_alloc(ALIGNMENT, (size * b / ALIGNMENT + 1) * ALIGNMENT));

  std::ifstream inFile(root, std::ios::in | std::ios::binary);
  if (inFile) {
    inFile.seekg(location[0], std::ios::beg);
    inFile.read(reinterpret_cast<char*>(p), location[1]);
    inFile.close();
  } else {
    std::memcpy(p, &root[location[0]], location[1]);
  }
  return p;
}

template <typename T>
void InitVector(T* v, int num_size, float range1, float range2, int seed) {
  float low_value = std::max(range1, static_cast<float>(std::numeric_limits<T>::lowest()) + 1);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> u(low_value, range2);
  for (int i = 0; i < num_size; ++i) {
    v[i] = static_cast<T>(u(gen));
  }
}
template void InitVector<float>(float* v, int num_size, float range1, float range2, int seed);
template void InitVector<uint16_t>(uint16_t* v, int num_size, float range1, float range2, int seed);  // bf16
template void InitVector<int8_t>(int8_t* v, int num_size, float range1, float range2, int seed);
template void InitVector<uint8_t>(uint8_t* v, int num_size, float range1, float range2, int seed);
template void InitVector<int32_t>(int32_t* v, int num_size, float range1, float range2, int seed);

// Displayed in milliseconds.
int64_t Time() {
  std::chrono::system_clock::time_point now_time = std::chrono::system_clock::now();
  std::chrono::system_clock::duration dur = now_time.time_since_epoch();
  std::chrono::microseconds micros = std::chrono::duration_cast<std::chrono::microseconds>(dur);
  return micros.count();
}

float Duration(int64_t start, int64_t end) {
  float duration = (end - start) / 1e3;
  return duration;
}

int64_t Product(const vector<int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), (int64_t)1, std::multiplies<int64_t>());
}

// Get the shapes vector with the absolute perm. Default or empty perm is (0, 1,
// 2, 3, ...). e.g.: shape_before = (64, 384, 16, 64), perm = (0, 2, 1, 3),
// return (64, 16, 384, 64)
vector<int64_t> GetShapes(const vector<int64_t>& origin_shape, const vector<int64_t>& absolute_perm) {
  if (absolute_perm.empty()) {
    return origin_shape;
  }
  int shape_size = origin_shape.size();
  vector<int64_t> transed_shape(shape_size, 0);
  for (int i = 0; i < shape_size; ++i) {
    int trans_axis_id = absolute_perm[i];
    transed_shape[i] = origin_shape[trans_axis_id];
  }
  return transed_shape;
}

// Get the strides vector with the absolute perm. Default or empty perm is (0,
// 1, 2, 3, ...). Tensor's each stride is the product of all higher dimensions
// Stride[0] = Shape(1)*Shape(2)*...*Shape(n).
// e.g.: axis = (0, 1, 2, 3), shape = (64, 16, 384, 64), return stride =
// (16*384*64, 384*64, 64, 1)
vector<int64_t> GetStrides(const vector<int64_t>& origin_shape, const vector<int64_t>& absolute_perm) {
  int shape_size = origin_shape.size();
  vector<int64_t> origin_strides(shape_size, 1);
  if (shape_size >= 2) {
    for (int i = shape_size - 2; i >= 0; --i) {
      origin_strides[i] = origin_shape[i + 1] * origin_strides[i + 1];
    }
  }
  return GetShapes(origin_strides, absolute_perm);
}

template <typename T>
bool CompareData(const void* buf1, int64_t elem_num1, const void* buf_true, int64_t elem_true, float eps) {
  if (buf1 == buf_true) {
    return false;
  }
  if (elem_num1 != elem_true) {
    return false;
  }
  const auto buf1_data = static_cast<const T*>(buf1);
  const auto buf2_data = static_cast<const T*>(buf_true);
  for (int64_t i = 0; i < elem_num1; ++i) {
    auto err = fabs(buf1_data[i] - buf2_data[i]);
    if (err > eps) {
      LOG(ERROR) << "idx: " << i << ", predict: " << static_cast<int>(buf1_data[i])
                 << ", true: " << static_cast<int>(buf2_data[i]) << ", err: " << err << ", eps: " << eps;
      return false;
    }
  }
  return true;
}
template <>
bool CompareData<float>(const void* buf1, int64_t elem_num1, const void* buf_true, int64_t elem_true, float eps) {
  if (buf1 == buf_true) {
    return false;
  }
  if (elem_num1 != elem_true) {
    return false;
  }
  const auto buf1_data = static_cast<const float*>(buf1);
  const auto buf2_data = static_cast<const float*>(buf_true);
  float err = 0;
  float ref = 0;
  for (int64_t i = 0; i < elem_num1; ++i) {
    ref += buf2_data[i] * buf2_data[i];
    err += (buf1_data[i] - buf2_data[i]) * (buf1_data[i] - buf2_data[i]);
  }
  if (std::sqrt(err) / std::sqrt(ref) > eps) {
    LOG(ERROR) << "Reference matrix: " << std::sqrt(ref) << ", Error: " << std::sqrt(err)
               << ", Relative error:" << std::sqrt(err) / std::sqrt(ref) << ", eps" << eps;
    return false;
  }
  return true;
}
template bool CompareData<uint16_t>(const void* buf1, int64_t elem_num1, const void* buf_true, int64_t elem_true,
                                    float eps);  // bf16
template bool CompareData<int8_t>(const void* buf1, int64_t elem_num1, const void* buf_true, int64_t elem_true,
                                  float eps);
template bool CompareData<uint8_t>(const void* buf1, int64_t elem_num1, const void* buf_true, int64_t elem_true,
                                   float eps);

bool CompareShape(const vector<int64_t>& shape1, const vector<int64_t>& shape2) {
  if (shape1.size() != shape2.size()) return false;
  for (int i = 0; i < shape1.size(); i++) {
    if (shape1[i] != shape2[i]) return false;
  }
  return true;
}

vector<float> GetScales(const void* mins, const void* maxs, const int64_t size, const string& dtype) {
  const float* mins_p = static_cast<const float*>(mins);
  const float* maxs_p = static_cast<const float*>(maxs);
  vector<float> scales;
  if (dtype == "u8") {
    for (int i = 0; i < size; i++) {
      float max_sub_min = maxs_p[i] - mins_p[i];
      max_sub_min = max_sub_min < 1e-10 ? 1e-10 : max_sub_min;
      scales.emplace_back(255.f / max_sub_min);
    }
  } else if (dtype == "s8") {
    for (int i = 0; i < size; i++) {
      float abs_max = max(abs(maxs_p[i]), abs(mins_p[i]));
      abs_max = abs_max < 1e-10 ? 1e-10 : abs_max;
      scales.emplace_back(127.f / abs_max);
    }
  } else if (dtype == "fp32") {
    for (int i = 0; i < size; i++) {
      scales.emplace_back(1.f);
    }
  } else {
    LOG(ERROR) << "Can't suppport dst_dtype: " << dtype << " now!";
  }
  return scales;
}

vector<float> GetRescales(const vector<float>& src0_scales, const vector<float>& src1_scales,
                          const vector<float>& dst_scales, const string& dst_dtype, const bool append_eltwise) {
  vector<float> rescales;
  if (dst_dtype == "fp32" || dst_dtype == "bf16") {
    for (int i = 0; i < src1_scales.size(); i++) {
      rescales.emplace_back(1. / (src0_scales[0] * src1_scales[i]));
    }
  } else if (dst_dtype == "s8" && !dst_scales.empty()) {
    for (int i = 0; i < src1_scales.size(); i++) {
      auto rescale =
          append_eltwise ? 1. / (src0_scales[0] * src1_scales[i]) : dst_scales[0] / (src0_scales[0] * src1_scales[i]);
      rescales.emplace_back(rescale);
    }
  } else if (dst_dtype == "u8" && !dst_scales.empty()) {
    for (int i = 0; i < src1_scales.size(); i++) {
      auto rescale =
          append_eltwise ? 1. / (src0_scales[0] * src1_scales[i]) : dst_scales[0] / (src0_scales[0] * src1_scales[i]);
      rescales.emplace_back(rescale);
    }
  } else {
    LOG(ERROR) << "Can't suppport dst_dtype: " << dst_dtype << " now!";
  }
  return rescales;
}

vector<int64_t> GetDstShape(const vector<int64_t>& dst_shape, size_t dst_size, const vector<int64_t>& ref_shape,
                            const vector<int64_t>& reshape_dims) {
  vector<int64_t> pre_dst_shape(dst_shape);
  if (!ref_shape.empty()) {
    auto shape_vec = ref_shape;
    int j = 0;
    for (int i = 0; i < pre_dst_shape.size(); i++) {
      if (pre_dst_shape[i] == -1) {
        pre_dst_shape[i] = shape_vec[reshape_dims[j++]];
      }
      if (j >= reshape_dims.size()) {
        break;
      }
    }
  }

  int64_t src_size = dst_size;
  int idx = -1;
  int64_t shape_acc = 1;
  for (int i = 0; i < pre_dst_shape.size(); i++) {
    if (pre_dst_shape[i] != -1) {
      shape_acc *= pre_dst_shape[i];
    } else {
      idx = i;
    }
  }
  if (idx != -1) {
    pre_dst_shape[idx] = src_size / shape_acc;
  }
  return pre_dst_shape;
}

vector<int> GetZeroPoints(const void* mins, const vector<float>& scales, const string& dtype) {
  const float* mins_p = static_cast<const float*>(mins);
  vector<int> zerops;
  if (dtype == "u8") {
    for (int i = 0; i < scales.size(); i++) zerops.emplace_back(nearbyint(-mins_p[i] * scales[i]));
  } else if (dtype == "s8") {
    for (int i = 0; i < scales.size(); i++) zerops.emplace_back(nearbyint(-128 - mins_p[i] * scales[i]));
  } else {
    LOG(ERROR) << "Can't suppport dtype: " << dtype << " now!";
  }
  return zerops;
}

vector<int> GetZeroPoints(const float* mins, const float* scales, const string& dtype, const int size) {
  vector<int> zerops;
  if (dtype == "u8") {
    for (int i = 0; i < size; i++) zerops.emplace_back(nearbyint(-mins[i] * scales[i]));
  } else if (dtype == "s8") {
    for (int i = 0; i < size; i++) zerops.emplace_back(nearbyint(-128 - mins[i] * scales[i]));
  } else {
    LOG(ERROR) << "Can't suppport dtype: " << dtype << " now!";
  }
  return zerops;
}

void AddZeroPoints(const int size, const string& dtype, const float* src_data, const float* range_mins,
                   const vector<float>& scales, float* dst_data) {
  if (dtype == "u8") {
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
      dst_data[i] = src_data[i] - range_mins[0];
    }
  } else if (dtype == "s8") {
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
      dst_data[i] = src_data[i] - 128 / scales[0] - range_mins[0];
    }
  } else {
    LOG(ERROR) << "Can't suppport dst_dtype: " << dtype << " now!";
  }
  return;
}

#ifdef __AVX512F__

void Quantize_bf16_s8(const int size, const void* src_data, const std::vector<float>& scales, void* dst_data) {
  const uint16_t* src_data_ = reinterpret_cast<const uint16_t*>(src_data);
  int8_t* dst_data_ = reinterpret_cast<int8_t*>(dst_data);
  __m512 min_with_scale_s8 = _mm512_set1_ps(0);
  __m512 scale = _mm512_set1_ps(scales[0]);
  __m512i zero = _mm512_setzero_epi32();
  int offset = size / 16 * 16;
#pragma omp parallel for
  for (int i = 0; i < size; i += 16) {
    if (i < offset) {
#if __AVX512BF16__ && __GNUC__ > 11
      __m256bh src_bf16 = (__m256bh)_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_data_ + i));
      __m512 src_fp32 = _mm512_cvtpbh_ps(src_bf16);
#else
      __m256i src_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_data_ + i));
      __m512 src_fp32 = cvt_bf16_to_fp32(src_bf16);
#endif
      __m512 dst_fp32 = _mm512_fmsub_ps(src_fp32, scale, min_with_scale_s8);
      __m512i dst_int32 = _mm512_cvt_roundps_epi32(dst_fp32, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m128i dst_int8 = _mm512_cvtsepi32_epi8(dst_int32);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_data_ + i), dst_int8);
    } else {
      __mmask16 mask = (1ULL << (size - offset)) - 1;
#if __AVX512BF16__ && __GNUC__ > 11
      __m256bh src_bf16 = (__m256bh)_mm256_maskz_loadu_epi16(mask, src_data_ + offset);
      __m512 src_fp32 = _mm512_maskz_cvtpbh_ps(mask, src_bf16);
#else
      __m256i src_bf16 = _mm256_maskz_loadu_epi16(mask, src_data_ + offset);
      __m512 src_fp32 = cvt_bf16_to_fp32(mask, src_bf16);
#endif
      __m512 dst_fp32 = _mm512_maskz_fmsub_ps(mask, src_fp32, scale, min_with_scale_s8);
      __m512i dst_int32 =
          _mm512_maskz_cvt_roundps_epi32(mask, dst_fp32, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m128i dst_int8 = _mm512_maskz_cvtsepi32_epi8(mask, dst_int32);
      _mm_mask_storeu_epi8(reinterpret_cast<__m128i*>(dst_data_ + offset), mask, dst_int8);
    }
  }
}

void Quantize_bf16_u8(const int size, const void* src_data, const float* range_mins, const std::vector<float>& scales,
                      void* dst_data) {
  const uint16_t* src_data_ = reinterpret_cast<const uint16_t*>(src_data);
  uint8_t* dst_data_ = reinterpret_cast<uint8_t*>(dst_data);
  __m512 _min_with_scale_u8 = _mm512_set1_ps(range_mins[0] * scales[0]);
  __m512 scale = _mm512_set1_ps(scales[0]);
  __m512i zero = _mm512_setzero_epi32();
  int offset = size / 16 * 16;
#pragma omp parallel for
  for (int i = 0; i < size; i += 16) {
    if (i < offset) {
#if __AVX512BF16__ && __GNUC__ > 11
      __m256bh src_bf16 = (__m256bh)_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_data_ + i));
      __m512 src_fp32 = _mm512_cvtpbh_ps(src_bf16);
#else
      __m256i src_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_data_ + i));
      __m512 src_fp32 = cvt_bf16_to_fp32(src_bf16);
#endif
      __m512 dst_fp32 = _mm512_fmsub_ps(src_fp32, scale, _min_with_scale_u8);
      __m512i dst_int32 = _mm512_cvt_roundps_epi32(dst_fp32, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      dst_int32 = _mm512_max_epi32(dst_int32, zero);
      __m128i dst_int8 = _mm512_cvtusepi32_epi8(dst_int32);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_data_ + i), dst_int8);
    } else {
      __mmask16 mask = (1ULL << (size - offset)) - 1;
#if __AVX512BF16__ && __GNUC__ > 11
      __m256bh src_bf16 = (__m256bh)_mm256_maskz_loadu_epi16(mask, src_data_ + offset);
      __m512 src_fp32 = _mm512_maskz_cvtpbh_ps(mask, src_bf16);
#else
      __m256i src_bf16 = _mm256_maskz_loadu_epi16(mask, src_data_ + offset);
      __m512 src_fp32 = cvt_bf16_to_fp32(mask, src_bf16);
#endif
      __m512 dst_fp32 = _mm512_maskz_fmsub_ps(mask, src_fp32, scale, _min_with_scale_u8);
      __m512i dst_int32 =
          _mm512_maskz_cvt_roundps_epi32(mask, dst_fp32, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      dst_int32 = _mm512_max_epi32(dst_int32, zero);
      __m128i dst_int8 = _mm512_maskz_cvtusepi32_epi8(mask, dst_int32);
      _mm_mask_storeu_epi8(reinterpret_cast<__m128i*>(dst_data_ + offset), mask, dst_int8);
    }
  }
}

void Quantize_fp32_bf16(const int size, const void* src_data, const vector<float>& scales, void* dst_data) {
  const float* src_data_ = static_cast<const float*>(src_data);

  int avx512_loop_len = size >> 4;

  uint16_t* dst_data_ = static_cast<uint16_t*>(dst_data);
#pragma omp parallel for
  for (int i = 0; i < avx512_loop_len; ++i) {
    __m512 _src_data = _mm512_loadu_ps(src_data_ + (i << 4));
#if __AVX512BF16__ && __GNUC__ > 11
    __m256i data_bf16 = (__m256i)_mm512_cvtneps_pbh(_src_data);
#else
    auto y = _mm512_bsrli_epi128(_mm512_castps_si512(_src_data), 2);
    __m256i data_bf16 = _mm512_cvtepi32_epi16(y);
#endif
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_data_ + (i << 4)), data_bf16);
  }
  union {
    unsigned int u;
    float f;
  } typecast;
#pragma omp parallel for
  for (int i = (avx512_loop_len << 4); i < size; i++) {
    typecast.f = src_data_[i];
    dst_data_[i] = typecast.u >> 16;
  }
  return;
}

void Quantize_fp32_u8(const int size, const void* src_data, const float* range_mins, const vector<float>& scales,
                      void* dst_data) {
  const float* src_data_ = static_cast<const float*>(src_data);

  int avx512_loop_len = size >> 4;

  __m512 _min_with_scale_u8 = _mm512_set1_ps(range_mins[0] * scales[0]);
  __m512 _scale = _mm512_set1_ps(scales[0]);
  __m512i zero = _mm512_setzero_epi32();

  unsigned char* dst_data_ = static_cast<unsigned char*>(dst_data);
#pragma omp parallel for
  for (int i = 0; i < avx512_loop_len; ++i) {
    __m512 _src_data = _mm512_loadu_ps(src_data_ + (i << 4));
    __m512 data = _mm512_fmsub_ps(_src_data, _scale, _min_with_scale_u8);
    __m512i data_x32 = _mm512_cvt_roundps_epi32(data, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    data_x32 = _mm512_max_epi32(data_x32, zero);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_data_ + (i << 4)), _mm512_cvtusepi32_epi8(data_x32));
  }

#pragma omp parallel for
  for (int i = (avx512_loop_len << 4); i < size; i++) {
    int32_t data = nearbyint((src_data_[i] - range_mins[0]) * scales[0]);
    data = data < 0 ? 0 : data;
    data = data > 255 ? 255 : data;
    dst_data_[i] = static_cast<unsigned char>(data);
  }
  return;
}

void Quantize_fp32_s8(const int size, const void* src_data, const vector<float>& scales, void* dst_data) {
  const float* src_data_ = static_cast<const float*>(src_data);

  int avx512_loop_len = size >> 4;

  __m512 _min_with_scale_s8 = _mm512_set1_ps(0);
  __m512 _scale = _mm512_set1_ps(scales[0]);

  char* dst_data_ = static_cast<char*>(dst_data);
#pragma omp parallel for
  for (int i = 0; i < avx512_loop_len; ++i) {
    __m512 _src_data = _mm512_loadu_ps(src_data_ + (i << 4));
    __m512 data = _mm512_fmsub_ps(_src_data, _scale, _min_with_scale_s8);
    __m512i data_x32 = _mm512_cvt_roundps_epi32(data, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_data_ + (i << 4)), _mm512_cvtsepi32_epi8(data_x32));
  }
#pragma omp parallel for
  for (int i = (avx512_loop_len << 4); i < size; i++) {
    int32_t data = nearbyint(src_data_[i] * scales[0]);
    data = data < -128 ? -128 : data;
    data = data > 127 ? 127 : data;
    dst_data_[i] = static_cast<char>(data);
  }
  return;
}

void Quantize_avx512(const int size, const string& dtype, const void* src_data, const float* range_mins,
                     const vector<float>& scales, void* dst_data) {
  if (dtype == "u8") {
    Quantize_fp32_u8(size, src_data, range_mins, scales, dst_data);
  } else if (dtype == "s8") {
    Quantize_fp32_s8(size, src_data, scales, dst_data);
  } else {
    Quantize_fp32_bf16(size, src_data, scales, dst_data);
  }
  return;
}

#else
void Quantize_u8(const int size, const void* src_data, const float* range_mins, const vector<float>& scales,
                 void* dst_data) {
  const float* src_data_ = static_cast<const float*>(src_data);
  unsigned char* dst_data_ = static_cast<unsigned char*>(dst_data);
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    int32_t data = nearbyint((src_data_[i] - range_mins[0]) * scales[0]);
    data = data < 0 ? 0 : data;
    data = data > 255 ? 255 : data;
    dst_data_[i] = static_cast<unsigned char>(data);
  }
}

void Quantize_others(const int size, const string& dtype, const void* src_data, const vector<float>& scales,
                     void* dst_data) {
  const float* src_data_ = static_cast<const float*>(src_data);
  if (dtype == "s8") {
    char* dst_data_ = static_cast<char*>(dst_data);
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
      int32_t data = nearbyint(src_data_[i] * scales[0]);
      data = data < -128 ? -128 : data;
      data = data > 127 ? 127 : data;
      dst_data_[i] = static_cast<char>(data);
    }
  } else if (dtype == "bf16") {
    uint16_t* dst_data_ = static_cast<uint16_t*>(dst_data);
    union {
      unsigned int u;
      float f;
    } typecast;
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
      typecast.f = src_data_[i];
      dst_data_[i] = typecast.u >> 16;
    }
  } else {
    LOG(ERROR) << "Can't suppport dst_dtype: " << dtype << " now!";
  }
  return;
}

void Quantize(const int size, const string& dtype, const void* src_data, const float* range_mins,
              const vector<float>& scales, void* dst_data) {
  if (dtype == "u8") {
    Quantize_u8(size, src_data, range_mins, scales, dst_data);
  } else {
    Quantize_others(size, dtype, src_data, scales, dst_data);
  }
  return;
}
#endif

void BF16_to_FP32(const int size, void* src_data, void* dst_data) {
  union {
    unsigned int u;
    float f;
  } typecast;
  uint16_t* src_data_ = static_cast<uint16_t*>(src_data);
  float* dst_data_ = static_cast<float*>(dst_data);
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    typecast.u = src_data_[i] << 16;
    dst_data_[i] = typecast.f;
  }
}

// Transpose from A to default B.
// e.g.: transpose from {2, 0, 1} to default {0, 1, 2} is {1, 2, 0}
vector<int64_t> ReversePerm(const vector<int64_t>& perm_to) {
  if (perm_to.empty()) return {};
  int dsize = perm_to.size();
  vector<int64_t> perm_from(dsize, 0);
  std::iota(perm_from.begin(), perm_from.end(), 0);
  if (perm_to.empty()) {
    return perm_from;
  }
#pragma omp parallel for
  for (int i = 0; i < dsize; ++i) {
    int index = perm_to[i];
    perm_from[index] = i;
  }
  return perm_from;
}

template <typename T>
T StringToNum(const string& str) {
  std::istringstream iss(str);
  T num;
  iss >> num;
  return num;
}

template float StringToNum<float>(const string& str);
template int64_t StringToNum<int64_t>(const string& str);
template int StringToNum<int>(const string& str);
template size_t StringToNum<size_t>(const string& str);

template <typename T>
float GetSparseRatio(const T* data, const vector<int64_t>& shape, const vector<int64_t>& blocksize) {
  const int64_t blocknum = (shape[0] / blocksize[0]) * (shape[1] / blocksize[1]);
  int64_t zero_count = blocknum;
  for (int64_t b_row = 0; b_row < shape[0] / blocksize[0]; b_row++) {
    for (int64_t b_col = 0; b_col < shape[1] / blocksize[1]; b_col++) {
      const T* dense_start = data + b_row * blocksize[0] * shape[1] + b_col * blocksize[1];
      bool not_zero = false;
      for (int64_t i = 0; i < blocksize[0]; i++) {
        for (int64_t j = 0; j < blocksize[1]; j++) {
          if (dense_start[i * shape[1] + j] != 0) {
            zero_count--;
            not_zero = true;
            break;
          }
        }
        if (not_zero) {
          break;
        }
      }
    }
  }
  float zero_ratio = blocknum == 0 ? 0 : static_cast<float>(zero_count) / blocknum;
  return zero_ratio;
}
template float GetSparseRatio<float>(const float* data, const vector<int64_t>& shape, const vector<int64_t>& blocksize);
template float GetSparseRatio<int8_t>(const int8_t* data, const vector<int64_t>& shape,
                                      const vector<int64_t>& blocksize);

template <typename T>
void PrintToFile(const T* data, const std::string& name, size_t size) {
  // print output file
  auto pos = name.rfind("/");
  string output_file = (pos != string::npos ? name.substr(pos + 1) : name) + ".txt";
  std::ofstream output_data(output_file);
  for (size_t i = 0; i < size; ++i) {
    output_data << static_cast<float>(data[i]) << "\n";
  }
  output_data.close();
}
template void PrintToFile<float>(const float* data, const std::string& name, size_t size);
template void PrintToFile<unsigned char>(const unsigned char* data, const std::string& name, size_t size);
template void PrintToFile<char>(const char* data, const std::string& name, size_t size);
template void PrintToFile<int32_t>(const int32_t* data, const std::string& name, size_t size);
template void PrintToFile<int64_t>(const int64_t* data, const std::string& name, size_t size);

template <typename T>
void StringSplit(vector<T>* split_list, const string& str_list, const string& split_op) {
  std::string::size_type pos1 = 0;
  std::string::size_type pos2 = str_list.find(split_op);
  while (std::string::npos != pos2) {
    T element = StringToNum<T>(str_list.substr(pos1, pos2));
    split_list->push_back(element);
    pos1 = pos2 + split_op.size();
    pos2 = str_list.find(split_op, pos1);
  }
  if (pos1 != str_list.size()) {
    T element = StringToNum<T>(str_list.substr(pos1));
    split_list->push_back(element);
  }
}
template void StringSplit<float>(vector<float>* split_list, const string& string_list, const string& split_op);
template void StringSplit<int64_t>(vector<int64_t>* split_list, const string& string_list, const string& split_op);
template void StringSplit<char>(vector<char>* split_list, const string& string_list, const string& split_op);
template void StringSplit<unsigned char>(vector<unsigned char>* split_list, const string& string_list,
                                         const string& split_op);

void InitSparse(int K, int N, int N_BLKSIZE, int K_BLKSIZE, int N_SPARSE, float* B) {
  unsigned int seed = 0;
  srand(seed);
  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
      B[k * N + n] = rand() % 11 - 5;  // NOLINT
    }
  }
  // sparsify B
  for (int nb = 0; nb < N / N_BLKSIZE; nb++) {
    for (int kb = 0; kb < K / K_BLKSIZE; kb++) {
      bool zero_fill = rand() % N_SPARSE != 0;  // NOLINT
      if (zero_fill) {
        for (int n = 0; n < N_BLKSIZE; n++) {
          for (int k = 0; k < K_BLKSIZE; k++) {
            B[(kb * K_BLKSIZE + k) * N + nb * N_BLKSIZE + n] = 0;
          }
        }
      }
    }
  }
}

/************* ref ************/
template <typename dst_type, typename src_type>
void ref_mov_ker(dst_type* inout, const src_type* in, size_t len) {
#pragma omp parallel for
  for (int i = 0; i < len; i++) {
    *(inout + i) = *(in + i);
  }
}
template void ref_mov_ker(float* inout, const float* in, size_t len);
template void ref_mov_ker(uint16_t* inout, const uint16_t* in, size_t len);
template void ref_mov_ker(uint8_t* inout, const uint8_t* in, size_t len);

template <typename dst_type, typename src_type>
void ref_add_ker(dst_type* inout, src_type* in, size_t len) {
#pragma omp parallel for
  for (int i = 0; i < len; i++) {
    *(inout + i) += *(in + i);
  }
}
template void ref_add_ker(float* inout, float* in, size_t len);
template void ref_add_ker(uint16_t* inout, uint16_t* in, size_t len);
template void ref_add_ker(uint8_t* inout, uint8_t* in, size_t len);

/************* fp32 ************/
void zero_ker(float* out, size_t len) {
  int64_t i = 0;
#if __AVX512F__
  __m512 zero_512 = _mm512_setzero_ps();
  for (i = 0; i < len - 15; i += 16) {
    _mm512_storeu_ps(out + i, zero_512);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    _mm512_mask_storeu_ps(out + i, mask, zero_512);
  }
#else
  memset(out, 0, len * sizeof(float));
#endif
}

void move_ker(float* out, const float* in, size_t len) {
  int64_t i = 0;
#if __AVX512F__
  for (i = 0; i < len - 15; i += 16) {
    auto in0 = _mm512_loadu_ps(in + i);
    _mm512_storeu_ps(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_ps(mask, in + i);
    _mm512_mask_storeu_ps(out + i, mask, in0);
  }
#else
  ref_mov_ker(out, in, len);
#endif
}

void add_ker(float* inout, float* in, size_t len) {
  int i = 0;
#if __AVX512F__
  for (i = 0; i < len - 31; i += 32) {
    auto out1 = _mm512_loadu_ps(inout + i);
    auto out2 = _mm512_loadu_ps(inout + i + 16);
    auto in1 = _mm512_loadu_ps(in + i);
    auto in2 = _mm512_loadu_ps(in + i + 16);
    out1 = _mm512_add_ps(out1, in1);
    out2 = _mm512_add_ps(out2, in2);
    _mm512_storeu_ps(inout + i, out1);
    _mm512_storeu_ps(inout + i + 16, out2);
  }

  if (i < len - 15) {
    auto out1 = _mm512_loadu_ps(inout + i);
    auto in1 = _mm512_loadu_ps(in + i);
    _mm512_storeu_ps(inout + i, _mm512_add_ps(out1, in1));
    i += 16;
  }

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto out1 = _mm512_maskz_loadu_ps(mask, inout + i);
    auto in1 = _mm512_maskz_loadu_ps(mask, in + i);
    _mm512_mask_storeu_ps(inout + i, mask, _mm512_add_ps(out1, in1));
  }
#else
  ref_add_ker(inout, in, len);
#endif
}

/************* bf16 ************/
#ifdef __AVX512F__
// Conversion from BF16 to FP32
inline __m512 cvt_bf16_to_fp32(const __m256i src) {
  auto y = _mm512_cvtepu16_epi32(src);
  return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
}

inline __m512 cvt_bf16_to_fp32(__mmask16 mask, const __m256i src) {
  __m512i y = _mm512_maskz_cvtepu16_epi32(mask, src);
  auto x = _mm512_bslli_epi128(y, 2);
  return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
}

// Conversion from FP32 to BF16
inline __m256i trunc_fp32_to_bf16(const __m512 src) {
  auto y = _mm512_bsrli_epi128(_mm512_castps_si512(src), 2);
  return _mm512_cvtepi32_epi16(y);
}

__m256i cvt_fp32_to_bf16(const __m512 src) {
#if __AVX512BF16__ && __GNUC__ > 11
  return (__m256i)_mm512_cvtneps_pbh(src);
#else
  return trunc_fp32_to_bf16(src);
#endif
}
#elif __AVX2__
const uint8_t avx2_bf16_convert_maigc_num[32] = {0x02, 0x03, 0x06, 0x07, 0x0a, 0x0b, 0x0e, 0x0f, 0x80, 0x80, 0x80,
                                                 0x80, 0x80, 0x80, 0x80, 0x80, 0x02, 0x03, 0x06, 0x07, 0x0a, 0x0b,
                                                 0x0e, 0x0f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80};
__m128i cvt_fp32_to_bf16(const __m256 src) {
  auto shuffle_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(avx2_bf16_convert_maigc_num));
  __m256i trunc_elements = _mm256_shuffle_epi8(_mm256_castps_si256(src), shuffle_v);
  __m256i ordered = _mm256_permute4x64_epi64(trunc_elements, 0x58);
  return _mm256_castsi256_si128(ordered);
}
#endif

void zero_ker(uint16_t* out, size_t len) {
  int64_t i = 0;
#if __AVX512F__
  __m512i zero_512 = _mm512_setzero_si512();
  for (i = 0; i < len - 31; i += 32) {
    _mm512_storeu_si512(out + i, zero_512);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    _mm512_mask_storeu_epi16(out + i, mask, zero_512);
  }
#else
  memset(out, 0, len * sizeof(uint16_t));
#endif
}

void move_ker(uint16_t* out, const uint16_t* in, size_t len) {
  int64_t i = 0;
#if __AVX512F__
  for (i = 0; i < len - 31; i += 32) {
    auto in0 = _mm512_loadu_si512(in + i);
    _mm512_storeu_si512(out + i, in0);
  }

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto in0 = _mm512_maskz_loadu_epi16(mask, in + i);
    _mm512_mask_storeu_epi16(out + i, mask, in0);
  }
#else
  ref_mov_ker(out, in, len);
#endif
}

void add_ker(uint16_t* inout, uint16_t* in, size_t len) {
  int i = 0;
#if __AVX512F__
  for (i = 0; i < len - 31; i += 32) {
    auto inout1 = cvt_bf16_to_fp32(_mm256_loadu_si256(reinterpret_cast<__m256i*>(inout + i)));
    auto inout2 = cvt_bf16_to_fp32(_mm256_loadu_si256(reinterpret_cast<__m256i*>(inout + i + 16)));
    auto in1 = cvt_bf16_to_fp32(_mm256_loadu_si256(reinterpret_cast<__m256i*>(in + i)));
    auto in2 = cvt_bf16_to_fp32(_mm256_loadu_si256(reinterpret_cast<__m256i*>(in + i + 16)));
    inout1 = _mm512_add_ps(inout1, in1);
    inout2 = _mm512_add_ps(inout2, in2);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(inout + i), cvt_fp32_to_bf16(inout1));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(inout + i + 16), cvt_fp32_to_bf16(inout2));
  }

  if (i < len - 15) {
    auto inout1 = cvt_bf16_to_fp32(_mm256_loadu_si256(reinterpret_cast<__m256i*>(inout + i)));
    auto in1 = cvt_bf16_to_fp32(_mm256_loadu_si256(reinterpret_cast<__m256i*>(in + i)));
    inout1 = _mm512_add_ps(inout1, in1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(inout + i), cvt_fp32_to_bf16(inout1));
    i += 16;
  }

  if (i < len) {
    auto mask = (1 << (len - i)) - 1;
    auto inout1 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, inout + i));
    auto in1 = cvt_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, in + i));
    inout1 = _mm512_add_ps(inout1, in1);
    _mm256_mask_storeu_epi16(inout + i, mask, cvt_fp32_to_bf16(inout1));
  }
#else
  ref_add_ker(inout, in, len);
#endif
}

/************* uint8 ************/
void zero_ker(uint8_t* out, size_t len) {
  int64_t i;
#if __AVX512F__
  __m512i zero_512 = _mm512_setzero_si512();
  for (i = 0; i < len - 63; i += 64) {
    _mm512_storeu_si512(out + i, zero_512);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    _mm512_mask_storeu_epi8(out + i, mask, zero_512);
  }
#else
  memset(out, 0, sizeof(uint8_t) * len);
#endif
}

void move_ker(uint8_t* out, const uint8_t* in, size_t len) {
  int64_t i;
#if __AVX512F__
  for (i = 0; i < len - 63; i += 64) {
    auto in0 = _mm512_loadu_si512(in + i);
    _mm512_storeu_si512(out + i, in0);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_epi8(mask, in + i);
    _mm512_mask_storeu_epi8(out + i, mask, in0);
  }
#else
  ref_mov_ker(out, in, len);
#endif
}

void add_ker(uint8_t* inout, uint8_t* in, size_t len) {
  int64_t i;
#if __AVX512F__
  for (i = 0; i < len - 63; i += 64) {
    auto in0 = _mm512_loadu_si512(in + i);
    auto out = _mm512_loadu_si512(inout + i);
    out = _mm512_adds_epi8(out, in0);  // add with saturate
    _mm512_storeu_si512(inout + i, out);
  }

  if (i < len) {
    auto mask = ((1 << (len - i)) - 1);
    auto in0 = _mm512_maskz_loadu_epi8(mask, in + i);
    auto out = _mm512_maskz_loadu_epi8(mask, inout + i);
    out = _mm512_adds_epi8(out, in0);
    _mm512_mask_storeu_epi8(inout + i, mask, out);
  }
#else
  ref_add_ker(inout, in, len);
#endif
}

void runtime_minmax(const float* data, size_t length, float* min_num, float* max_num) {
  int block_size = (length / CPU_COUNT) / ALIGN_NUM * ALIGN_NUM;
  if (block_size == 0) {
    auto result = std::minmax_element(data, data + length);
    *min_num = *result.first;
    *max_num = *result.second;
    return;
  }
  int block_num = length / block_size;
  vector<float> block_mins(block_num + (length % block_size != 0)), block_maxs(block_num + (length % block_size != 0));
#if __AVX512F__
#pragma omp parallel for
  for (int i = 0; i < block_num; i++) {
    block_minmax_avx512(data + i * block_size, block_size, &block_mins[i], &block_maxs[i]);
  }
  if (length % block_size != 0) {
    block_minmax_avx512(data + block_num * block_size, length - block_num * block_size, &block_mins[block_num],
                        &block_maxs[block_num]);
  }
#else
#pragma omp parallel for
  for (int i = 0; i < block_num; i++) {
    block_minmax(data + i * block_size, block_size, &block_mins[i], &block_maxs[i]);
  }
  if (length % block_size != 0) {
    block_minmax(data + block_num * block_size, length - block_num * block_size, &block_mins[block_num],
                 &block_maxs[block_num]);
  }
#endif
  *min_num = *std::min_element(block_mins.begin(), block_mins.end());
  *max_num = *std::max_element(block_maxs.begin(), block_maxs.end());
}

#ifdef __AVX512F__
void block_minmax_avx512(const float* Input, size_t N, float* Min, float* Max) {
  float tmp_min = std::numeric_limits<float>::max();
  float tmp_max = std::numeric_limits<float>::lowest();

  if (N >= 16) {
    __m512 MaximumVector0 = _mm512_set1_ps(tmp_max);
    __m512 MinimumVector0 = _mm512_set1_ps(tmp_min);

    if (N >= 64) {
      __m512 MaximumVector1 = MaximumVector0;
      __m512 MaximumVector2 = MaximumVector0;
      __m512 MaximumVector3 = MaximumVector0;

      __m512 MinimumVector1 = MinimumVector0;
      __m512 MinimumVector2 = MinimumVector0;
      __m512 MinimumVector3 = MinimumVector0;

      while (N >= 64) {
        __m512 InputVector0 = _mm512_loadu_ps(Input);
        __m512 InputVector1 = _mm512_loadu_ps(Input + 16);
        __m512 InputVector2 = _mm512_loadu_ps(Input + 32);
        __m512 InputVector3 = _mm512_loadu_ps(Input + 48);

        MaximumVector0 = _mm512_max_ps(MaximumVector0, InputVector0);
        MaximumVector1 = _mm512_max_ps(MaximumVector1, InputVector1);
        MaximumVector2 = _mm512_max_ps(MaximumVector2, InputVector2);
        MaximumVector3 = _mm512_max_ps(MaximumVector3, InputVector3);

        MinimumVector0 = _mm512_min_ps(MinimumVector0, InputVector0);
        MinimumVector1 = _mm512_min_ps(MinimumVector1, InputVector1);
        MinimumVector2 = _mm512_min_ps(MinimumVector2, InputVector2);
        MinimumVector3 = _mm512_min_ps(MinimumVector3, InputVector3);

        Input += 64;
        N -= 64;
      }

      MaximumVector0 = _mm512_max_ps(MaximumVector0, MaximumVector1);
      MaximumVector2 = _mm512_max_ps(MaximumVector2, MaximumVector3);
      MaximumVector0 = _mm512_max_ps(MaximumVector0, MaximumVector2);

      MinimumVector0 = _mm512_min_ps(MinimumVector0, MinimumVector1);
      MinimumVector2 = _mm512_min_ps(MinimumVector2, MinimumVector3);
      MinimumVector0 = _mm512_min_ps(MinimumVector0, MinimumVector2);
    }

    while (N >= 16) {
      __m512 InputVector0 = _mm512_loadu_ps(Input);
      MaximumVector0 = _mm512_max_ps(MaximumVector0, InputVector0);
      MinimumVector0 = _mm512_min_ps(MinimumVector0, InputVector0);

      Input += 16;
      N -= 16;
    }

    float minx16[32];
    void* min_ptr = reinterpret_cast<void*>(minx16);
    std::size_t min_space = sizeof(minx16) - 1;
    std::align(64, sizeof(float), min_ptr, min_space);
    float* aligned_min = reinterpret_cast<float*>(min_ptr);
    _mm512_store_ps(aligned_min, MinimumVector0);
    float maxx16[32];
    void* max_ptr = reinterpret_cast<void*>(maxx16);
    std::size_t max_space = sizeof(maxx16) - 1;
    std::align(64, sizeof(float), max_ptr, max_space);
    float* aligned_max = reinterpret_cast<float*>(max_ptr);
    _mm512_store_ps(aligned_max, MaximumVector0);
    for (int i = 0; i < 16; i++) {
      tmp_max = std::max(tmp_max, aligned_max[i]);
      tmp_min = std::min(tmp_min, aligned_min[i]);
    }
  }

  while (N > 0) {
    tmp_max = std::max(tmp_max, *Input);
    tmp_min = std::min(tmp_min, *Input);

    Input += 1;
    N -= 1;
  }

  *Min = tmp_min;
  *Max = tmp_max;
}
#else
void block_minmax(const float* Input, size_t N, float* Min, float* Max) {
  float tmp_min = std::numeric_limits<float>::max();
  float tmp_max = std::numeric_limits<float>::lowest();

  if (N >= 8) {
    __m256 MaximumVector0 = _mm256_set1_ps(tmp_max);
    __m256 MinimumVector0 = _mm256_set1_ps(tmp_min);

    if (N >= 32) {
      __m256 MaximumVector1 = MaximumVector0;
      __m256 MaximumVector2 = MaximumVector0;
      __m256 MaximumVector3 = MaximumVector0;

      __m256 MinimumVector1 = MinimumVector0;
      __m256 MinimumVector2 = MinimumVector0;
      __m256 MinimumVector3 = MinimumVector0;

      while (N >= 32) {
        __m256 InputVector0 = _mm256_loadu_ps(Input);
        __m256 InputVector1 = _mm256_loadu_ps(Input + 8);
        __m256 InputVector2 = _mm256_loadu_ps(Input + 16);
        __m256 InputVector3 = _mm256_loadu_ps(Input + 24);

        MaximumVector0 = _mm256_max_ps(MaximumVector0, InputVector0);
        MaximumVector1 = _mm256_max_ps(MaximumVector1, InputVector1);
        MaximumVector2 = _mm256_max_ps(MaximumVector2, InputVector2);
        MaximumVector3 = _mm256_max_ps(MaximumVector3, InputVector3);

        MinimumVector0 = _mm256_min_ps(MinimumVector0, InputVector0);
        MinimumVector1 = _mm256_min_ps(MinimumVector1, InputVector1);
        MinimumVector2 = _mm256_min_ps(MinimumVector2, InputVector2);
        MinimumVector3 = _mm256_min_ps(MinimumVector3, InputVector3);

        Input += 32;
        N -= 32;
      }

      MaximumVector0 = _mm256_max_ps(MaximumVector0, MaximumVector1);
      MaximumVector2 = _mm256_max_ps(MaximumVector2, MaximumVector3);
      MaximumVector0 = _mm256_max_ps(MaximumVector0, MaximumVector2);

      MinimumVector0 = _mm256_min_ps(MinimumVector0, MinimumVector1);
      MinimumVector2 = _mm256_min_ps(MinimumVector2, MinimumVector3);
      MinimumVector0 = _mm256_min_ps(MinimumVector0, MinimumVector2);
    }

    while (N >= 8) {
      __m256 InputVector0 = _mm256_loadu_ps(Input);
      MaximumVector0 = _mm256_max_ps(MaximumVector0, InputVector0);
      MinimumVector0 = _mm256_min_ps(MinimumVector0, InputVector0);

      Input += 8;
      N -= 8;
    }

    float minx8[16];
    void* min_ptr = reinterpret_cast<void*>(minx8);
    std::size_t min_space = sizeof(minx8) - 1;
    std::align(32, sizeof(float), min_ptr, min_space);
    float* aligned_min = reinterpret_cast<float*>(min_ptr);
    _mm256_store_ps(aligned_min, MinimumVector0);
    float maxx8[16];
    void* max_ptr = reinterpret_cast<void*>(maxx8);
    std::size_t max_space = sizeof(maxx8) - 1;
    std::align(32, sizeof(float), max_ptr, max_space);
    float* aligned_max = reinterpret_cast<float*>(max_ptr);
    _mm256_store_ps(aligned_max, MaximumVector0);
    for (int i = 0; i < 8; i++) {
      tmp_max = std::max(tmp_max, aligned_max[i]);
      tmp_min = std::min(tmp_min, aligned_min[i]);
    }
  }

  while (N > 0) {
    tmp_max = std::max(tmp_max, *Input);
    tmp_min = std::min(tmp_min, *Input);

    Input += 1;
    N -= 1;
  }

  *Min = tmp_min;
  *Max = tmp_max;
}
#endif

/************ InnerProductPrimitiveFwdFactory member function ************/
size_t InnerProductPrimitiveFwdFactory::GenKey(const string& src0_dtype, const string& src1_dtype,
                                               const string& dst_dtype, const vector<int64_t>& src0_shape,
                                               const vector<int64_t>& src1_shape, const vector<int64_t>& dst_perm,
                                               const string& append_op, const vector<int64_t>& post_op_shape,
                                               const float& output_scale, const dnnl::engine* eng) {
  size_t seed = 0;
  // primitive kind
  string prefix = "inner_product_fwd_";
  seed = hash_val(prefix, src0_dtype, src1_dtype, dst_dtype);
  seed = get_array_hash(seed, src0_shape, src0_shape.size());
  seed = get_array_hash(seed, src1_shape, src1_shape.size());
  // if dst_shape has reverse_perm
  if (!dst_perm.empty()) {
    seed = get_array_hash(seed, dst_perm, dst_perm.size());
  }
  if (append_op != "") {
    seed = hash_combine(seed, append_op);
    if (append_op == "sum" || append_op == "binary_add") {
      seed = get_array_hash(seed, post_op_shape, post_op_shape.size());
    }
  }
  // if has output_scale
  if (output_scale != 1.f) {
    seed = hash_combine(seed, output_scale);
  }
  // hash each dnnl engine ptr
  seed = hash_combine(seed, eng);
  return seed;
}

size_t InnerProductPrimitiveFwdFactory::Key(const string& src0_dtype, const string& src1_dtype, const string& dst_dtype,
                                            const vector<int64_t>& src0_shape, const vector<int64_t>& src1_shape,
                                            const vector<int64_t>& dst_perm, const string& append_op,
                                            const vector<int64_t>& post_op_shape, const float& output_scale,
                                            const dnnl::engine* eng) {
  return InnerProductPrimitiveFwdFactory::GetInstance().GenKey(
      src0_dtype, src1_dtype, dst_dtype, src0_shape, src1_shape, dst_perm, append_op, post_op_shape, output_scale, eng);
}

bool InnerProductPrimitiveFwdFactory::IsInFactory(const size_t& key) {
  return InnerProductPrimitiveFwdFactory::GetInstance().IsInCache(key);
}

dnnl::inner_product_forward& InnerProductPrimitiveFwdFactory::Get(const size_t& key) {
  return static_cast<dnnl::inner_product_forward&>(InnerProductPrimitiveFwdFactory::GetInstance().GetPrimitive(key));
}

void InnerProductPrimitiveFwdFactory::Set(const size_t& key, dnnl::primitive primitive) {
  InnerProductPrimitiveFwdFactory::GetInstance().SetPrimitive(key, primitive);
}

void InnerProductPrimitiveFwdFactory::ClearFactory() { InnerProductPrimitiveFwdFactory::GetInstance().Clear(); }

bool InnerProductPrimitiveFwdFactory::DoNotCache() {
  return InnerProductPrimitiveFwdFactory::GetInstance().do_not_cache_;
}

InnerProductPrimitiveFwdFactory& InnerProductPrimitiveFwdFactory::GetInstance() {
  static InnerProductPrimitiveFwdFactory instance_;
  return instance_;
}

/************ MatMulPrimitiveFwdFactory member function ************/
size_t MatMulPrimitiveFwdFactory::GenKey(const string& src0_dtype, const string& src1_dtype, const string& dst_dtype,
                                         const vector<int64_t>& src0_shape, const vector<int64_t>& src1_shape,
                                         const vector<int64_t>& src0_perm, const vector<int64_t>& src1_perm,
                                         const vector<int64_t>& dst_perm, const string& append_op,
                                         const vector<int64_t>& post_op_shape, const float& output_scale,
                                         const dnnl::engine* eng) {
  size_t seed = 0;
  // primitive kind
  string prefix = "matmul_fwd_";
  seed = hash_val(prefix, src0_dtype, src1_dtype, dst_dtype);
  seed = get_array_hash(seed, src0_shape, src0_shape.size());
  seed = get_array_hash(seed, src1_shape, src1_shape.size());
  if (!src0_perm.empty()) {
    seed = get_array_hash(seed, src0_perm, src0_perm.size());
  }
  if (!src1_perm.empty()) {
    seed = get_array_hash(seed, src1_perm, src1_perm.size());
  }
  // if dst_shape has reverse_perm
  if (!dst_perm.empty()) {
    seed = get_array_hash(seed, dst_perm, dst_perm.size());
  }
  if (append_op != "") {
    seed = hash_combine(seed, append_op);
    if (append_op == "sum" || append_op == "binary_add") {
      seed = get_array_hash(seed, post_op_shape, post_op_shape.size());
    }
  }
  // if has output_scale
  if (output_scale != 1.f) {
    seed = hash_combine(seed, output_scale);
  }
  // hash each dnnl engine ptr
  seed = hash_combine(seed, eng);
  return seed;
}

size_t MatMulPrimitiveFwdFactory::Key(const string& src0_dtype, const string& src1_dtype, const string& dst_dtype,
                                      const vector<int64_t>& src0_shape, const vector<int64_t>& src1_shape,
                                      const vector<int64_t>& src0_perm, const vector<int64_t>& src1_perm,
                                      const vector<int64_t>& dst_perm, const string& append_op,
                                      const vector<int64_t>& post_op_shape, const float& output_scale,
                                      const dnnl::engine* eng) {
  return MatMulPrimitiveFwdFactory::GetInstance().GenKey(src0_dtype, src1_dtype, dst_dtype, src0_shape, src1_shape,
                                                         src0_perm, src1_perm, dst_perm, append_op, post_op_shape,
                                                         output_scale, eng);
}

bool MatMulPrimitiveFwdFactory::IsInFactory(const size_t& key) {
  return MatMulPrimitiveFwdFactory::GetInstance().IsInCache(key);
}

dnnl::matmul& MatMulPrimitiveFwdFactory::Get(const size_t& key) {
  return static_cast<dnnl::matmul&>(MatMulPrimitiveFwdFactory::GetInstance().GetPrimitive(key));
}

void MatMulPrimitiveFwdFactory::Set(const size_t& key, dnnl::primitive primitive) {
  MatMulPrimitiveFwdFactory::GetInstance().SetPrimitive(key, primitive);
}

void MatMulPrimitiveFwdFactory::ClearFactory() { MatMulPrimitiveFwdFactory::GetInstance().Clear(); }

bool MatMulPrimitiveFwdFactory::DoNotCache() { return MatMulPrimitiveFwdFactory::GetInstance().do_not_cache_; }

MatMulPrimitiveFwdFactory& MatMulPrimitiveFwdFactory::GetInstance() {
  static MatMulPrimitiveFwdFactory instance_;
  return instance_;
}

/************ ConvolutionPrimitiveFwdFactory member function ************/
size_t ConvolutionPrimitiveFwdFactory::GenKey(const string& src0_dtype, const string& src1_dtype,
                                              const string& dst_dtype, const vector<int64_t>& src0_shape,
                                              const vector<int64_t>& src1_shape, const vector<int64_t>& dst_perm,
                                              const string& append_op, const vector<int64_t>& post_op_shape,
                                              const float& output_scale, const int64_t& group,
                                              const vector<int64_t>& pads, const vector<int64_t>& strides,
                                              const dnnl::engine* eng) {
  size_t seed = 0;
  // primitive kind
  string prefix = "convolution_fwd_";
  seed = hash_val(prefix, src0_dtype, src1_dtype, dst_dtype);
  seed = get_array_hash(seed, src0_shape, src0_shape.size());
  seed = get_array_hash(seed, src1_shape, src1_shape.size());
  // if dst_shape has reverse_perm
  if (!dst_perm.empty()) {
    seed = get_array_hash(seed, dst_perm, dst_perm.size());
  }
  if (append_op != "") {
    seed = hash_combine(seed, append_op);
    if (append_op == "sum" || append_op == "binary_add") {
      seed = get_array_hash(seed, post_op_shape, post_op_shape.size());
    }
  }
  // if has output_scale
  if (output_scale != 1.f) {
    seed = hash_combine(seed, output_scale);
  }
  seed = hash_combine(seed, group);
  if (!pads.empty()) seed = get_array_hash(seed, pads, pads.size());
  if (!strides.empty()) seed = get_array_hash(seed, strides, strides.size());
  // hash each dnnl engine ptr
  seed = hash_combine(seed, eng);
  return seed;
}

size_t ConvolutionPrimitiveFwdFactory::Key(const string& src0_dtype, const string& src1_dtype, const string& dst_dtype,
                                           const vector<int64_t>& src0_shape, const vector<int64_t>& src1_shape,
                                           const vector<int64_t>& dst_perm, const string& append_op,
                                           const vector<int64_t>& post_op_shape, const float& output_scale,
                                           const int64_t& group, const vector<int64_t>& pads,
                                           const vector<int64_t>& strides, const dnnl::engine* eng) {
  return ConvolutionPrimitiveFwdFactory::GetInstance().GenKey(src0_dtype, src1_dtype, dst_dtype, src0_shape, src1_shape,
                                                              dst_perm, append_op, post_op_shape, output_scale, group,
                                                              pads, strides, eng);
}

bool ConvolutionPrimitiveFwdFactory::IsInFactory(const size_t& key) {
  return ConvolutionPrimitiveFwdFactory::GetInstance().IsInCache(key);
}

dnnl::convolution_forward& ConvolutionPrimitiveFwdFactory::Get(const size_t& key) {
  return static_cast<dnnl::convolution_forward&>(ConvolutionPrimitiveFwdFactory::GetInstance().GetPrimitive(key));
}

void ConvolutionPrimitiveFwdFactory::Set(const size_t& key, dnnl::primitive primitive) {
  ConvolutionPrimitiveFwdFactory::GetInstance().SetPrimitive(key, primitive);
}

void ConvolutionPrimitiveFwdFactory::ClearFactory() { ConvolutionPrimitiveFwdFactory::GetInstance().Clear(); }

bool ConvolutionPrimitiveFwdFactory::DoNotCache() {
  return ConvolutionPrimitiveFwdFactory::GetInstance().do_not_cache_;
}

ConvolutionPrimitiveFwdFactory& ConvolutionPrimitiveFwdFactory::GetInstance() {
  static ConvolutionPrimitiveFwdFactory instance_;
  return instance_;
}

}  // namespace executor
