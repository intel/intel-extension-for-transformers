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

#include "group_norm.hpp"
namespace executor {

#define SIMD_SUM                                      \
  *zmm_sum_x = _mm512_add_ps(*zmm_sum_x, zmm_src);    \
  auto zmm_pow_src = _mm512_mul_ps(zmm_src, zmm_src); \
  *zmm_sum_powx = _mm512_add_ps(*zmm_sum_powx, zmm_pow_src);

#define SIMD_SUM_MASK                                                     \
  *zmm_sum_x = _mm512_mask_add_ps(*zmm_sum_x, mask, *zmm_sum_x, zmm_src); \
  auto zmm_pow_src = _mm512_mul_ps(zmm_src, zmm_src);                     \
  *zmm_sum_powx = _mm512_mask_add_ps(*zmm_sum_powx, mask, *zmm_sum_powx, zmm_pow_src);

void fp32_sum(int64_t norm_dim_elt_num, int dt_bytewidth, char* src, __m512* zmm_sum_x, __m512* zmm_sum_powx) {
  int64_t i = 0;
  int tail = norm_dim_elt_num % 16;
  for (; i < norm_dim_elt_num / 16; i++) {
    auto zmm_src = _mm512_loadu_ps(static_cast<float*>(static_cast<void*>((src + i * 16 * dt_bytewidth))));
    SIMD_SUM
  }
  if (tail != 0) {
    auto mask = _cvtu32_mask16(0xffff >> (16 - tail));
    auto zmm_src = _mm512_loadu_ps(static_cast<float*>(static_cast<void*>((src + i * 16 * dt_bytewidth))));
    SIMD_SUM_MASK
  }
}

inline __m512 bf16_load(float* addr) {
  auto bf16_data = _mm256_castps_si256(_mm256_loadu_ps(addr));
  auto shift_data = _mm512_cvtepu16_epi32(bf16_data);
  return _mm512_castsi512_ps(_mm512_slli_epi32(shift_data, 0x10));
}

void bf16_sum(int64_t norm_dim_elt_num, int dt_bytewidth, char* src, __m512* zmm_sum_x, __m512* zmm_sum_powx) {
  int64_t i = 0;
  int tail = norm_dim_elt_num % 16;
  for (int64_t i = 0; i < norm_dim_elt_num / 16; i++) {
    __m512 zmm_src = bf16_load(static_cast<float*>(static_cast<void*>((src + i * 16 * dt_bytewidth))));
    SIMD_SUM
  }
  if (tail != 0) {
    auto mask = _cvtu32_mask16(0xffff >> (16 - tail));
    __m512 zmm_src = bf16_load(static_cast<float*>(static_cast<void*>((src + i * 16 * dt_bytewidth))));
    SIMD_SUM_MASK
  }
}

#define SIMD_NORM_OFFSET                                               \
  auto zmm_gamma = _mm512_set1_ps(gamma_data[i]);                      \
  zmm_gamma = _mm512_mul_ps(zmm_gamma, *zmm_rsqrt14_var);              \
  auto zmm_beta = _mm512_set1_ps(beta_data[i]);                        \
  char* cur_channel_src = cur_group_src + i * map_size * dt_bytewidth; \
  char* cur_channel_dst = cur_group_dst + i * map_size * dt_bytewidth;

void fp32_norm(int map_size, int dt_bytewidth, int channels_per_group, const float* gamma_data, const float* beta_data,
               char* cur_group_src, char* cur_group_dst, __m512* zmm_rsqrt14_var, __m512* zmm_mean) {
  int tail = map_size % 16;
  for (int64_t i = 0; i < channels_per_group; i++) {
    SIMD_NORM_OFFSET
    int64_t j = 0;
    auto norm = [&] {
      auto zmm_dst =
          _mm512_loadu_ps(static_cast<float*>(static_cast<void*>((cur_channel_src + j * 16 * dt_bytewidth))));
      zmm_dst = _mm512_sub_ps(zmm_dst, *zmm_mean);
      return _mm512_fmadd_ps(zmm_dst, zmm_gamma, zmm_beta);
    };
    for (; j < map_size / 16; j++) {
      auto zmm_dst = norm();
      _mm512_storeu_ps(static_cast<float*>(static_cast<void*>((cur_channel_dst + j * 16 * dt_bytewidth))), zmm_dst);
    }
    if (tail != 0) {
      auto zmm_dst = norm();
      auto mask = _cvtu32_mask16(0xffff >> (16 - tail));
      _mm512_mask_storeu_ps(static_cast<float*>(static_cast<void*>((cur_channel_dst + j * 16 * dt_bytewidth))), mask,
                            zmm_dst);
    }
  }
}

void bf16_norm(int map_size, int dt_bytewidth, int channels_per_group, const float* gamma_data, const float* beta_data,
               char* cur_group_src, char* cur_group_dst, __m512* zmm_rsqrt14_var, __m512* zmm_mean) {
  int tail = map_size % 16;
  for (int64_t i = 0; i < channels_per_group; i++) {
    SIMD_NORM_OFFSET
    auto norm = [&] {
      __m512 zmm_dst = bf16_load(static_cast<float*>(static_cast<void*>((cur_channel_src + i * 16 * dt_bytewidth))));
      zmm_dst = _mm512_sub_ps(zmm_dst, *zmm_mean);
      zmm_dst = _mm512_fmadd_ps(zmm_dst, zmm_gamma, zmm_beta);
      auto zmm_shift = _mm512_srli_epi32(_mm512_castps_si512(zmm_dst), 0x10);
      return _mm512_cvtepi32_epi16(zmm_shift);
    };
    int64_t j = 0;
    for (; j < map_size / 16; j++) {
      auto ymm_bf16 = norm();
      _mm256_storeu_ps(static_cast<float*>(static_cast<void*>((cur_channel_dst + j * 16 * dt_bytewidth))),
                       _mm256_castsi256_ps(ymm_bf16));
    }
    if (tail != 0) {
      auto ymm_bf16 = norm();
      auto mask = _cvtu32_mask16(0xffff >> (16 - tail));
      _mm256_mask_storeu_ps(static_cast<float*>(static_cast<void*>((cur_channel_dst + j * 16 * dt_bytewidth))), mask,
                            _mm256_castsi256_ps(ymm_bf16));
    }
  }
}

void GroupNormOperator::NormGroup(char* cur_group_src, const float* gamma_data, const float* beta_data,
                                  char* cur_group_dst, int map_size) {
  int64_t norm_dim_elt_num = channels_per_group_ * map_size;
  float div_const = 1.f / norm_dim_elt_num;
  auto zmm_sum_x = _mm512_set1_ps(0.f);
  auto zmm_sum_powx = _mm512_set1_ps(0.f);
  sum_func(norm_dim_elt_num, dt_bytewidth_, cur_group_src, &zmm_sum_x, &zmm_sum_powx);
  auto reduce_sum = _mm512_reduce_add_ps(zmm_sum_x);
  auto reduce_powsum = _mm512_reduce_add_ps(zmm_sum_powx);
  float mean = reduce_sum * div_const;
  float pow_mean = mean * mean;
  float powx_mean = reduce_powsum * div_const;
  float var = powx_mean - pow_mean + epsilon_;
  auto zmm_mean = _mm512_set1_ps(mean);
  auto zmm_var = _mm512_set1_ps(var);
  // may introduce relative error, can try rsqrt28 for higher acc.
  auto zmm_rsqrt14_var = _mm512_rsqrt14_ps(zmm_var);
  norm_func(map_size, dt_bytewidth_, channels_per_group_, gamma_data, beta_data, cur_group_src, cur_group_dst,
            &zmm_rsqrt14_var, &zmm_mean);
}

// GroupNorm base on AVX512 intrinsic and parallel on Group.
void GroupNormOperator::GroupNormParallelG(const void* src_data, const float* gamma_data, const float* beta_data,
                                           void* dst_data, const vector<int64_t>& src_shape) {
  auto map_size = std::accumulate(src_shape.begin() + 2, src_shape.end(), 1, std::multiplies<int>());
#pragma omp parallel for collapse(2)
  for (int64_t batch = 0; batch < src_shape[0]; batch++) {
    for (int64_t group = 0; group < group_; group++) {
      auto offset = (batch * channels_ + group * channels_per_group_) * map_size * dt_bytewidth_;
      char* cur_group_src = static_cast<char*>(const_cast<void*>(src_data)) + offset;
      char* cur_group_dst = static_cast<char*>(const_cast<void*>(dst_data)) + offset;
      float* cur_gamma = const_cast<float*>(gamma_data) + group * channels_per_group_;
      float* cur_beta = const_cast<float*>(beta_data) + group * channels_per_group_;
      NormGroup(cur_group_src, cur_gamma, cur_beta, cur_group_dst, map_size);
    }
  }
}

void GroupNormRef(const float* src_data, const float* gamma_data, const float* beta_data, float* dst_data,
                  const vector<int64_t>& src_shape, const float eps, const int64_t group, const int64_t channels,
                  const bool affine) {
  // x = (x - mean) / sqrt(var + eps) * gamma + beta
  const int64_t batch_size = src_shape[0];
  int64_t map_size = 1;
  for (int i = 2; i < src_shape.size(); ++i) {
    map_size *= src_shape[i];
  }
  const int64_t channels_per_group = channels / group;

#pragma omp parallel for
  for (int64_t n = 0; n < batch_size; n++) {
    const float* src_single_data = src_data + n * channels * map_size;
    float* dst_single_data = dst_data + n * channels * map_size;
#pragma omp simd
    for (int64_t g = 0; g < group; g++) {
      const float* src_group_data = src_single_data + g * channels_per_group * map_size;
      float* dst_group_data = dst_single_data + g * channels_per_group * map_size;
      // mean and var
      float sum = 0.f;
      for (int64_t q = 0; q < channels_per_group; q++) {
        const float* ptr = src_group_data + q * map_size;
        for (int64_t i = 0; i < map_size; i++) {
          sum += ptr[i];
        }
      }
      float mean = sum / (channels_per_group * map_size);

      float sqsum = 0.f;
      for (int64_t q = 0; q < channels_per_group; q++) {
        const float* ptr = src_group_data + q * map_size;
        for (int64_t i = 0; i < map_size; i++) {
          float tmp = ptr[i] - mean;
          sqsum += tmp * tmp;
        }
      }
      float var = sqsum / (channels_per_group * map_size);

      for (int64_t q = 0; q < channels_per_group; q++) {
        float a;
        float b;
        if (affine) {
          float gamma = gamma_data[g * channels_per_group + q];
          float beta = beta_data[g * channels_per_group + q];

          a = static_cast<float>(gamma / sqrt(var + eps));
          b = -mean * a + beta;
        } else {
          a = static_cast<float>(1.f / (sqrt(var + eps)));
          b = -mean * a;
        }

        const float* ptr = src_group_data + q * map_size;
        float* dst_ptr = dst_group_data + q * map_size;
        for (int64_t i = 0; i < map_size; i++) {
          dst_ptr[i] = ptr[i] * a + b;
        }
      }
    }
  }
}

GroupNormOperator::GroupNormOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("epsilon");
  if (iter != attrs_map.end()) {
    epsilon_ = StringToNum<float>(attrs_map["epsilon"]);
  }
  iter = attrs_map.find("group");
  if (iter != attrs_map.end()) {
    group_ = StringToNum<int64_t>(attrs_map["group"]);
  }
  iter = attrs_map.find("channels");
  if (iter != attrs_map.end()) {
    channels_ = StringToNum<int64_t>(attrs_map["channels"]);
  }
  channels_per_group_ = channels_ / group_;
}

void GroupNormOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  Tensor* src = input[0];
  assert(src->dtype() == "fp32" || src->dtype() == "bf16");
  dt_bytewidth_ = src->dtype() == "fp32" ? 4 : 2;
  if (dt_bytewidth_ == 4) {
    sum_func = fp32_sum;
    norm_func = fp32_norm;
  } else {
    sum_func = bf16_sum;
    norm_func = bf16_norm;
  }
  output[0]->set_dtype(src->dtype());
  Tensor* gamma = input[1];
  Tensor* beta = input[2];
  assert(gamma->shape()[0] == channels_);
  assert(beta->shape()[0] == channels_);
  const float* gamma_data = static_cast<const float*>(gamma->data());
  const float* beta_data = static_cast<const float*>(beta->data());
  for (int64_t i = 0; i < channels_; ++i) {
    if (gamma_data[i] != 1.f || beta_data[i] != 0.f) {
      affine_ = true;
      break;
    }
  }
}

void GroupNormOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  const vector<int64_t> src_shape = input[0]->shape();
  assert(src_shape.size() > 1);
  output[0]->set_shape(src_shape);
}

void GroupNormOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  Tensor* src = input[0];
  const vector<int64_t>& src_shape = src->shape();
  const float* src_data = static_cast<const float*>(src->data());
  Tensor* gamma = input[1];
  const float* gamma_data = static_cast<const float*>(gamma->data());
  Tensor* beta = input[2];
  const float* beta_data = static_cast<const float*>(beta->data());
  Tensor* dst = output[0];
  float* dst_data = static_cast<float*>(dst->mutable_data());
  GroupNormParallelG(src_data, gamma_data, beta_data, dst_data, src_shape);
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(GroupNorm);
}  // namespace executor
