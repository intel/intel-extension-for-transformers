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

#include "rmsnorm.hpp"

#include "common.hpp"

namespace executor {

#define FP32_POWXSUM(zmm1, zmm2)    \
  zmm2 = _mm512_mul_ps(zmm2, zmm2); \
  zmm1 = _mm512_add_ps(zmm1, zmm2);

void fp32_norm(char* src, const float* gamma, char* dst, int norm_dim, __m512* scale) {
  for (int i = 0; i < norm_dim; i += 16) {
    auto zmm_src = _mm512_loadu_ps(src + i * 4);
    auto zmm_gamma = _mm512_loadu_ps(gamma + i);
    auto zmm_fin_scale = _mm512_mul_ps(zmm_gamma, *scale);
    _mm512_storeu_ps(dst + i * 4, _mm512_mul_ps(zmm_src, zmm_fin_scale));
  }
}

__m512 bf16_load(float* addr) {
  auto bf16_data = _mm256_loadu_ps(addr);
#if __AVX512BF16__ && __GNUC__ > 11
  return _mm512_cvtpbh_ps((__m256bh)bf16_data);
#else
  auto y = _mm512_cvtepu16_epi32((__m256i)bf16_data);
  return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
#endif
}

void bf16_norm(char* src, const float* gamma, char* dst, int norm_dim, __m512* scale) {
  auto fp32_cvt_bf16 = [](__m512& zmm_dst) {
#if __AVX512BF16__ && __GNUC__ > 11
    return _mm512_cvtneps_pbh(zmm_dst);
#else
    auto y = _mm512_bsrli_epi128(_mm512_castps_si512(zmm_dst), 2);
    return _mm512_cvtepi32_epi16(y);
#endif
  };

  for (int i = 0; i < norm_dim; i += 16) {
    auto zmm_src = bf16_load(static_cast<float*>(static_cast<void*>(src + i * 2)));
    auto zmm_gamma = _mm512_loadu_ps(gamma + i);
    auto zmm_fin_scale = _mm512_mul_ps(zmm_gamma, *scale);
    auto zmm_dst = _mm512_mul_ps(zmm_src, zmm_fin_scale);
    auto bf16_dst = fp32_cvt_bf16(zmm_dst);
    _mm256_storeu_ps(static_cast<float*>(static_cast<void*>(dst + i * 2)), (__m256)bf16_dst);
  }
}

template <int dt_bytewidth>
void RmsNormOperator::RmsNormParallelB(const void* src_data, const float* gamma_data, void* dst_data) {
#pragma omp parallel for
  for (int batch = 0; batch < batchs_; batch++) {
    auto offset = batch * norm_dim_ * dt_bytewidth_;
    auto src = static_cast<const char*>(src_data) + offset;
    auto dst = static_cast<char*>(dst_data) + offset;

    auto zmm_powx_sum = _mm512_setzero_ps();

    for (int j = 0; j < norm_dim_; j += 16) {
#if dt_bytewidth == 2
      auto zmm_src = bf16_load(static_cast<float*>(static_cast<void*>(const_cast<char*>(src) + j * 2)));
#else
      auto zmm_src = _mm512_loadu_ps(src + j * dt_bytewidth_);
#endif
      FP32_POWXSUM(zmm_powx_sum, zmm_src);
    }

    auto powx_mean = (_mm512_reduce_add_ps(zmm_powx_sum) + epsilon_) / norm_dim_;

    auto zmm_scale = _mm512_set1_ps(powx_mean);
    zmm_scale = _mm512_rsqrt14_ps(zmm_scale);
    parallelB_norm_callback_(const_cast<char*>(src), gamma_data, dst, norm_dim_, &zmm_scale);
  }
}

RmsNormOperator::RmsNormOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  epsilon_ = StringToNum<float>(attrs_map["epsilon"]);
}

void RmsNormOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  Tensor* src = input[0];
  assert(src->dtype() == "fp32" || src->dtype() == "bf16");
  dt_bytewidth_ = src->dtype() == "fp32" ? 4 : 2;
  if (dt_bytewidth_ == 4) {
    parallelB_norm_callback_ = fp32_norm;
  } else {
    parallelB_norm_callback_ = bf16_norm;
  }
  output[0]->set_dtype(src->dtype());
}

void RmsNormOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  const vector<int64_t> src_shape = input[0]->shape();
  assert(src_shape.size() > 1);
  norm_dim_ = *(src_shape.end() - 1);
  assert(norm_dim_ % (64 / dt_bytewidth_) == 0);  // can't process unaligned data currently.
  batchs_ = std::accumulate(src_shape.begin(), src_shape.end() - 1, 1, std::multiplies<int64_t>());
  output[0]->set_shape(src_shape);
}

void RmsNormOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  Tensor* src = input[0];
  const void* src_data = src->data();
  Tensor* gamma = input[1];
  const float* gamma_data = static_cast<const float*>(gamma->data());
  Tensor* dst = output[0];
  void* dst_data = dst->mutable_data();

  if (dt_bytewidth_ == 4)
    RmsNormParallelB<4>(src_data, gamma_data, dst_data);
  else
    RmsNormParallelB<2>(src_data, gamma_data, dst_data);

  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(RmsNorm);
}  // namespace executor
