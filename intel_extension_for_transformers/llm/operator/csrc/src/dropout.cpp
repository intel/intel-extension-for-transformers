//  Copyright (c) 2023 Intel Corporation
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
#include <ATen/core/TensorBody.h>
#include <immintrin.h>

#include <cassert>

#include "../include/dropout.hpp"
#include "bestla/bestla_utils.h"
#include "bestla/kernel_avx2.h"
#include "bestla/kernel_avx512f.h"

#pragma GCC push_options
#pragma GCC target("avx512f", "avx512bw", "avx512vl")
#if CompileBF16()
#pragma GCC target("avx512bf16")
#endif

template <bool BF16>
static inline void write_rand(char* data, int thread_idx, int64_t elt_num, int dt_size, double p, char* mask_ptr) {
  int i = 0;
  auto zmm_scale = _mm512_set1_ps(1.f / (1.f - p));
  auto zmm_p = _mm512_set1_ps(float(p));
  int align_elt_num = elt_num / 16 * 16;
  for (; i < align_elt_num; i += 16) {
    auto randv = rand_generator.gen_randfp(thread_idx);
    auto mul_scale = _mm512_set1_ps(0.f);
    auto zero_mask = _mm512_cmplt_ps_mask(zmm_p, randv);
    mul_scale = _mm512_mask_mov_ps(mul_scale, zero_mask, zmm_scale);
    if constexpr (!BF16) {
      auto ans = _mm512_loadu_ps(data + i * dt_size);
      ans = _mm512_mul_ps(ans, mul_scale);
      _mm512_storeu_ps(data + i * dt_size, ans);
      _mm512_storeu_ps(mask_ptr + i * dt_size, mul_scale);
    } else {
      auto ans = _mm512_castsi512_ps(
          _mm512_bslli_epi128(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(data + i * dt_size))), 2));
      ans = _mm512_mul_ps(ans, mul_scale);
      __m256i bf16_ans, bf16_mul_scale;
#if CompileBF16()
      bf16_ans = (__m256i)_mm512_cvtneps_pbh(ans);
      bf16_mul_scale = (__m256i)_mm512_cvtneps_pbh(mul_scale);
#else
      bf16_ans = bestla::kernel::avx512f::zmm_cvt_fp32_bf16(ans);
      bf16_mul_scale = bestla::kernel::avx512f::zmm_cvt_fp32_bf16(mul_scale);
#endif
      _mm256_storeu_si256((__m256i*)(data + i * dt_size), bf16_ans);
      _mm256_storeu_si256((__m256i*)(mask_ptr + i * dt_size), bf16_mul_scale);
    }
  }
  if (i < elt_num) {
    auto randv = rand_generator.gen_randfp(thread_idx);
    auto ls_mask = _cvtu32_mask16(0xffff >> (16 - elt_num + i));
    auto mul_scale = _mm512_set1_ps(0.f);
    auto zero_mask = _mm512_cmplt_ps_mask(zmm_p, randv);
    mul_scale = _mm512_mask_mov_ps(mul_scale, zero_mask, zmm_scale);
    if constexpr (!BF16) {
      __m512 ans;
      ans = _mm512_mask_loadu_ps(ans, ls_mask, data + i * dt_size);
      ans = _mm512_mul_ps(ans, mul_scale);
      _mm512_mask_storeu_ps(data + i * dt_size, ls_mask, ans);
      _mm512_mask_storeu_ps(mask_ptr + i * dt_size, ls_mask, mul_scale);
    } else {
      __m256i ymm_tmp;
      auto ans = _mm512_castsi512_ps(
          _mm512_bslli_epi128(_mm512_cvtepu16_epi32(_mm256_mask_loadu_epi16(ymm_tmp, ls_mask, data + i * dt_size)), 2));
      ans = _mm512_mul_ps(ans, mul_scale);
      __m256i bf16_ans, bf16_mul_scale;
#if CompileBF16()
      bf16_ans = (__m256i)_mm512_cvtneps_pbh(ans);
      bf16_mul_scale = (__m256i)_mm512_cvtneps_pbh(mul_scale);
#else
      bf16_ans = bestla::kernel::avx512f::zmm_cvt_fp32_bf16(ans);
      bf16_mul_scale = bestla::kernel::avx512f::zmm_cvt_fp32_bf16(mul_scale);
#endif
      _mm256_mask_storeu_epi16(data + i * dt_size, ls_mask, bf16_ans);
      _mm256_mask_storeu_epi16(mask_ptr + i * dt_size, ls_mask, bf16_mul_scale);
    }
  }
}

template <bool BF16>
static inline void mul(char* grad, int thread_idx, int64_t elt_num, int dt_size, char* mask_ptr) {
  int i = 0;
  int align_elt_num = elt_num / 16 * 16;
  for (; i < align_elt_num; i += 16) {
    if constexpr (!BF16) {
      auto ans = _mm512_loadu_ps(grad + i * dt_size);
      ans = _mm512_mul_ps(ans, _mm512_loadu_ps(mask_ptr + i * dt_size));
      _mm512_storeu_ps(grad + i * dt_size, ans);
    } else {
      auto ans = _mm512_castsi512_ps(
          _mm512_bslli_epi128(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(grad + i * dt_size))), 2));
      auto zmm_mask = _mm512_castsi512_ps(
          _mm512_bslli_epi128(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(mask_ptr + i * dt_size))), 2));
      ans = _mm512_mul_ps(ans, zmm_mask);
      __m256i bf16_ans;
#if CompileBF16()
      bf16_ans = (__m256i)_mm512_cvtneps_pbh(ans);
#else
      bf16_ans = bestla::kernel::avx512f::zmm_cvt_fp32_bf16(ans);
#endif
      _mm256_storeu_si256((__m256i*)(grad + i * dt_size), bf16_ans);
    }
  }
  if (i < elt_num) {
    auto ls_mask = _cvtu32_mask16(0xffff >> (16 - elt_num + i));
    if constexpr (!BF16) {
      __m512 ans, zmm_mask;
      ans = _mm512_mask_loadu_ps(ans, ls_mask, grad + i * dt_size);
      ans = _mm512_mul_ps(ans, _mm512_mask_loadu_ps(zmm_mask, ls_mask, mask_ptr + i * dt_size));
      _mm512_mask_storeu_ps(grad + i * dt_size, ls_mask, ans);
    } else {
      __m256i ymm_tmp;
      auto ans = _mm512_castsi512_ps(
          _mm512_bslli_epi128(_mm512_cvtepu16_epi32(_mm256_mask_loadu_epi16(ymm_tmp, ls_mask, grad + i * dt_size)), 2));
      auto zmm_mask = _mm512_castsi512_ps(_mm512_bslli_epi128(
          _mm512_cvtepu16_epi32(_mm256_mask_loadu_epi16(ymm_tmp, ls_mask, mask_ptr + i * dt_size)), 2));
      ans = _mm512_mul_ps(ans, zmm_mask);
      __m256i bf16_ans;
#if CompileBF16()
      bf16_ans = (__m256i)_mm512_cvtneps_pbh(ans);
#else
      bf16_ans = bestla::kernel::avx512f::zmm_cvt_fp32_bf16(ans);
#endif
      _mm256_mask_storeu_epi16(grad + i * dt_size, ls_mask, bf16_ans);
    }
  }
}
#pragma GCC pop_options

#pragma GCC push_options
#pragma GCC target("avx2")
template <bool BF16>
static inline void write_rand_avx2(char* data, int thread_idx, int64_t elt_num, int dt_size, double p, char* mask_ptr) {
  int i = 0;
  auto ymm_scale = _mm256_set1_ps(1.f / (1.f - p));
  auto ymm_p = _mm256_set1_ps(float(p));
  int align_elt_num = elt_num / 8 * 8;
  auto bf16_and_helper = _mm256_set1_epi32(0X00000001);
  auto bf16_add_helper = _mm256_set1_epi32(0x00007FFF);
  for (; i < align_elt_num; i += 8) {
    auto randv = rand_generator.gen_randfp_avx2(thread_idx);
    auto mul_scale = _mm256_set1_ps(0.f);
    auto zero_mask = _mm256_cmp_ps(ymm_p, randv, 1);
    mul_scale = _mm256_blendv_ps(mul_scale, ymm_scale, zero_mask);
    if constexpr (!BF16) {
      auto ans = _mm256_load_ps(reinterpret_cast<float*>(data + i * dt_size));
      ans = _mm256_mul_ps(ans, mul_scale);
      _mm256_store_ps(reinterpret_cast<float*>(data + i * dt_size), ans);
      _mm256_store_ps(reinterpret_cast<float*>(mask_ptr + i * dt_size), mul_scale);
    } else {
      auto bf16_v = _mm_loadu_si128(reinterpret_cast<__m128i*>(data + i * dt_size));
      auto fp32_v = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_cvtepu16_epi32(bf16_v), 2));
      fp32_v = _mm256_mul_ps(fp32_v, mul_scale);
      auto ans = bestla::kernel::avx2::cvt_fp32_to_bf16(fp32_v, &bf16_and_helper, &bf16_add_helper);
      auto bf16_scale = bestla::kernel::avx2::cvt_fp32_to_bf16(mul_scale, &bf16_and_helper, &bf16_add_helper);
      _mm_store_ps(reinterpret_cast<float*>(data + i * dt_size), _mm_castsi128_ps(ans));
      _mm_store_ps(reinterpret_cast<float*>(mask_ptr + i * dt_size), _mm_castsi128_ps(bf16_scale));
    }
  }
  if (i < elt_num) {
    auto randv = rand_generator.gen_randfp_avx2(thread_idx);
    auto mul_scale = _mm256_set1_ps(0.f);
    auto zero_mask = _mm256_cmp_ps(ymm_p, randv, 1);
    if constexpr (!BF16) {
      float* fp_data_ptr = reinterpret_cast<float*>(data);
      float* fp_mask_ptr = reinterpret_cast<float*>(mask_ptr);
      mul_scale = _mm256_blendv_ps(mul_scale, ymm_scale, zero_mask);
      float mul_scale_arr[8];
      _mm256_storeu_ps(mul_scale_arr, mul_scale);
      for (int j = 0; j < (elt_num - align_elt_num); j++) {
        fp_data_ptr[i + j] = fp_data_ptr[i + j] * mul_scale_arr[j];
        fp_mask_ptr[i + j] = mul_scale_arr[j];
      }
    } else {
      bestla::utils::bf16* bf16_data_ptr = reinterpret_cast<bestla::utils::bf16*>(data);
      bestla::utils::bf16* bf16_mask_ptr = reinterpret_cast<bestla::utils::bf16*>(mask_ptr);
      mul_scale = _mm256_blendv_ps(mul_scale, ymm_scale, zero_mask);
      float mul_scale_arr[8];
      _mm256_storeu_ps(mul_scale_arr, mul_scale);
      for (int j = 0; j < (elt_num - align_elt_num); j++) {
        bf16_data_ptr[i + j].fromfloat(bf16_data_ptr[i + j].tofloat() * mul_scale_arr[j]);
        bf16_mask_ptr[i + j].fromfloat(mul_scale_arr[j]);
      }
    }
  }
}

template <bool BF16>
static inline void mul_avx2(char* grad, int thread_idx, int64_t elt_num, int dt_size, char* mask_ptr) {
  int i = 0;
  int align_elt_num = elt_num / 8 * 8;
  auto bf16_and_helper = _mm256_set1_epi32(0X00000001);
  auto bf16_add_helper = _mm256_set1_epi32(0x00007FFF);
  for (; i < align_elt_num; i += 8) {
    if constexpr (!BF16) {
      auto ans = _mm256_load_ps(reinterpret_cast<float*>(grad + i * dt_size));
      ans = _mm256_mul_ps(ans, _mm256_load_ps(reinterpret_cast<float*>(mask_ptr + i * dt_size)));
      _mm256_store_ps(reinterpret_cast<float*>(grad + i * dt_size), ans);
    } else {
      auto bf16_grad = _mm_loadu_si128(reinterpret_cast<__m128i*>(grad + i * dt_size));
      auto bf16_mask = _mm_loadu_si128(reinterpret_cast<__m128i*>(mask_ptr + i * dt_size));
      auto fp32_grad = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_cvtepu16_epi32(bf16_grad), 2));
      auto fp32_mask = _mm256_castsi256_ps(_mm256_bslli_epi128(_mm256_cvtepu16_epi32(bf16_mask), 2));
      fp32_grad = _mm256_mul_ps(fp32_grad, fp32_mask);
      auto ans = bestla::kernel::avx2::cvt_fp32_to_bf16(fp32_grad, &bf16_and_helper, &bf16_add_helper);
      _mm_store_ps(reinterpret_cast<float*>(grad + i * dt_size), _mm_castsi128_ps(ans));
    }
  }
  if (i < elt_num) {
    if constexpr (!BF16) {
      float* fp_data_ptr = reinterpret_cast<float*>(grad);
      float* fp_mask_ptr = reinterpret_cast<float*>(mask_ptr);
      for (int j = 0; j < (elt_num - align_elt_num); j++) {
        fp_data_ptr[i + j] = fp_data_ptr[i + j] * fp_mask_ptr[i + j];
      }
    } else {
      bestla::utils::bf16* bf16_data_ptr = reinterpret_cast<bestla::utils::bf16*>(grad);
      bestla::utils::bf16* bf16_mask_ptr = reinterpret_cast<bestla::utils::bf16*>(mask_ptr);
      for (int j = 0; j < (elt_num - align_elt_num); j++) {
        bf16_data_ptr[i + j].fromfloat(bf16_data_ptr[i + j].tofloat() * bf16_mask_ptr[i + j].tofloat());
      }
    }
  }
}
#pragma GCC pop_options

torch::Tensor dropout_fwd(torch::Tensor& output, double p) {
  auto elt_num = output.numel();
  auto core_num = omp_get_max_threads();
  auto task_each_core = bestla::utils::updiv(int(elt_num / core_num), 16) * 16;
  torch::Tensor mask = torch::empty_like(output);
#pragma omp parallel
  {
    auto ker_idx = omp_get_thread_num();
    auto tasks =
        elt_num - ker_idx * task_each_core > task_each_core ? task_each_core : elt_num - ker_idx * task_each_core;
    if (output.scalar_type() == torch::kFloat32) {
      if (dispatcher_utils::check_avx512f()) {
        write_rand<false>(reinterpret_cast<char*>(output.data_ptr()) + ker_idx * task_each_core * output.element_size(),
                          ker_idx, tasks, output.element_size(), p,
                          reinterpret_cast<char*>(mask.data_ptr()) + ker_idx * task_each_core * output.element_size());
      } else {
        write_rand_avx2<false>(
            reinterpret_cast<char*>(output.data_ptr()) + ker_idx * task_each_core * output.element_size(), ker_idx,
            tasks, output.element_size(), p,
            reinterpret_cast<char*>(mask.data_ptr()) + ker_idx * task_each_core * output.element_size());
      }
    } else if (output.scalar_type() == torch::kBFloat16) {
      if (dispatcher_utils::check_avx512f()) {
        write_rand<true>(reinterpret_cast<char*>(output.data_ptr()) + ker_idx * task_each_core * output.element_size(),
                         ker_idx, tasks, output.element_size(), p,
                         reinterpret_cast<char*>(mask.data_ptr()) + ker_idx * task_each_core * output.element_size());
      } else {
        write_rand_avx2<true>(
            reinterpret_cast<char*>(output.data_ptr()) + ker_idx * task_each_core * output.element_size(), ker_idx,
            tasks, output.element_size(), p,
            reinterpret_cast<char*>(mask.data_ptr()) + ker_idx * task_each_core * output.element_size());
      }
    } else {
      TORCH_CHECK(false, "Qbits: unsupported input data type in dropout operator.");
    }
  }
  return mask;
}

void dropout_bwd(torch::Tensor& grad, torch::Tensor& mask) {
  auto elt_num = grad.numel();
  auto core_num = omp_get_max_threads();
  auto task_each_core = bestla::utils::updiv(int(elt_num / core_num), 16) * 16;
#pragma omp parallel
  {
    auto ker_idx = omp_get_thread_num();
    auto tasks =
        elt_num - ker_idx * task_each_core > task_each_core ? task_each_core : elt_num - ker_idx * task_each_core;
    if (grad.scalar_type() == torch::kFloat32) {
      if (dispatcher_utils::check_avx512f()) {
        mul<false>(reinterpret_cast<char*>(grad.data_ptr()) + ker_idx * task_each_core * grad.element_size(), ker_idx,
                   tasks, grad.element_size(),
                   reinterpret_cast<char*>(mask.data_ptr()) + ker_idx * task_each_core * grad.element_size());
      } else {
        mul_avx2<false>(reinterpret_cast<char*>(grad.data_ptr()) + ker_idx * task_each_core * grad.element_size(),
                        ker_idx, tasks, grad.element_size(),
                        reinterpret_cast<char*>(mask.data_ptr()) + ker_idx * task_each_core * grad.element_size());
      }
    } else if (grad.scalar_type() == torch::kBFloat16) {
      if (dispatcher_utils::check_avx512f()) {
        mul<true>(reinterpret_cast<char*>(grad.data_ptr()) + ker_idx * task_each_core * grad.element_size(), ker_idx,
                  tasks, grad.element_size(),
                  reinterpret_cast<char*>(mask.data_ptr()) + ker_idx * task_each_core * grad.element_size());
      } else {
        mul_avx2<true>(reinterpret_cast<char*>(grad.data_ptr()) + ker_idx * task_each_core * grad.element_size(),
                       ker_idx, tasks, grad.element_size(),
                       reinterpret_cast<char*>(mask.data_ptr()) + ker_idx * task_each_core * grad.element_size());
      }
    } else {
      TORCH_CHECK(false, "Qbits: unsupported input data type in dropout operator.");
    }
  }
}
