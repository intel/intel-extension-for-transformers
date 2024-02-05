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
#include "../dispatcher/include/bestla_weightonly_dispatcher.hpp"
#include "bestla/bestla_utils.h"
#include <ATen/core/TensorBody.h>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <vector>

class RandBuffer {
 public:
  RandBuffer() {
    std::srand((int)std::time(0));
    auto thread_num = bestla::device::CpuDevice::getInstance()->getThreads();
    load_buffer = bestla::utils::amalloc<uint32_t>(thread_num * 16);
    iws = bestla::utils::amalloc<int>(thread_num * 16);
    for (int i = 0; i < thread_num; i++) initMWC(rand(), i);
  }

  ~RandBuffer() {
    if (iws != NULL) bestla::utils::afree(iws);
    if (load_buffer != NULL) bestla::utils::afree(load_buffer);
  }

#pragma GCC push_options
#pragma GCC target("avx512f")
  __m512 gen_randfp(int thread_idx) {
    auto zmm_one = _mm512_set1_epi32(0x3F800000);
    auto rand = next1(thread_idx);
    auto r1 = _mm512_sub_epi32(
        zmm_one, _mm512_and_epi32(_mm512_srl_epi32(rand, _mm_cvtsi32_si128((int32_t)8)), _mm512_set1_epi32(1)));
    auto r2 = _mm512_or_epi32(_mm512_srl_epi32(rand, _mm_cvtsi32_si128((int32_t)9)), zmm_one);
    auto ans = _mm512_sub_ps(_mm512_castsi512_ps(r2), _mm512_castsi512_ps(r1));
    return ans;
  }
#pragma GCC pop_options

#pragma GCC push_options
#pragma GCC target("avx2")
  __m256 gen_randfp_avx2(int thread_idx) {
    auto ymm_one = _mm256_set1_epi32(0x3F800000);
    auto rand = next1_avx2(thread_idx);
    auto r1 = _mm256_sub_epi32(
        ymm_one, _mm256_and_si256(_mm256_srl_epi32(rand, _mm_cvtsi32_si128((int32_t)8)), _mm256_set1_epi32(1)));
    auto r2 = _mm256_or_si256(_mm256_srl_epi32(rand, _mm_cvtsi32_si128((int32_t)9)), ymm_one);
    auto ans = _mm256_sub_ps(_mm256_castsi256_ps(r2), _mm256_castsi256_ps(r1));
    return ans;
  }
#pragma GCC pop_options

 private:
  enum constants : uint32_t {
    // Factors for MWC generators
    mwcfac0 = 4294963023u,  // Factor for each MWC generator
    mwcfac1 = 3947008974u,
    mwcfac2 = 4162943475u,
    mwcfac3 = 2654432763u,
    mwcfac4 = 3874257210u,
    mwcfac5 = 2936881968u,
    mwcfac6 = 4294957665u,
    mwcfac7 = 2811536238u,
    shw1 = 30,  // Shift counts for MWC tempering
    shw2 = 35,
    shw3 = 13
  };

  uint32_t* load_buffer;
  int* iws;

  const uint32_t MWCFactors[16] = {  // Factor for MWC
      mwcfac0, 0, mwcfac1, 0, mwcfac2, 0, mwcfac3, 0, mwcfac4, 0, mwcfac5, 0, mwcfac6, 0, mwcfac7, 0};

#pragma GCC push_options
#pragma GCC target("avx512f")
  __m512i next1(int thread_idx) {  // Get 512 bits from MWC
    uint32_t* buffer = load_buffer + 16 * thread_idx;
    // Factors for multiply-with-carry
    __m512i x, f, y, y_cp;
    x = _mm512_loadu_si512(buffer);      // previous x and carry
    f = _mm512_loadu_si512(MWCFactors);  // factors
    y = _mm512_mul_epu32(x, f);          // 32*32->64 bit unsigned multiply
    // add old carry
    x = _mm512_srl_epi64(x, _mm_cvtsi32_si128((int32_t)32u));
    y = _mm512_add_epi64(x, y);
    _mm512_storeu_si512(buffer, y);  // new x and carry
    // The uppermost bits of the carry are not sufficiently random. Randomize
    // some more for output
    y_cp = _mm512_sll_epi64(y, _mm_cvtsi32_si128(shw1));
    y = _mm512_xor_epi32(y, y_cp);
    y_cp = _mm512_srl_epi64(y, _mm_cvtsi32_si128((int32_t)shw2));
    y = _mm512_xor_epi32(y, y_cp);
    y_cp = _mm512_sll_epi64(y, _mm_cvtsi32_si128(shw3));
    y = _mm512_xor_epi32(y, y_cp);
    return y;
  }
#pragma GCC pop_options

#pragma GCC push_options
#pragma GCC target("avx2")
  __m256i next1_avx2(int thread_idx) {  // Get 256 bits from MWC
    uint32_t* buffer = load_buffer + 8 * thread_idx;
    auto iw = iws[thread_idx * 16];
    __m256i x, f, y, y_cp;
    x = _mm256_loadu_si256((__m256i*)(buffer + iw));
    f = _mm256_loadu_si256((__m256i*)(MWCFactors + iw));
    y = _mm256_mul_epu32(x, f);
    x = _mm256_srl_epi64(x, _mm_cvtsi32_si128((int32_t)32u));
    y = _mm256_add_epi64(x, y);
    _mm256_storeu_si256((__m256i*)(buffer + iw), y);
    y_cp = _mm256_sll_epi64(y, _mm_cvtsi32_si128(shw1));
    y = _mm256_xor_si256(y, y_cp);
    y_cp = _mm256_srl_epi64(y, _mm_cvtsi32_si128((int32_t)shw2));
    y = _mm256_xor_si256(y, y_cp);
    y_cp = _mm256_sll_epi64(y, _mm_cvtsi32_si128(shw3));
    y = _mm256_xor_si256(y, y_cp);
    iws[thread_idx * 16] = (iw + 8) & 15;
    return y;
  }

  void initMWC(int seed, int thread_idx) {
    uint32_t* buffer = load_buffer + 16 * thread_idx;
    iws[thread_idx * 16] = 0;
    int i;
    // Fill buffer with function of seed
    uint32_t tmp = seed;

    // Seeds (0,0) and (-1,factor-1) will degenerate into constant output.
    // This seeding method should avoid these forbidden seeds:

    for (i = 0; i < 16; i++) {
      tmp = buffer[i] = 1566083941u * (tmp ^ (tmp >> 27)) + i;
    }
    // Randomize 4 rounds
    for (i = 0; i < 8; i++) next1_avx2(thread_idx);
    iws[thread_idx * 16] = 0;
  }
#pragma GCC pop_options
};
static RandBuffer rand_generator;
void dropout_bwd(torch::Tensor& grad, torch::Tensor& mask);
torch::Tensor dropout_fwd(torch::Tensor& output, double p);
