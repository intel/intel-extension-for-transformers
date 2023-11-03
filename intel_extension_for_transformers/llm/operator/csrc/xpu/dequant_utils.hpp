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

#include "common.hpp"
#include <ipex.h>
#include <torch/extension.h>

template <int TILE_K, int TILE_N, int LOCAL_K, int LOCAL_N>
void gpu_dequant_s4fullrange_f32_KxN(sycl::queue &q,
                                     sycl::buffer<int8_t, 2> &src,
                                     sycl::buffer<float, 2> &dst,
                                     sycl::buffer<fp16, 1> &scale, int k, int n,
                                     int blksize, int k_pos, int n_pos, bool trans) {
  q.submit([&](sycl::handler &h) {
    sycl::accessor s4_wei{src, h};
    sycl::accessor fp32_wei{dst, h};
    sycl::accessor s{scale, h};
    sycl::range global{TILE_K, TILE_N};
    sycl::range local{LOCAL_K, LOCAL_N};
    h.parallel_for(sycl::nd_range{global, local}, [=](sycl::nd_item<2> it) {
      int i = it.get_global_id(0) + k_pos;
      int s4_j = it.get_global_id(1) + n_pos / 2;
      int fp32_j = s4_j * 2;
      if (i < k && fp32_j + 1 < n) {
        int8_t s8_l = s4_wei[i][s4_j] & 0x0f;
        int8_t s8_h = (s4_wei[i][s4_j] >> 4) & 0x0f;
        if (!trans) {
          fp32_wei[i][fp32_j] = (s8_l - 8) * s[i / blksize * n + fp32_j];
          fp32_wei[i][fp32_j + 1] = (s8_h - 8) * s[i / blksize * n + fp32_j + 1];
        }
        else {
          fp32_wei[fp32_j][i] = (s8_l - 8) * s[i / blksize * n + fp32_j];
          fp32_wei[fp32_j + 1][i] = (s8_h - 8) * s[i / blksize * n + fp32_j + 1];
        }
      }
    });
  });
}

void gpu_dequant(sycl::queue &q, CompressWei4Bit *compress_wei,
                 float *dequant_weight, bool transpose,
                 const std::string &compute_type,
                 const std::string &weight_type) {
  int8_t *bit4_wei =
      reinterpret_cast<int8_t *>(compress_wei->get_4bit_wei_ptr());
  fp16 *scale = reinterpret_cast<fp16 *>(compress_wei->get_scale_ptr());
  auto row = transpose ? compress_wei->_N : compress_wei->_K;
  auto col = transpose ? compress_wei->_K : compress_wei->_N;
  sycl::buffer<float, 2> dst_buf(
      dequant_weight, sycl::range<2>(row, col));
  sycl::buffer<fp16, 1> scale_buf(
      scale, sycl::range<1>(compress_wei->_K / compress_wei->_blksize *
                            compress_wei->_N));
  sycl::buffer<int8_t, 2> src_buf(
      reinterpret_cast<int8_t *>(bit4_wei),
      sycl::range<2>(compress_wei->_K, compress_wei->_N / 2));
  constexpr int KTILE = 1024, NTILE = 1024;
  constexpr int LOCAL_K = 32, LOCAL_N = 32;
  using namespace std::chrono;
  auto m_start = high_resolution_clock::now();
  for (int i = 0; i < compress_wei->_K; i += KTILE) {
    for (int j = 0; j < compress_wei->_N; j += NTILE) {
      gpu_dequant_s4fullrange_f32_KxN<KTILE, NTILE / 2, LOCAL_K, LOCAL_N>(
          q, src_buf, dst_buf, scale_buf, compress_wei->_K, compress_wei->_N,
          compress_wei->_blksize, i, j, transpose);
    }
  }
  auto m_end = high_resolution_clock::now();
  std::cout << "GPU dequant cost"
            << duration_cast<nanoseconds>(m_end - m_start).count() / 1e6 << "ms"
            << std::endl;
  return;
}