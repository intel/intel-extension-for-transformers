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

#pragma once

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
  for (int i = 0; i < compress_wei->_K; i += KTILE) {
    for (int j = 0; j < compress_wei->_N; j += NTILE) {
      gpu_dequant_s4fullrange_f32_KxN<KTILE, NTILE / 2, LOCAL_K, LOCAL_N>(
          q, src_buf, dst_buf, scale_buf, compress_wei->_K, compress_wei->_N,
          compress_wei->_blksize, i, j, transpose);
    }
  }
  return;
}

void s8_quant_row_blk(const float *srcptr, int8_t *dstptr, int row, int col,
                      int ld_src, int ld_dst, fp16 *scales, int blocksize, bool trans) {
  int raw_blocksize = blocksize;
  for (int i = 0; i < col; i++) {
    int align_row_loop = row / blocksize * blocksize;
    int j = 0;

    auto s4_fullrange_calc_store_scale_and_quantv_sym = [&](int blocksize, bool trans) {
      float amax = 0.f, max = 0.f;
      for (size_t ij = 0; ij < blocksize; ij++) {
        int idx = trans ? i * ld_src + ij + j : (j + ij) * ld_src + i;
        auto v = srcptr[idx];
        if (amax < std::abs(v)) {
          amax = std::abs(v);
          max = v;
        }
      }
      fp16 scale = max / -8.f;
      fp16 rscale = scale != 0.f ? 1.f / scale : 0.f;
      scales[j / raw_blocksize * ld_dst + i] = scale;
      for (size_t ij = 0; ij < blocksize; ij++) {
        int idx = trans ? i * ld_src + ij + j : (j + ij) * ld_src + i;
        auto quant_v = srcptr[idx] * rscale;
        int8_t x = MIN(15, (int8_t)(quant_v + 8.5f));
        dstptr[(j + ij) * ld_dst + i] = x << 4;
      }
    };
    for (; j < align_row_loop; j += blocksize)
      s4_fullrange_calc_store_scale_and_quantv_sym(blocksize, trans);
    if (j < row)
      s4_fullrange_calc_store_scale_and_quantv_sym(row - align_row_loop, trans);
  }
}

void compress_s8_s4(const int8_t *srcptr, gblas::int4x2 *dstptr, int row,
                    int col, int ld_src, int ld_dst) {
  for (int j = 0; j < row; j++) {
    for (int ii = 0; ii < col; ii += 2) {
      gblas::int4x2 tmp;
      tmp.x = gblas::int4x2::convert(srcptr[j * ld_src + ii + 0]);
      tmp.y = gblas::int4x2::convert(srcptr[j * ld_src + ii + 1]);
      dstptr[j * ld_dst / 2 + ii / 2] = tmp;
    }
  }
}

