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

torch::Tensor quantize(float *weight, int k, int n, int blksize,
                       std::string weight_type, std::string cmpt_type, bool trans) {
  CompressWei4Bit compress_wei(k, n, blksize);
  torch::Tensor ret =
      torch::zeros(compress_wei.get_serialize_size(), torch::kInt8);
  // void* ret = malloc(compress_wei.get_serialize_size());
  if (weight_type == "s4fullrange_scalef32") {
    std::vector<int8_t> s8quant_tmp(k * n);
    fp16 *scale = reinterpret_cast<fp16 *>(compress_wei.get_scale_ptr());
    if (trans)
      s8_quant_row_blk(weight, s8quant_tmp.data(), k, n, k, n, scale, blksize, trans);
    else
      s8_quant_row_blk(weight, s8quant_tmp.data(), k, n, n, n, scale, blksize, trans);
    gblas::int4x2 *wei =
        reinterpret_cast<gblas::int4x2 *>(compress_wei.get_4bit_wei_ptr());
    compress_s8_s4(s8quant_tmp.data(), wei, k, n, n, n);
    compress_wei.serialize(ret.data_ptr<int8_t>());
  } else {
    assert(0);
  }
  return ret;
}

static torch::Tensor gbits_quantize(const torch::Tensor &weight, bool transpose,
                                    int64_t block_size,
                                    const std::string &compute_type,
                                    const std::string &weight_type) {
  int n = transpose ? weight.sizes()[0] : weight.sizes()[1];
  int k = transpose ? weight.sizes()[1] : weight.sizes()[0];
  torch::Tensor output;
  output = quantize(weight.data_ptr<float>(), k, n, block_size, weight_type,
                    compute_type, transpose);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gbits_quantize, "gbits_quantize forward (XPU)");
}
