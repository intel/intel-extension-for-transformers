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
#include "jit_blas_utils.h"

namespace jblas {
namespace kernel {
namespace ref {
static inline JBLAS_CODE padding_interleave(void* srcptr, void* dstptr, int row, int col, int rowpad, int colpad,
                                            int srcstride, int dststride, int NTile, int SrcBytes, int RowPack) {
  auto sptr = (char*)srcptr;
  auto dptr = (char*)dstptr;
  int DstBytes = RowPack * SrcBytes;
  std::vector<char> Zeros(SrcBytes, 0);
  for (int irow = 0; irow < rowpad; irow += RowPack) {
    for (int icol = 0; icol < colpad; icol += NTile) {
      for (int iin = 0; iin < NTile; iin++) {
        for (int ipa = 0; ipa < RowPack; ipa++) {
          bool valid_row = (irow + ipa) < row;
          bool valid_col = (icol + iin) < col;
          auto tmpptr =
              valid_row && valid_col ? (sptr + (irow + ipa) * srcstride + (icol + iin) * SrcBytes) : Zeros.data();
          for (int iele = 0; iele < SrcBytes; iele++) {
            (dptr + irow * NTile * SrcBytes + icol * dststride + iin * DstBytes + ipa * SrcBytes)[iele] = tmpptr[iele];
          }
        }
      }
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE dequan_s8_f32(int8_t* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst,
                                       float* scales) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      dstptr[i * ld_dst + j] = float(srcptr[i * ld_src + j]) * scales[j];
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE dequan_s8_bf16(int8_t* srcptr, uint16_t* dstptr, int row, int col, int ld_src, int ld_dst,
                                        float* scales) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      dstptr[i * ld_dst + j] =
          jblas::utils::cast<float, jblas::utils::bf16>(float(srcptr[i * ld_src + j]) * scales[j]).x;
    }
  }
  return JblasSuccess;
}

template <typename _T>
static inline JBLAS_CODE transpose2d(const _T* srcptr, _T* dstptr, int row, int col, int ld_src, int ld_dst) {
  for (int i = 0; i < col; i++) {
    for (size_t j = 0; j < row; j++) {
      dstptr[j + i * ld_dst] = srcptr[j * ld_src + i];
    }
  }
  return JblasSuccess;
}

template <int NTile>
static inline JBLAS_CODE compress_s8_s4(int8_t* srcptr, jblas::utils::int4x2* dstptr, int row, int col, int ld_src,
                                        int ld_dst) {
  for (int i = 0; i < col; i += NTile) {
    for (int j = 0; j < row; j++) {
      for (int ii = 0; ii < NTile; ii += 2) {
        jblas::utils::int4x2 tmp;
        tmp.x = jblas::utils::int4x2::convert(srcptr[i * ld_src + j * NTile + ii + 0]);
        tmp.y = jblas::utils::int4x2::convert(srcptr[i * ld_src + j * NTile + ii + 1]);
        dstptr[i * ld_dst + j * NTile / 2 + ii / 2] = tmp;
      }
    }
  }
  return JblasSuccess;
}

template <int NTile>
static inline JBLAS_CODE decompress_s4_f32(jblas::utils::int4x2* srcptr, float* dstptr, int row, int col, int ld_src,
                                           int ld_dst, float* scales) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src + j / 2];
      auto noffset = i * NTile + j % NTile;
      dstptr[i * ld_dst + j + 0] = float((int8_t)tmp.x << 4) * scales[noffset + 0];
      dstptr[i * ld_dst + j + 1] = float((int8_t)tmp.y << 4) * scales[noffset + 1];
    }
  }
  return JblasSuccess;
}

inline JBLAS_CODE decompress_kblock_s4_f32(utils::int4x2* srcptr, float* dstptr, int row, int col, int ld_src,
                                           int ld_dst, float* scales, int k_offset, int kblock, int NPad) {
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src + j / 2];
      dstptr[i * ld_dst + j + 0] = float((int8_t)tmp.x << 4) * sptr[j + 0];
      dstptr[i * ld_dst + j + 1] = float((int8_t)tmp.y << 4) * sptr[j + 1];
    }
  }
  return JblasSuccess;
}

inline JBLAS_CODE decompress_s4_s8(utils::int4x2* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src + j / 2];
      dstptr[i * ld_dst + j + 0] = (int8_t)tmp.x << 4;
      dstptr[i * ld_dst + j + 1] = (int8_t)tmp.y << 4;
    }
  }
  return JblasSuccess;
}

inline JBLAS_CODE decompress_kblock_s8_f32(int8_t* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst,
                                           float* scales, int k_offset, int kblock, int NPad) {
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 1) {
      auto tmp = srcptr[i * ld_src + j];
      dstptr[i * ld_dst + j] = float(tmp) * sptr[j + 0];
    }
  }
  return JblasSuccess;
}

inline JBLAS_CODE decompress_kblock_s4_f32(utils::int4x2* srcptr, float* dstptr, int row, int col, int ld_src,
                                           int ld_dst, utils::bf16* scales, int k_offset, int kblock, int NPad) {
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src + j / 2];
      dstptr[i * ld_dst + j + 0] = float((int8_t)tmp.x << 4) * sptr[j + 0].tofloat();
      dstptr[i * ld_dst + j + 1] = float((int8_t)tmp.y << 4) * sptr[j + 1].tofloat();
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE memcpy2d(void* srcptr, void* dstptr, int row, int col, int srcstride, int dststride) {
  auto bsrcptr = (char*)srcptr;
  auto bdstptr = (char*)dstptr;
  for (int i = 0; i < row; i++) {
    std::memcpy(bdstptr + i * dststride, bsrcptr + i * srcstride, col);
  }
  return JblasSuccess;
}

inline JBLAS_CODE quantize_f32_s8_rowblock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src,
                                           int ld_dst, float* scales, int blocksize) {
  for (int i = 0; i < col; i++) {
    for (size_t j = 0; j < row; j += blocksize) {
      float maxval = std::numeric_limits<float>::min();
      for (size_t ij = 0; ij < blocksize; ij++) {
        maxval = std::max(maxval, std::abs(srcptr[(j + ij) * ld_src + i]));
      }
      float scale = maxval / 127;
      float rscale = 1.f / scale;
      scales[j / blocksize * ld_dst + i] = scale;
      for (size_t ij = 0; ij < blocksize; ij++) {
        dstptr[(j + ij) * ld_dst + i] = utils::cast<float, int8_t>(srcptr[(j + ij) * ld_src + i] * rscale);
      }
    }
  }
  return JblasSuccess;
}

inline JBLAS_CODE quantize_f32_u8_colblock(int row, int col, const float* srcptr, int ld_src, uint8_t* dstptr,
                                           int ld_dst, float* scales, int ld_scale, uint8_t* zps, int blocksize) {
  for (int i = 0; i < row; i++) {
    for (size_t j = 0; j < col; j += blocksize) {
      float maxval = std::numeric_limits<float>::min();
      float minval = 0.f;
      for (size_t ij = 0; ij < blocksize; ij++) {
        maxval = std::max(srcptr[(j + ij) + i * ld_src], maxval);
        minval = std::min(srcptr[(j + ij) + i * ld_src], minval);
      }
      float scale = (maxval - minval) / 255;
      uint8_t zp = utils::cast<float, uint8_t>((0 - minval) / scale);
      float rscale = 1.f / scale;
      scales[j / blocksize + i * ld_scale] = scale;
      zps[j / blocksize + i * ld_scale] = zp;
      for (size_t ij = 0; ij < blocksize; ij++) {
        dstptr[(j + ij) + i * ld_dst] = utils::cast<float, uint8_t>(srcptr[(j + ij) + i * ld_src] * rscale + zp);
      }
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE alphabeta_f32_f32(const float alpha, const float* srcptr, const int srcstep, const float beta,
                                           const float* src1ptr, const int src1step, float* dstptr, const int dststep,
                                           const int M, const int N) {
  if (beta != 0.f) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        dstptr[i * dststep + j] = alpha * srcptr[i * srcstep + j] + beta * src1ptr[i * src1step + j];
      }
    }
    return JblasSuccess;
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      dstptr[i * dststep + j] = alpha * srcptr[i * srcstep + j];
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE quanout_s32_u32(const float alpha, const int32_t* srcptr, const int srcstep, uint8_t* dstptr,
                                         const int dststep, const int M, const int N, float scaleSrc, float scaleDst,
                                         int zpDst) {
  float factor = alpha * scaleSrc / scaleDst;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float fsrc = float(srcptr[i * srcstep + j]) * factor;
      dstptr[i * dststep + j] = utils::cast<float, uint8_t>(fsrc + float(zpDst));
    }
  }
  return JblasSuccess;
}

inline JBLAS_CODE minmax_f32_kblock(const float* srcptr, int row, int col, int ld_src, float* minmaxptr, int ld_minmax,
                                    int fsize_minmax, int blocksize) {
  for (int i = 0; i < row; i++) {
    if (col >= blocksize) {
      for (int icol = 0; icol < col; icol += blocksize) {
        float maxval = std::numeric_limits<float>::min();
        float minval = std::numeric_limits<float>::max();
        for (int ii = 0; ii < blocksize; ii++) {
          maxval = std::max(srcptr[i * ld_src + icol + ii], maxval);
          minval = std::min(srcptr[i * ld_src + icol + ii], minval);
        }
        auto colptr = &minmaxptr[i * ld_minmax + icol / blocksize * fsize_minmax];
        colptr[0] = minval;
        colptr[1] = maxval;
      }
    } else {
      float maxval = std::numeric_limits<float>::min();
      float minval = std::numeric_limits<float>::max();
      for (int icol = 0; icol < col; icol++) {
        maxval = std::max(srcptr[i * ld_src + icol], maxval);
        minval = std::min(srcptr[i * ld_src + icol], minval);
      }
      minmaxptr[i * ld_minmax + 0] = minval;
      minmaxptr[i * ld_minmax + 1] = maxval;
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE accumulate_dequantize_s32_f32(const int32_t* srcptr, float* dstptr, float alpha, float beta,
                                                       int row, int col, int ld_src, int ld_dst, float* ascales,
                                                       int ldas, float* wscales) {
  for (int irow = 0; irow < row; irow++) {
    for (int icol = 0; icol < col; icol++) {
      float scale = ascales[irow * ldas] * wscales[icol] * alpha;
      dstptr[irow * ld_dst + icol] = scale * srcptr[irow * ld_src + icol] + beta * dstptr[irow * ld_dst + icol];
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE broadcast_u8(int num, const uint8_t& srcval, uint8_t* dstptr) {
  int i = 0;
  for (; i < num; i++) {
    dstptr[i] = srcval;
  }
  return JblasSuccess;
}
}  // namespace ref
}  // namespace kernel
}  // namespace jblas