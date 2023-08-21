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
template <typename T_SRC, typename T_DST = T_SRC>
static inline JBLAS_CODE padding_interleave(const T_SRC* src_ptr, T_DST* dst_ptr, int row, int col, int rowpad,
                                            int colpad, int src_step, int dst_step, int NTile, int RowPack) {
  const T_DST dst_0(0);
  static_assert(sizeof(T_SRC) == sizeof(T_DST), "SRC & DST size should be the same");
  for (int i = 0; i < rowpad; i += RowPack) {
    for (int j = 0; j < colpad; j += NTile) {
      for (int jj = 0; jj < NTile; jj++) {
        for (int ii = 0; ii < RowPack; ii++) {
          dst_ptr[i * NTile + j * dst_step + jj * RowPack + ii] =
              (i + ii) < row && (j + jj) < col  //
                  ? static_cast<T_DST>(src_ptr[(i + ii) * src_step + (j + jj)])
                  : dst_0;
        }
      }
    }
  }
  return JblasSuccess;
}

// M x N ===> M/MTile x N/colPack x MTile x colPack (leading dim stride = MTile * dst_stride)
static inline JBLAS_CODE padding_trans_interleave(void* src_ptr, void* dst_ptr, int row, int col, int rowpad,
                                                  int colpad, int src_stride, int dst_stride, int MTile, int SrcBytes,
                                                  int ColPack) {
  // Note: rows/cols and i/j are in terms of src
  const auto src = reinterpret_cast<char*>(src_ptr);
  const auto dst = reinterpret_cast<char*>(dst_ptr);
  int DstBytes = ColPack * SrcBytes;
  std::vector<char> Zeros(SrcBytes, 0);

  for (int i = 0; i < rowpad; i += MTile) {
    for (int j = 0; j < colpad; j += ColPack) {
      for (int ii = 0; ii < MTile; ii++) {
        for (int jj = 0; jj < ColPack; jj++) {
          const auto packPtr = (i + ii) < row && (j + jj) < col  //
                                   ? src + (i + ii) * src_stride + (j + jj) * SrcBytes
                                   : Zeros.data();
          for (int iele = 0; iele < SrcBytes; iele++) {
            dst[i * dst_stride + j * MTile * SrcBytes + ii * DstBytes + jj * SrcBytes + iele] = packPtr[iele];
          }
        }
      }
    }
  }
  return JblasSuccess;
}
template <typename T_SRC, typename T_DST = T_SRC>
static inline JBLAS_CODE padding_trans_interleave(const T_SRC* src, T_DST* dst, int row, int col, int rowpad,
                                                  int colpad, int src_step, int dst_step, int MTile, int ColPack) {
  // Note: rows/cols and i/j are in terms of src
  static_assert(sizeof(T_SRC) == sizeof(T_DST), "SRC & DST size should be the same");
  const T_DST dst_0(0);
  for (int i = 0; i < rowpad; i += MTile) {
    for (int j = 0; j < colpad; j += ColPack) {
      for (int ii = 0; ii < MTile; ii++) {
        for (int jj = 0; jj < ColPack; jj++) {
          dst[i * dst_step + j * MTile + ii * ColPack + jj] =
              (i + ii) < row && (j + jj) < col  //
                  ? static_cast<T_DST>(src[(i + ii) * src_step + (j + jj)])
                  : dst_0;
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
        dstptr[i * ld_dst / 2 + j * NTile / 2 + ii / 2] = tmp;
      }
    }
  }
  return JblasSuccess;
}

template <int NTile>
static inline JBLAS_CODE compress_fp4(int8_t* srcptr, jblas::utils::fp4x2* dstptr, int row, int col, int ld_src,
                                      int ld_dst) {
  for (int i = 0; i < col; i += NTile) {
    for (int j = 0; j < row; j++) {
      for (int ii = 0; ii < NTile; ii += 2) {
        jblas::utils::fp4x2 tmp;
        tmp.x = srcptr[i * ld_src + j * NTile + ii + 0];
        tmp.y = srcptr[i * ld_src + j * NTile + ii + 1];
        dstptr[i * ld_dst / 2 + j * NTile / 2 + ii / 2] = tmp;
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

inline JBLAS_CODE decompress_kblock_s4_fp(utils::int4x2* srcptr, float* dstptr, int row, int col, int ld_src,
                                          int ld_dst, float* scales, int k_offset, int kblock, int NPad) {
  // float fixed rowpack==1
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
      auto tmp = srcptr[i * ld_src / 2 + j / 2];
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

inline JBLAS_CODE decompress_kblock_s4_fp(utils::int4x2* srcptr, float* dstptr, int row, int col, int ld_src,
                                          int ld_dst, utils::bf16* scales, int k_offset, int kblock, int NPad) {
  // float fixed rowpack==1
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

inline JBLAS_CODE decompress_kblock_s4_fp(utils::int4x2* srcptr, utils::bf16* dstptr, int row, int col, int ld_src,
                                          int ld_dst, utils::bf16* scales, int k_offset, int kblock, int NPad) {
  // bf16 fixed rowpack==2
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src + j / 2];
      utils::bf16 bf16_ret1, bf16_ret2;
      bf16_ret1.fromfloat(float((int8_t)tmp.x << 4) * sptr[j / 2].tofloat());  // interleave with the same scale
      bf16_ret2.fromfloat(float((int8_t)tmp.y << 4) * sptr[j / 2].tofloat());
      dstptr[i * ld_dst + j + 0] = bf16_ret1;
      dstptr[i * ld_dst + j + 1] = bf16_ret2;
    }
  }
  return JblasSuccess;
}

inline JBLAS_CODE decompress_kblock_s4_fp(utils::int4x2* srcptr, utils::bf16* dstptr, int row, int col, int ld_src,
                                          int ld_dst, float* scales, int k_offset, int kblock, int NPad) {
  // bf16 fixed rowpack==2
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src + j / 2];
      utils::bf16 bf16_ret1, bf16_ret2;
      bf16_ret1.fromfloat(float((int8_t)tmp.x << 4) * sptr[j / 2]);  // interleave with the same scale
      bf16_ret2.fromfloat(float((int8_t)tmp.y << 4) * sptr[j / 2]);
      dstptr[i * ld_dst + j + 0] = bf16_ret1;
      dstptr[i * ld_dst + j + 1] = bf16_ret2;
    }
  }
  return JblasSuccess;
}

inline float fp4_dequantize(uint8_t val, float absmax) {
  float sign = (val & 0b1000) == 8 ? -1.0f : 1.0f;
  if ((val & 0b0100) == 4)                   // 0
    if ((val & 0b0010) == 2)                 // 01
      if ((val & 0b0001) == 1)               // 111
        return 0.25000000f * absmax * sign;  // 1111
      else
        return 0.16666667f * absmax * sign;  // 1110
    else if ((val & 0b0001) == 1)            // 110
      return 0.50000000f * absmax * sign;    // 1101
    else
      return 0.33333333f * absmax * sign;  // 1100
  else if ((val & 0b0010) == 2)            // 10
    if ((val & 0b0001) == 1)               // 101
      return 1.00000000f * absmax * sign;  // 1011
    else
      return 0.66666667f * absmax * sign;     // 1010
  else if ((val & 0b0001) == 1)               // 100
    return 5.208333333e-03f * absmax * sign;  // 1001
  else
    return 0.00000000f * absmax * sign;  // 1000
}

inline int8_t fp4_quantize(float x) {
  int sign = x < 0 ? 0b1000 : 0b0000;
  x = fabsf(x);
  if (x > 0.29166667f)
    if (x > 0.583333f)
      if (x > 0.8333333f)
        return 0b0011 + sign;
      else
        return 0b0010 + sign;
    else if (x > 0.4166667f)
      return 0b101 + sign;
    else
      return 0b100 + sign;
  else if (x > 0.0859375f)
    if (x > 0.20833333f)
      return 0b0111 + sign;
    else
      return 0b0110 + sign;
  else if (x > 0.00260417f)
    return 0b0001 + sign;
  else
    return 0b0000 + sign;
}

inline JBLAS_CODE decompress_kblock_fp4_fp(utils::fp4x2* srcptr, float* dstptr, int row, int col, int ld_src,
                                           int ld_dst, float* scales, int k_offset, int kblock, int NPad) {
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src + j / 2];
      dstptr[i * ld_dst + j + 0] = fp4_dequantize(tmp.x, sptr[j + 0]);
      dstptr[i * ld_dst + j + 1] = fp4_dequantize(tmp.y, sptr[j + 1]);
    }
  }
  return JblasSuccess;
}

inline JBLAS_CODE decompress_kblock_fp4_fp(utils::fp4x2* srcptr, utils::bf16* dstptr, int row, int col, int ld_src,
                                           int ld_dst, float* scales, int k_offset, int kblock, int NPad) {
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src + j / 2];
      utils::bf16 bf16_ret1, bf16_ret2;
      bf16_ret1.fromfloat(fp4_dequantize(tmp.x, sptr[j / 2]));  // interleave with the same scale
      bf16_ret2.fromfloat(fp4_dequantize(tmp.y, sptr[j / 2]));
      dstptr[i * ld_dst + j + 0] = bf16_ret1;
      dstptr[i * ld_dst + j + 1] = bf16_ret2;
    }
  }
  return JblasSuccess;
}

inline JBLAS_CODE decompress_kblock_fp4_fp(utils::fp4x2* srcptr, float* dstptr, int row, int col, int ld_src,
                                           int ld_dst, utils::bf16* scales, int k_offset, int kblock, int NPad) {
  // float fixed rowpack==1
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src + j / 2];
      dstptr[i * ld_dst + j + 0] = fp4_dequantize(tmp.x, sptr[j + 0].tofloat());
      dstptr[i * ld_dst + j + 1] = fp4_dequantize(tmp.y, sptr[j + 1].tofloat());
    }
  }
  return JblasSuccess;
}

inline JBLAS_CODE decompress_kblock_fp4_fp(utils::fp4x2* srcptr, utils::bf16* dstptr, int row, int col, int ld_src,
                                           int ld_dst, utils::bf16* scales, int k_offset, int kblock, int NPad) {
  // bf16 fixed rowpack==2
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src + j / 2];
      utils::bf16 bf16_ret1, bf16_ret2;
      bf16_ret1.fromfloat(fp4_dequantize(tmp.x, sptr[j / 2].tofloat()));
      bf16_ret2.fromfloat(fp4_dequantize(tmp.y, sptr[j / 2].tofloat()));
      dstptr[i * ld_dst + j + 0] = bf16_ret1;
      dstptr[i * ld_dst + j + 1] = bf16_ret2;
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE memcpy2d_dw2highw(void* srcptr, void* dstptr, int row, int col, int srcstride, int dststride) {
  auto bsrcptr = (char*)srcptr;
  auto bdstptr = (char*)dstptr;
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      std::memcpy(bdstptr + i * dststride + j * sizeof(jblas::utils::bf16),
                  bsrcptr + i * srcstride + j * sizeof(float) + 2, sizeof(jblas::utils::bf16));
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

inline JBLAS_CODE quantize_f32_fp4_rowblock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src,
                                            int ld_dst, float* scales, int blocksize) {
  for (int i = 0; i < col; i++) {
    for (size_t j = 0; j < row; j += blocksize) {
      float absmax = std::numeric_limits<float>::min();
      for (size_t ij = 0; ij < blocksize; ij++) {
        absmax = std::max(absmax, std::abs(srcptr[(j + ij) * ld_src + i]));
      }
      scales[j / blocksize * ld_dst + i] = absmax;
      for (size_t ij = 0; ij < blocksize; ij++) {
        dstptr[(j + ij) * ld_dst + i] = fp4_quantize(srcptr[(j + ij) * ld_src + i]);
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

inline JBLAS_CODE quantize_f32_s8_colblock(int row, int col, const float* srcptr, int ld_src, int8_t* dstptr,
                                           int ld_dst, float* scales, int ld_scale, int blocksize) {
  for (int i = 0; i < row; i++) {
    for (size_t j = 0; j < col; j += blocksize) {
      float maxval = std::numeric_limits<float>::min();
      float minval = 0.f;
      for (size_t ij = 0; ij < blocksize; ij++) {
        maxval = std::max(srcptr[(j + ij) + i * ld_src], maxval);
        minval = std::min(srcptr[(j + ij) + i * ld_src], minval);
      }
      maxval = std::max(std::abs(maxval), std::abs(minval));
      float scale = maxval / 127;
      float rscale = 1.f / scale;
      scales[j / blocksize + i * ld_scale] = scale;
      for (size_t ij = 0; ij < blocksize; ij++) {
        dstptr[(j + ij) + i * ld_dst] = utils::cast<float, int8_t>(srcptr[(j + ij) + i * ld_src] * rscale);
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
