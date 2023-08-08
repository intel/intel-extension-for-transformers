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
        dstptr[i * ld_dst / 2 + j * NTile / 2 + ii / 2] = tmp;
      }
    }
  }
  return JblasSuccess;
}

template <int NTile>
static inline JBLAS_CODE compress_f4(int8_t* srcptr, jblas::utils::f4x2* dstptr, int row, int col, int ld_src,
                                     int ld_dst) {
  for (int i = 0; i < col; i += NTile) {
    for (int j = 0; j < row; j++) {
      for (int ii = 0; ii < NTile; ii += 2) {
        jblas::utils::f4x2 tmp;
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

inline float fp4_bnb_dequantize(uint8_t val, float absmax) {
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

inline int8_t fp4_bnb_quantize(float x) {
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

inline int8_t fp4_e2m1_quantize(float x) {
  // FP4 with bias of 1
  // first bit is a sign
  // subnormals
  // 0b000 = 0
  // 0b001 = 0.0625
  // 0b010 = 1
  // 0b011 = 1.5
  // 0b100 = 2
  // 0b101 = 3
  // 0b110 = 4
  // 0b111 = 6

  int sign = x < 0 ? 0b1000 : 0b0000;
  x = fabsf(x);
  if (x > 1.75f) {
    if (x > 3.5f) {
      if (x > 5.f)
        return 0b111 + sign;  // 6
      else
        return 0b110 + sign;  // 4
    } else {
      if (x > 2.5f)
        return 0b101 + sign;  // 3
      else
        return 0b100 + sign;  // 2
    }
  } else {
    if (x > 0.53125f) {
      if (x > 1.25f)
        return 0b011 + sign;  // 1.5
      else
        return 0b010 + sign;  // 1
    } else {
      if (x > 0.03125f)
        return 0b0001 + sign;  // 0.0625
      else
        return 0b0000 + sign;  // 0
    }
  }
}

inline float fp4_e2m1_dequantize(uint8_t val, float absmax) {
  float sign = (val & 0b1000) == 8 ? -1.0f : 1.0f;
  if ((val & 0b0100) == 4)            // 0
    if ((val & 0b0010) == 2)          // 01
      if ((val & 0b0001) == 1)        // 111
        return 0.5f * absmax * sign;  // 1111
      else
        return 0.3333333333333f * absmax * sign;  // 1110
    else if ((val & 0b0001) == 1)                 // 110
      return 0.25f * absmax * sign;               // 1101
    else
      return 0.1666666666667f * absmax * sign;  // 1100
  else if ((val & 0b0010) == 2)                 // 10
    if ((val & 0b0001) == 1)                    // 101
      return 0.125f * absmax * sign;            // 1011
    else
      return 0.08333333333333f * absmax * sign;  // 1010
  else if ((val & 0b0001) == 1)                  // 100
    return 5.208333333e-03f * absmax * sign;     // 1001
  else
    return 0.00000000f * absmax * sign;  // 1000
}

inline float nf4_dequantize(int8_t val, float absmax) {
  if ((val & 0b1000) == 8)
    if ((val & 0b0100) == 4)      // 1
      if ((val & 0b0010) == 2)    // 11
        if ((val & 0b0001) == 1)  // 111
          return 1.0f * absmax;
        else
          return 0.7229568362236023f * absmax;
      else if ((val & 0b0001) == 1)  // 110
        return 0.5626170039176941f * absmax;
      else
        return 0.44070982933044434f * absmax;
    else if ((val & 0b0010) == 2)  // 10
      if ((val & 0b0001) == 1)     // 101
        return 0.33791524171829224f * absmax;
      else
        return 0.24611230194568634f * absmax;
    else if ((val & 0b0001) == 1)  // 100
      return 0.16093020141124725f * absmax;
    else
      return 0.07958029955625534f * absmax;

  else if ((val & 0b0100) == 4)  // 0
    if ((val & 0b0010) == 2)     // 01
      if ((val & 0b0001) == 1)   // 011
        return 0.0f * absmax;
      else
        return -0.09105003625154495f * absmax;
    else if ((val & 0b0001) == 1)  // 010
      return -0.18477343022823334f * absmax;
    else
      return -0.28444138169288635f * absmax;
  else if ((val & 0b0010) == 2)  // 00
    if ((val & 0b0001) == 1)     // 001
      return -0.39491748809814453f * absmax;
    else
      return -0.5250730514526367f * absmax;
  else if ((val & 0b0001) == 1)  // 000
    return -0.6961928009986877f * absmax;
  else
    return -1.0f * absmax;
}

inline int8_t nf4_quantize(float x) {
  if (x > 0.03979014977812767f)
    if (x > 0.3893125355243683f)      // 1
      if (x > 0.6427869200706482f)    // 11
        if (x > 0.8614784181118011f)  // 111
          return 0b1111;
        else
          return 0b1110;
      else if (x > 0.5016634166240692f)  // 110
        return 0b1101;
      else
        return 0b1100;
    else if (x > 0.2035212516784668f)  // 10
      if (x > 0.2920137718319893f)     // 101
        return 0b1011;
      else
        return 0b1010;
    else if (x > 0.1202552504837513f)  // 100
      return 0b1001;
    else
      return 0b1000;
  else if (x > -0.33967943489551544f)  // 0
    if (x > -0.13791173323988914f)     // 01
      if (x > -0.045525018125772476f)  // 011
        return 0b0111;
      else
        return 0b0110;
    else if (x > -0.23460740596055984f)  // 010
      return 0b0101;
    else
      return 0b0100;
  else if (x > -0.6106329262256622f)  // 00
    if (x > -0.4599952697753906f)     // 001
      return 0b0011;
    else
      return 0b0010;
  else if (x > -0.8480964004993439f)  // 000
    return 0b0001;
  else
    return 0b0000;
}

template <JBLAS_F4_TYPE F4_T>
inline float f4_dequantize(int8_t v, float scale) {
  static_assert(F4_T == FP4_BNB || F4_T == NF4 || F4_T == FP4_E2M1, "Unsupported F4 type");
  switch (F4_T) {
    case FP4_BNB:
      return fp4_bnb_dequantize(v, scale);
    case NF4:
      return nf4_dequantize(v, scale);
    case FP4_E2M1:
      return fp4_e2m1_dequantize(v, scale);
    default:
      break;
  }
}

template <JBLAS_F4_TYPE F4_T>
inline int8_t f4_quantize(float x) {
  static_assert(F4_T == FP4_BNB || F4_T == NF4 || F4_T == FP4_E2M1, "Unsupported F4 type");
  switch (F4_T) {
    case FP4_BNB:
      return fp4_bnb_quantize(x);
    case NF4:
      return nf4_quantize(x);
    case FP4_E2M1:
      return fp4_e2m1_quantize(x);
    default:
      break;
  }
}

template <JBLAS_F4_TYPE F4_T>
inline JBLAS_CODE decompress_kblock_f4_fp(utils::f4x2* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst,
                                          float* scales, int k_offset, int kblock, int NPad) {
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src + j / 2];
      dstptr[i * ld_dst + j + 0] = f4_dequantize<F4_T>(tmp.x, sptr[j + 0]);
      dstptr[i * ld_dst + j + 1] = f4_dequantize<F4_T>(tmp.y, sptr[j + 1]);
    }
  }
  return JblasSuccess;
}

template <JBLAS_F4_TYPE F4_T>
inline JBLAS_CODE decompress_kblock_f4_fp(utils::f4x2* srcptr, utils::bf16* dstptr, int row, int col, int ld_src,
                                          int ld_dst, float* scales, int k_offset, int kblock, int NPad) {
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src + j / 2];
      utils::bf16 bf16_ret1, bf16_ret2;
      bf16_ret1.fromfloat(f4_dequantize<F4_T>(tmp.x, sptr[j / 2]));  // interleave with the same scale
      bf16_ret2.fromfloat(f4_dequantize<F4_T>(tmp.y, sptr[j / 2]));
      dstptr[i * ld_dst + j + 0] = bf16_ret1;
      dstptr[i * ld_dst + j + 1] = bf16_ret2;
    }
  }
  return JblasSuccess;
}

template <JBLAS_F4_TYPE F4_T>
inline JBLAS_CODE decompress_kblock_f4_fp(utils::f4x2* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst,
                                          utils::bf16* scales, int k_offset, int kblock, int NPad) {
  // float fixed rowpack==1
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src + j / 2];
      dstptr[i * ld_dst + j + 0] = f4_dequantize<F4_T>(tmp.x, sptr[j + 0].tofloat());
      dstptr[i * ld_dst + j + 1] = f4_dequantize<F4_T>(tmp.y, sptr[j + 1].tofloat());
    }
  }
  return JblasSuccess;
}

template <JBLAS_F4_TYPE F4_T>
inline JBLAS_CODE decompress_kblock_f4_fp(utils::f4x2* srcptr, utils::bf16* dstptr, int row, int col, int ld_src,
                                          int ld_dst, utils::bf16* scales, int k_offset, int kblock, int NPad) {
  // bf16 fixed rowpack==2
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src + j / 2];
      utils::bf16 bf16_ret1, bf16_ret2;
      bf16_ret1.fromfloat(f4_dequantize<F4_T>(tmp.x, sptr[j / 2].tofloat()));
      bf16_ret2.fromfloat(f4_dequantize<F4_T>(tmp.y, sptr[j / 2].tofloat()));
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

template <JBLAS_F4_TYPE F4_T>
inline JBLAS_CODE quantize_f32_f4_rowblock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src,
                                           int ld_dst, float* scales, int blocksize) {
  for (int i = 0; i < col; i++) {
    for (size_t j = 0; j < row; j += blocksize) {
      float absmax = std::numeric_limits<float>::min();
      for (size_t ij = 0; ij < blocksize; ij++) {
        absmax = std::max(absmax, std::abs(srcptr[(j + ij) * ld_src + i]));
      }
      scales[j / blocksize * ld_dst + i] = absmax;
      for (size_t ij = 0; ij < blocksize; ij++) {
        dstptr[(j + ij) * ld_dst + i] = f4_quantize<F4_T>(srcptr[(j + ij) * ld_src + i]);
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