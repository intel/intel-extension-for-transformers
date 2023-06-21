#pragma once
#include "jit_blas_utils.h"

namespace jblas {
namespace kernel {
namespace ref {
static inline JBLAS_CODE padding_interleave(void* srcptr, void* dstptr, int row,
                                            int col, int rowpad, int colpad,
                                            int srcstride, int dststride,
                                            int NTile, int SrcBytes,
                                            int RowPack) {
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
              valid_row && valid_col
                  ? (sptr + (irow + ipa) * srcstride + (icol + iin) * SrcBytes)
                  : Zeros.data();
          for (int iele = 0; iele < SrcBytes; iele++) {
            (dptr + irow * NTile * SrcBytes + icol * dststride +
             iin * DstBytes + ipa * SrcBytes)[iele] = tmpptr[iele];
          }
        }
      }
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE dequan_s8_f32(int8_t* srcptr, float* dstptr, int row,
                                       int col, int ld_src, int ld_dst,
                                       float* scales) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      dstptr[i * ld_dst + j] = float(srcptr[i * ld_src + j]) * scales[j];
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE dequan_s8_bf16(int8_t* srcptr, uint16_t* dstptr,
                                        int row, int col, int ld_src,
                                        int ld_dst, float* scales) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      dstptr[i * ld_dst + j] = jblas::utils::cast<float, jblas::utils::bf16>(
                                   float(srcptr[i * ld_src + j]) * scales[j])
                                   .x;
    }
  }
  return JblasSuccess;
}

template <typename _T>
static inline JBLAS_CODE transpose2d(const _T* srcptr, _T* dstptr, int row,
                                     int col, int ld_src, int ld_dst) {
  for (int i = 0; i < col; i++) {
    for (size_t j = 0; j < row; j++) {
      dstptr[j + i * ld_dst] = srcptr[j * ld_src + i];
    }
  }
  return JblasSuccess;
}

template <int NTile>
static inline JBLAS_CODE compress_s8_s4(int8_t* srcptr,
                                        jblas::utils::int4x2* dstptr, int row,
                                        int col, int ld_src, int ld_dst) {
  for (int i = 0; i < col; i += NTile) {
    for (int j = 0; j < row; j++) {
      for (int ii = 0; ii < NTile; ii += 2) {
        jblas::utils::int4x2 tmp;
        tmp.x = jblas::utils::int4x2::convert(
            srcptr[i * ld_src + j * NTile + ii + 0]);
        tmp.y = jblas::utils::int4x2::convert(
            srcptr[i * ld_src + j * NTile + ii + 1]);
        dstptr[i * ld_dst + j * NTile / 2 + ii / 2] = tmp;
      }
    }
  }
  return JblasSuccess;
}

template <int NTile>
static inline JBLAS_CODE decompress_s4_f32(jblas::utils::int4x2* srcptr,
                                           float* dstptr, int row, int col,
                                           int ld_src, int ld_dst,
                                           float* scales) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src + j / 2];
      auto noffset = i * NTile + j % NTile;
      dstptr[i * ld_dst + j + 0] =
          float((int8_t)tmp.x << 4) * scales[noffset + 0];
      dstptr[i * ld_dst + j + 1] =
          float((int8_t)tmp.y << 4) * scales[noffset + 1];
    }
  }
  return JblasSuccess;
}

inline JBLAS_CODE decompress_kblock_s4_f32(utils::int4x2* srcptr, float* dstptr,
                                           int row, int col, int ld_src,
                                           int ld_dst, float* scales,
                                           int k_offset, int kblock, int NPad) {
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

inline JBLAS_CODE decompress_kblock_s4_f32(utils::int4x2* srcptr, float* dstptr,
                                           int row, int col, int ld_src,
                                           int ld_dst, utils::bf16* scales,
                                           int k_offset, int kblock, int NPad) {
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src + j / 2];
      dstptr[i * ld_dst + j + 0] =
          float((int8_t)tmp.x << 4) * sptr[j + 0].tofloat();
      dstptr[i * ld_dst + j + 1] =
          float((int8_t)tmp.y << 4) * sptr[j + 1].tofloat();
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE memcpy2d(void* srcptr, void* dstptr, int row, int col,
                                  int srcstride, int dststride) {
  auto bsrcptr = (char*)srcptr;
  auto bdstptr = (char*)dstptr;
  for (int i = 0; i < row; i++) {
    std::memcpy(bdstptr + i * dststride, bsrcptr + i * srcstride, col);
  }
  return JblasSuccess;
}

inline JBLAS_CODE quantize_f32_s8_kblock(const float* srcptr, int8_t* dstptr,
                                         int row, int col, int ld_src,
                                         int ld_dst, float* scales,
                                         int blocksize) {
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
        dstptr[(j + ij) * ld_dst + i] =
            utils::cast<float, int8_t>(srcptr[(j + ij) * ld_src + i] * rscale);
      }
    }
  }
  return JblasSuccess;
}
static inline JBLAS_CODE alphabeta_f32_f32(
    const float alpha, const float* srcptr, const int srcstep, const float beta,
    const float* src1ptr, const int src1step, float* dstptr, const int dststep,
    const int M, const int N) {
  if (beta != 0.f) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        dstptr[i * dststep + j] =
            alpha * srcptr[i * srcstep + j] + beta * src1ptr[i * src1step + j];
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
}  // namespace ref
}  // namespace kernel
}  // namespace jblas