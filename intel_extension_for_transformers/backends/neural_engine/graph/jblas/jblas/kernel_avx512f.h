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
#if CompileAVX512F()
#include <immintrin.h>
#endif
namespace jblas {
namespace kernel {
namespace avx512f {
#if CompileAVX512F()
#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512bw", "avx512vl")
#else
#endif
static inline __m512i unpack_4bits(__m256i v4bits, __m512i vmask) {
  auto ymm1 = _mm256_slli_epi32(v4bits, 4);
  auto zmm = _mm512_cvtepi8_epi16(v4bits);
  auto zmm1 = _mm512_cvtepi8_epi16(ymm1);
  zmm = _mm512_slli_epi16(zmm, 8);
  zmm1 = _mm512_mask_mov_epi8(zmm1, 0xaaaaaaaaaaaaaaaa, zmm);
  zmm1 = _mm512_and_epi32(zmm1, vmask);
  return zmm1;
}

static inline void convert_s4_s8_48(int8_t* dstptr, int8_t* srcptr, __m512i vmask) {
  auto ymm = _mm256_maskz_loadu_epi64(0x7, (const __m256i*)(srcptr));
  auto zmm = unpack_4bits(ymm, vmask);
  _mm512_mask_storeu_epi64(dstptr, 0x3f, zmm);
}

static inline void convert_s4_s8_64(int8_t* dstptr, int8_t* srcptr, __m512i vmask) {
  auto ymm = _mm256_loadu_si256((const __m256i*)(srcptr));
  auto zmm = unpack_4bits(ymm, vmask);
  _mm512_storeu_si512(dstptr, zmm);
}

template <int N>
static inline void dequant_s8_N(float* dstptr, int8_t* srcptr, __m512* vscales) {
  static_assert(N % 16 == 0);
  int constexpr VLoop = N / 16;
  for (int iv = 0; iv < VLoop; iv += 1) {
    auto src_s8 = _mm_loadu_si128((__m128i*)(srcptr + iv * 16));
    auto zmm = _mm512_cvtepi8_epi32(src_s8);
    auto fzmm = _mm512_cvtepi32_ps(zmm);
    fzmm = _mm512_mul_ps(fzmm, vscales[iv]);
    _mm512_storeu_ps(dstptr + iv * 16, fzmm);
  }
}

static inline __m256i unpack_4bits_avx2(__m128i v4bits, __m256i vmask) {
  auto vsrc0_ = _mm_slli_epi32(v4bits, 4);
  auto v2src0 = _mm256_cvtepi8_epi16(v4bits);
  auto v2src0_ = _mm256_cvtepi8_epi16(vsrc0_);
  v2src0 = _mm256_slli_epi16(v2src0, 8);
  v2src0_ = _mm256_mask_mov_epi8(v2src0_, 0xaaaaaaaa, v2src0);
  v2src0_ = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(v2src0_), _mm256_castsi256_ps(vmask)));
  return v2src0_;
}

static inline void convert_s4_s8_48_avx2(int8_t* dstptr, int8_t* srcptr, __m256i vmask) {
  auto vsrc0 = _mm_loadu_si128((const __m128i*)srcptr);
  auto vsrc1 = _mm_loadl_epi64((const __m128i*)(srcptr + 16));
  auto dst0 = unpack_4bits_avx2(vsrc0, vmask);
  auto dst1 = unpack_4bits_avx2(vsrc1, vmask);
  _mm256_storeu_si256((__m256i*)dstptr, dst0);
  auto dst1low = _mm256_castsi256_si128(dst1);
  _mm_storeu_si128((__m128i*)(dstptr + 32), dst1low);
}

static inline void convert_s4_s8_24_avx2(int8_t* dstptr, int8_t* srcptr, __m256i vmask) {
  int8_t tmp[32];
  auto vsrc0 = _mm_loadu_si128((__m128i*)srcptr);
  auto dst0 = unpack_4bits_avx2(vsrc0, vmask);
  _mm256_storeu_si256((__m256i*)tmp, dst0);
  vsrc0 = _mm_loadu_si128((__m128i*)tmp);
  _mm_storeu_si128((__m128i*)(dstptr), vsrc0);
  *(int64_t*)(dstptr + 16) = *(int64_t*)(tmp + 16);
}

static inline void convert_s4_s8_64_avx2(int8_t* dstptr, int8_t* srcptr, __m256i vmask) {
  auto vsrc0 = _mm_loadu_si128((__m128i*)srcptr);
  auto vsrc1 = _mm_loadu_si128((__m128i*)(srcptr + 16));
  auto dst0 = unpack_4bits_avx2(vsrc0, vmask);
  auto dst1 = unpack_4bits_avx2(vsrc1, vmask);
  _mm256_storeu_si256((__m256i*)dstptr, dst0);
  _mm256_storeu_si256((__m256i*)(dstptr + 32), dst1);
}

template <int N>
static inline void dequant_s8_N_avx2(float* dstptr, int8_t* srcptr, __m256* vscales) {
  static_assert(N % 8 == 0);
  int constexpr VLoop = N / 8;
  for (int iv = 0; iv < VLoop; iv += 1) {
    auto src_s8 = _mm_loadu_si128((__m128i*)(srcptr + iv * 8));
    auto zmm = _mm256_cvtepi8_epi32(src_s8);
    auto fzmm = _mm256_cvtepi32_ps(zmm);
    fzmm = _mm256_mul_ps(fzmm, vscales[iv]);
    _mm256_storeu_ps(dstptr + iv * 8, fzmm);
  }
}

template <typename _ST>
static inline __m512 vec_loadscalex16(_ST* ptr) {
  return _mm512_loadu_ps(ptr);
}

template <>
inline __m512 vec_loadscalex16(utils::bf16* ptr) {
  auto vbf16 = _mm256_loadu_si256((__m256i*)ptr);
  auto vf32 = _mm512_cvtepu16_epi32(vbf16);
  return _mm512_castsi512_ps(_mm512_slli_epi32(vf32, 16));
}

template <typename _ST>
static inline JBLAS_CODE decompress_kblock_s4_f32(utils::int4x2* srcptr, float* dstptr, int row, int col, int ld_src,
                                                  int ld_dst, _ST* scales, int k_offset, int kblock, int NPad) {
  uint32_t mask = 0xf0f0f0f0;
  auto zmm_mask = _mm512_set1_epi32(*(int*)&mask);
  if (col == 48) {
    __m512 vscales[3];
    int constexpr UnrollRow = 4;
    int constexpr Loop64 = 48 * UnrollRow / 64;
    int8_t tmpbuf[48 * UnrollRow];
    int row0 = kblock - k_offset % kblock;
    row0 = row0 == kblock ? 0 : row0;
    row0 = row0 > row ? row : row0;
    int row1 = row - row0;
    int irow = 0;
    if (row0) {
      int rowpad4 = utils::padto_le(row0, UnrollRow);
      for (int iv = 0; iv < 3; iv++) {
        vscales[iv] = vec_loadscalex16(scales + (k_offset + irow) / kblock * NPad + iv * 16);
      }
      for (; irow < rowpad4; irow += UnrollRow) {
        for (int iter64 = 0; iter64 < Loop64; iter64++) {
          convert_s4_s8_64(tmpbuf + iter64 * 64, (int8_t*)(srcptr + irow * ld_src + 32 * iter64), zmm_mask);
        }
        for (int iterr = 0; iterr < UnrollRow; iterr++) {
          dequant_s8_N<48>(dstptr + (irow + iterr) * ld_dst, tmpbuf + iterr * 48, vscales);
        }
      }
      for (; irow < row0; irow++) {
        convert_s4_s8_48(tmpbuf, (int8_t*)(srcptr + irow * ld_src), zmm_mask);
        dequant_s8_N<48>(dstptr + irow * ld_dst, tmpbuf, vscales);
      }
    }

    int row1_blk = utils::padto_le(row1, kblock) + row0;
    assert(kblock % UnrollRow == 0);
    assert(ld_src == (48 / 2));  // no padding for unroll process

    for (; irow < row1_blk; irow += kblock) {
      for (int iv = 0; iv < 3; iv++) {
        vscales[iv] = vec_loadscalex16(scales + (k_offset + irow) / kblock * NPad + iv * 16);
      }

      for (int irr = 0; irr < kblock; irr += UnrollRow) {
        for (int iter64 = 0; iter64 < Loop64; iter64++) {
          convert_s4_s8_64(tmpbuf + iter64 * 64, (int8_t*)(srcptr + (irow + irr) * ld_src + 32 * iter64), zmm_mask);
        }
        for (int iterr = 0; iterr < UnrollRow; iterr++) {
          dequant_s8_N<48>(dstptr + (irow + irr + iterr) * ld_dst, tmpbuf + iterr * 48, vscales);
        }
      }
    }
    if (irow < row) {
      for (int iv = 0; iv < 3; iv++) {
        vscales[iv] = vec_loadscalex16(scales + (k_offset + irow) / kblock * NPad + iv * 16);
      }
    }
    for (; irow < row; irow++) {
      convert_s4_s8_48(tmpbuf, (int8_t*)(srcptr + irow * ld_src), zmm_mask);
      dequant_s8_N<48>(dstptr + irow * ld_dst, tmpbuf, vscales);
    }
    return JblasSuccess;
  }
  return JblasNotSupport;
}

static inline JBLAS_CODE decompress_s4_s8(utils::int4x2* srcptr, int8_t* dstptr, int row, int col, int ld_src,
                                          int ld_dst) {
  uint32_t mask = 0xf0f0f0f0;
  auto zmm_mask = _mm512_set1_epi32(*(int*)&mask);
  if (col == ld_src * 2) {
    size_t elesize = (size_t)row * col;
    size_t ele256 = utils::padto_le(elesize, 256);
    size_t ele64 = utils::padto_le(elesize, 64);
    size_t i = 0;
    for (; i < ele256; i += 256) {
      convert_s4_s8_64(dstptr + i + 0, (int8_t*)(srcptr + i / 2 + 0), zmm_mask);
      convert_s4_s8_64(dstptr + i + 64, (int8_t*)(srcptr + i / 2 + 32), zmm_mask);
      convert_s4_s8_64(dstptr + i + 128, (int8_t*)(srcptr + i / 2 + 64), zmm_mask);
      convert_s4_s8_64(dstptr + i + 192, (int8_t*)(srcptr + i / 2 + 96), zmm_mask);
    }
    if (i + 64 <= ele64) {
      for (; i < ele64; i += 64) {
        convert_s4_s8_64(dstptr + i, (int8_t*)(srcptr + i / 2), zmm_mask);
      }
    }
    for (; i < elesize; i += 2) {
      auto tmp = srcptr[i / 2];
      dstptr[i + 0] = (int8_t)tmp.x << 4;
      dstptr[i + 1] = (int8_t)tmp.y << 4;
    }
    return JblasSuccess;
  }
  return JblasNotSupport;
}

static inline JBLAS_CODE quantize_f32_s8_rowblock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src,
                                                  int ld_dst, float* scales, int blocksize) {
  int constexpr VLen = 16;
  auto v127 = _mm512_set1_ps(127.f);
  int col16 = utils::padto_le(col, 16);
  int i = 0;
  for (; i < col16; i += VLen) {
    for (size_t j = 0; j < row; j += blocksize) {
      __m512 vscale;
      __m512 vmaxval = _mm512_set1_ps(0.f);
      for (size_t ij = 0; ij < blocksize; ij++) {
        auto vsrc = _mm512_loadu_ps(&srcptr[(j + ij) * ld_src + i]);
        vsrc = _mm512_abs_ps(vsrc);
        vmaxval = _mm512_max_ps(vmaxval, vsrc);
      }
      vscale = _mm512_div_ps(vmaxval, v127);
      auto vrscale = _mm512_div_ps(v127, vmaxval);
      _mm512_storeu_ps(&scales[j / blocksize * ld_dst + i], vscale);
      for (size_t ij = 0; ij < blocksize; ij++) {
        auto vsrc = _mm512_loadu_ps(&srcptr[(j + ij) * ld_src + i]);
        vsrc = _mm512_mul_ps(vsrc, vrscale);
        auto vdsrc = _mm512_cvtps_epi32(vsrc);
        auto vbsrc = _mm512_cvtepi32_epi8(vdsrc);
        _mm_storeu_si128((__m128i*)&dstptr[(j + ij) * ld_dst + i], vbsrc);
      }
    }
  }
  for (; i < col; i += VLen) {
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

static inline JBLAS_CODE quantize_f32_u8_colblock(int row, int col, const float* srcptr, int ld_src, uint8_t* dstptr,
                                                  int ld_dst, float* scales, int ld_scale, uint8_t* zps,
                                                  int blocksize) {
  int constexpr VLen = 16;
  if (blocksize % VLen != 0) {
    return JblasNotSupport;
  }
  auto vff = _mm512_set1_epi32(255);
  auto v0 = _mm512_set1_epi32(0);
  int i = 0;
  for (int i = 0; i < row; i += 1) {
    for (size_t j = 0; j < col; j += blocksize) {
      __m512 vscale;
      __m512 vmaxval = _mm512_set1_ps(std::numeric_limits<float>::min());
      __m512 vminval = _mm512_set1_ps(0.f);
      for (size_t ij = 0; ij < blocksize; ij += VLen) {
        auto vsrc = _mm512_loadu_ps(&srcptr[(j + ij) + i * ld_src]);
        vmaxval = _mm512_max_ps(vmaxval, vsrc);
        vminval = _mm512_min_ps(vminval, vsrc);
      }
      auto maxval = _mm512_reduce_max_ps(vmaxval);
      auto minval = _mm512_reduce_min_ps(vminval);
      float scale = (maxval - minval) / 255;
      uint8_t zp = utils::cast<float, uint8_t>((0 - minval) / scale);
      scales[j / blocksize + i * ld_scale] = scale;
      zps[j / blocksize + i * ld_scale] = zp;
      auto vrscale = _mm512_set1_ps(1.f / scale);
      auto vdzp = _mm512_set1_epi32(zp);
      for (size_t ij = 0; ij < blocksize; ij += VLen) {
        auto vsrc = _mm512_loadu_ps(&srcptr[(j + ij) + i * ld_src]);
        vsrc = _mm512_mul_ps(vsrc, vrscale);
        auto vdsrc = _mm512_cvtps_epi32(vsrc);
        vdsrc = _mm512_add_epi32(vdsrc, vdzp);
        vdsrc = _mm512_min_epi32(vdsrc, vff);
        vdsrc = _mm512_max_epi32(vdsrc, v0);
        auto vbsrc = _mm512_cvtepi32_epi8(vdsrc);
        _mm_storeu_si128((__m128i*)&dstptr[(j + ij) + i * ld_dst], vbsrc);
      }
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE alphabeta_f32_f32(const float alpha, const float* srcptr, const int srcstep, const float beta,
                                           const float* src1ptr, const int src1step, float* dstptr, const int dststep,
                                           const int M, const int N) {
  int constexpr Vlen = 16;
  auto vN = utils::padto_le(N, Vlen);
  auto valpha = _mm512_set1_ps(alpha);
  auto vbeta = _mm512_set1_ps(beta);

  for (int i = 0; i < M; i++) {
    int j = 0;
    if (beta != 0.f) {
      for (; j < vN; j += Vlen) {
        auto vsrc = _mm512_loadu_ps(srcptr + i * srcstep + j);
        auto vsrc1 = _mm512_loadu_ps(src1ptr + i * src1step + j);
        auto vdst = _mm512_mul_ps(valpha, vsrc);
        vdst = _mm512_fmadd_ps(vbeta, vsrc1, vdst);
        _mm512_storeu_ps(dstptr + i * dststep + j, vdst);
      }
      for (; j < N; j += 1) {
        dstptr[i * dststep + j] = alpha * srcptr[i * srcstep + j] + beta * src1ptr[i * src1step + j];
      }
    } else {
      for (; j < vN; j += Vlen) {
        auto vsrc = _mm512_loadu_ps(srcptr + i * srcstep + j);
        auto vdst = _mm512_mul_ps(valpha, vsrc);
        _mm512_storeu_ps(dstptr + i * dststep + j, vdst);
      }
      for (; j < N; j += 1) {
        dstptr[i * dststep + j] = alpha * srcptr[i * srcstep + j];
      }
    }
  }
  return JblasSuccess;
}

static inline void vec_quanout_s32_u32_v16(const int32_t* srcptr, __m512& vfactor, __m512i& vzp, __m512i& vzeros,
                                           __m512i& v255, uint8_t* dstptr) {
  auto vsrcd = _mm512_loadu_si512(srcptr);
  auto vsrcf = _mm512_mul_ps(vfactor, _mm512_cvtepi32_ps(vsrcd));
  vsrcd = _mm512_cvtps_epi32(vsrcf);
  vsrcd = _mm512_add_epi32(vsrcd, vzp);
  vsrcd = _mm512_max_epi32(vsrcd, vzeros);
  vsrcd = _mm512_min_epi32(vsrcd, v255);
  auto vdstb = _mm512_cvtepi32_epi8(vsrcd);
  _mm_storeu_si128((__m128i*)dstptr, vdstb);
}

static inline JBLAS_CODE quanout_s32_u32(const float alpha, const int32_t* srcptr, const int srcstep, uint8_t* dstptr,
                                         const int dststep, const int M, const int N, float scaleSrc, float scaleDst,
                                         int zpDst) {
  float factor = alpha * scaleSrc / scaleDst;
  auto vfactor = _mm512_set1_ps(factor);
  auto vzp = _mm512_set1_epi32(zpDst);
  auto vzeros = _mm512_set1_epi32(0);
  auto v255 = _mm512_set1_epi32(255);
  int N64 = utils::padto_le(N, 64);
  int N48 = utils::padto_le(N, 48);
  int N16 = utils::padto_le(N, 16);
  for (int i = 0; i < M; i++) {
    int j = 0;
    for (; j < N64; j += 64) {
      for (int iv = 0; iv < 4; iv++) {
        vec_quanout_s32_u32_v16(&srcptr[i * srcstep + j + iv * 16], vfactor, vzp, vzeros, v255,
                                &dstptr[i * dststep + j + iv * 16]);
      }
    }
    if (N48 - j >= 48) {
      for (; j < N48; j += 48) {
        for (int iv = 0; iv < 3; iv++) {
          vec_quanout_s32_u32_v16(&srcptr[i * srcstep + j + iv * 16], vfactor, vzp, vzeros, v255,
                                  &dstptr[i * dststep + j + iv * 16]);
        }
      }
    }
    if (N16 - j >= 16) {
      for (; j < N16; j += 16) {
        vec_quanout_s32_u32_v16(&srcptr[i * srcstep + j], vfactor, vzp, vzeros, v255, &dstptr[i * dststep + j]);
      }
    }
    for (; j < N; j++) {
      float fsrc = float(srcptr[i * srcstep + j]) * factor;
      dstptr[i * dststep + j] = utils::cast<float, uint8_t>(fsrc + float(zpDst));
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE accumulate_dequantize_s32_f32(const int32_t* srcptr, float* dstptr, float alpha, float beta,
                                                       int row, int col, int ld_src, int ld_dst, float* ascales,
                                                       int ldas, float* wscales) {
  auto vbeta = _mm512_set1_ps(beta);
  int col16 = utils::padto_le(col, 16);
  for (int irow = 0; irow < row; irow++) {
    auto scale = ascales[irow * ldas] * alpha;
    auto valpha = _mm512_set1_ps(scale);
    int icol = 0;
    for (; icol < col16; icol += 16) {
      auto vwscale = _mm512_loadu_ps(wscales + icol);
      auto vscale = _mm512_mul_ps(valpha, vwscale);
      auto vdst = _mm512_loadu_ps(dstptr + irow * ld_dst + icol);
      vdst = _mm512_mul_ps(vdst, vbeta);
      auto vsrcd = _mm512_loadu_si512(srcptr + irow * ld_src + icol);
      auto vsrc = _mm512_cvtepi32_ps(vsrcd);
      vsrc = _mm512_fmadd_ps(vsrc, vscale, vdst);
      _mm512_storeu_ps(dstptr + irow * ld_dst + icol, vsrc);
    }
    for (; icol < col; icol += 1) {
      dstptr[irow * ld_dst + icol] =
          scale * wscales[icol] * srcptr[irow * ld_src + icol] + beta * dstptr[irow * ld_dst + icol];
    }
  }
  return JblasSuccess;
}

static inline JBLAS_CODE broadcast_u8(int num, const uint8_t& srcval, uint8_t* dstptr) {
  int i = 0;
  int constexpr VN = 64 / sizeof(srcval);
  int numv = utils::padto_le(num, VN);
  auto vsrc = _mm512_set1_epi8(srcval);
  for (; i < numv; i += VN) {
    _mm512_storeu_si512(dstptr + i, vsrc);
  }
  int num32 = utils::padto_le(num, 32);
  if (i + 32 <= num32) {
    for (; i < num32; i += 32) {
      _mm256_storeu_si256((__m256i*)(dstptr + i), _mm512_castsi512_si256(vsrc));
    }
  }
  for (; i < num; i++) {
    dstptr[i] = srcval;
  }
  return JblasSuccess;
}

#ifdef __GNUC__
#pragma GCC pop_options
#else
#endif
#endif
}  // namespace avx512f
}  // namespace kernel
}  // namespace jblas