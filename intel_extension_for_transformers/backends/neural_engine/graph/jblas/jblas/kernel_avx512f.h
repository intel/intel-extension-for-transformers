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
#if CompileAMXBF16()
#pragma GCC target("avx512bf16")
#endif
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

static inline void convert_s4_s8(int8_t* dstptr, int8_t* srcptr, __m512i vmask, int LoadMask) {
  auto ymm = _mm256_maskz_loadu_epi32(LoadMask, (const __m256i*)(srcptr));
  auto zmm = unpack_4bits(ymm, vmask);
  _mm512_mask_storeu_epi64(dstptr, LoadMask, zmm);
}

constexpr void (*pad_fp4)(int8_t* dstptr, int8_t* srcptr, __m512i vmask, int) = &convert_s4_s8;

template <int N, typename _DST_T>
static inline void dequant_s8_N(_DST_T* dstptr, int8_t* srcptr, __m512* vscales) {
  static_assert(N % 16 == 0);
  int constexpr VLoop = N / 16;
  for (int iv = 0; iv < VLoop; iv += 1) {
    auto src_s8 = _mm_loadu_si128((__m128i*)(srcptr + iv * 16));
    auto zmm = _mm512_cvtepi8_epi32(src_s8);
    auto fzmm = _mm512_cvtepi32_ps(zmm);
    fzmm = _mm512_mul_ps(fzmm, vscales[iv]);
    if (std::is_same<_DST_T, float>::value) {
      _mm512_storeu_ps(dstptr + iv * 16, fzmm);
    } else if (std::is_same<_DST_T, utils::bf16>::value) {
      auto bf16_v = _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(fzmm), 2));  // TODO: bf16 intrinsic
      _mm256_storeu_si256((__m256i*)(dstptr + iv * 16), bf16_v);
    } else {
      assert(false);
    }
  }
}

static float fp4_dequant_fp32_LUT[] = {
    0.00000000f,        5.208333333e-03f,   0.66666667f,        1.00000000f,        0.33333333f,
    0.50000000f,        0.16666667f,        0.25000000f,        -1.f * 0.00000000f, -1.f * 5.208333333e-03f,
    -1.f * 0.66666667f, -1.f * 1.00000000f, -1.f * 0.33333333f, -1.f * 0.50000000f, -1.f * 0.16666667f,
    -1.f * 0.25000000f};

template <int N, typename _DST_T>
static inline void dequant_fp4_N(_DST_T* dstptr, int8_t* srcptr, __m512* vscales) {
  static_assert(N % 16 == 0);
  int constexpr VLoop = N / 16;
  for (int iv = 0; iv < VLoop; iv += 1) {
    auto idx = _mm_loadu_si128((__m128i*)(srcptr + iv * 16));
    idx = _mm_srli_epi32(idx, 4);
    auto pad_idx = _mm512_cvtepu8_epi32(idx);
    auto lut = _mm512_loadu_si512(fp4_dequant_fp32_LUT);
    auto fp32_dq_v = _mm512_permutexvar_epi32(pad_idx, lut);
    auto fzmm = _mm512_mul_ps((__m512)fp32_dq_v, vscales[iv]);
    if (std::is_same<_DST_T, float>::value) {
      _mm512_storeu_ps(dstptr + iv * 16, fzmm);
    } else if (std::is_same<_DST_T, utils::bf16>::value) {
      // TODO(zhe): bf16 LUT optimization.
      auto bf16_v = _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(fzmm), 2));
      _mm256_storeu_si256((__m256i*)(dstptr + iv * 16), bf16_v);
    } else {
      assert(false);
    }
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

static inline void vec_broadcast_epi32_1_2(__m512i* dst2regs, __m512i* src1regs) {
  dst2regs[0] = _mm512_unpacklo_epi32(src1regs[0], src1regs[0]);
  dst2regs[1] = _mm512_unpackhi_epi32(src1regs[0], src1regs[0]);
}

static inline void vec_broadcast_ps_1_2(__m512* dst2regs, __m512* src1regs, __m512i idxreg) {
  auto tmpreg = _mm512_permutexvar_epi64(idxreg, _mm512_castps_si512(src1regs[0]));
  dst2regs[0] = _mm512_castsi512_ps(_mm512_unpacklo_epi32(tmpreg, tmpreg));
  dst2regs[1] = _mm512_castsi512_ps(_mm512_unpackhi_epi32(tmpreg, tmpreg));
}

static inline void vec_broadcast_epi32_2_4(__m512i* dst4regs, __m512i* src2regs) {
  vec_broadcast_epi32_1_2(dst4regs, src2regs);
  vec_broadcast_epi32_1_2(dst4regs + 2, src2regs + 1);
}

template <typename _ST>
static inline JBLAS_CODE decompress_kblock_bit4_fp32(utils::bit4x2* srcptr, float* dstptr, int row, int col, int ld_src,
                                                     int ld_dst, _ST* scales, int k_offset, int kblock, int NPad,
                                                     void (*dequantize)(float*, int8_t*, __m512*),
                                                     void (*pad_bit4)(int8_t*, int8_t*, __m512i, int)) {
  uint32_t mask = 0xf0f0f0f0;
  auto zmm_mask = _mm512_set1_epi32(*(int*)&mask);
  if (col == 48) {
    constexpr int LoadMask64 = (1 << (64 / 8)) - 1;
    constexpr int LoadMask48 = (1 << (48 / 8)) - 1;
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
          pad_bit4(tmpbuf + iter64 * 64, (int8_t*)(srcptr + irow * ld_src + 32 * iter64), zmm_mask, LoadMask64);
        }
        for (int iterr = 0; iterr < UnrollRow; iterr++) {
          dequantize(dstptr + (irow + iterr) * ld_dst, tmpbuf + iterr * 48, vscales);
        }
      }
      for (; irow < row0; irow++) {
        pad_bit4(tmpbuf, (int8_t*)(srcptr + irow * ld_src), zmm_mask, LoadMask48);
        dequantize(dstptr + irow * ld_dst, tmpbuf, vscales);
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
          pad_bit4(tmpbuf + iter64 * 64, (int8_t*)(srcptr + (irow + irr) * ld_src + 32 * iter64), zmm_mask, LoadMask64);
        }
        for (int iterr = 0; iterr < UnrollRow; iterr++) {
          dequantize(dstptr + (irow + irr + iterr) * ld_dst, tmpbuf + iterr * 48, vscales);
        }
      }
    }
    if (irow < row) {
      for (int iv = 0; iv < 3; iv++) {
        vscales[iv] = vec_loadscalex16(scales + (k_offset + irow) / kblock * NPad + iv * 16);
      }
    }
    for (; irow < row; irow++) {
      pad_bit4(tmpbuf, (int8_t*)(srcptr + irow * ld_src), zmm_mask, LoadMask48);
      dequantize(dstptr + irow * ld_dst, tmpbuf, vscales);
    }
    return JblasSuccess;
  }
  return JblasNotSupport;
}

template <typename _ST>
static inline JBLAS_CODE decompress_kblock_bit4_bf16(utils::bit4x2* srcptr, utils::bf16* dstptr, int row, int col,
                                                     int ld_src, int ld_dst, _ST* scales, int k_offset, int kblock,
                                                     int NPad, void (*dequantize)(utils::bf16*, int8_t*, __m512*),
                                                     void (*pad_bit4)(int8_t*, int8_t*, __m512i, int)) {
  uint32_t mask = 0xf0f0f0f0;
  auto zmm_mask = _mm512_set1_epi32(*(int*)&mask);
  auto broadcast_idx = _mm512_setr_epi64(0, 4, 1, 5, 2, 6, 3, 7);
  if (col % 64 == 0) {
    constexpr int LoadMask64 = (1 << (64 / 8)) - 1;
    for (int icol = 0; icol < col; icol += 64) {
      __m512 vscales[4];
      int8_t tmpbuf[64];
      int row0 = kblock - k_offset % kblock;
      row0 = row0 == kblock ? 0 : row0;
      row0 = row0 > row ? row : row0;
      int row1 = row - row0;
      int irow = 0;
      if (row0) {
        for (int iv = 0; iv < 2; iv++) {
          auto tmpscale = vec_loadscalex16(scales + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2);
          vec_broadcast_ps_1_2(vscales + iv * 2, &tmpscale, broadcast_idx);
        }

        for (; irow < row0; irow++) {
          pad_bit4(tmpbuf, (int8_t*)(srcptr + irow * ld_src + icol / 2), zmm_mask, LoadMask64);
          dequantize(dstptr + irow * ld_dst + icol, tmpbuf, vscales);
        }
      }

      int row1_blk = utils::padto_le(row1, kblock) + row0;
      for (; irow < row1_blk; irow += kblock) {
        for (int iv = 0; iv < 2; iv++) {
          auto tmpscale = vec_loadscalex16(scales + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2);
          vec_broadcast_ps_1_2(vscales + iv * 2, &tmpscale, broadcast_idx);
        }

        for (int irr = 0; irr < kblock; irr += 1) {
          pad_bit4(tmpbuf, (int8_t*)(srcptr + (irow + irr) * ld_src + icol / 2), zmm_mask, LoadMask64);
          dequantize(dstptr + (irow + irr) * ld_dst + icol, tmpbuf, vscales);
        }
      }
      if (irow < row) {
        for (int iv = 0; iv < 2; iv++) {
          auto tmpscale = vec_loadscalex16(scales + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2);
          vec_broadcast_ps_1_2(vscales + iv * 2, &tmpscale, broadcast_idx);
        }
      }
      for (; irow < row; irow++) {
        pad_bit4(tmpbuf, (int8_t*)(srcptr + irow * ld_src + icol / 2), zmm_mask, LoadMask64);
        dequantize(dstptr + irow * ld_dst + icol, tmpbuf, vscales);
      }
    }

    return JblasSuccess;
  }
  return JblasNotSupport;
}

template <typename _ST, typename _DST_T>
static inline JBLAS_CODE decompress_kblock_s4_fp(utils::int4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
                                                 int ld_dst, _ST* scales, int k_offset, int kblock, int NPad) {
  if (std::is_same<_DST_T, float>::value) {
    return decompress_kblock_bit4_fp32<_ST>(srcptr, (float*)dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock,
                                            NPad, &dequant_s8_N<48, float>, &convert_s4_s8);
  } else if (std::is_same<_DST_T, utils::bf16>::value) {
    return decompress_kblock_bit4_bf16<_ST>(srcptr, (utils::bf16*)dstptr, row, col, ld_src, ld_dst, scales, k_offset,
                                            kblock, NPad, &dequant_s8_N<64, utils::bf16>, &convert_s4_s8);
  }
  return JblasNotSupport;
}

template <typename _ST, typename _DST_T>
static inline JBLAS_CODE decompress_kblock_fp4_fp(utils::fp4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
                                                  int ld_dst, _ST* scales, int k_offset, int kblock, int NPad) {
  if (std::is_same<_DST_T, float>::value) {
    return decompress_kblock_bit4_fp32<_ST>(srcptr, (float*)dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock,
                                            NPad, &dequant_fp4_N<48, float>, pad_fp4);
  } else if (std::is_same<_DST_T, utils::bf16>::value) {
    return decompress_kblock_bit4_bf16<_ST>(srcptr, (utils::bf16*)dstptr, row, col, ld_src, ld_dst, scales, k_offset,
                                            kblock, NPad, &dequant_fp4_N<64, utils::bf16>, pad_fp4);
  }
  return JblasNotSupport;
}

static inline JBLAS_CODE decompress_s4_s8(utils::int4x2* srcptr, int8_t* dstptr, int row, int col, int ld_src,
                                          int ld_dst) {
  uint32_t mask = 0xf0f0f0f0;
  auto zmm_mask = _mm512_set1_epi32(*(int*)&mask);
  if (col == ld_src) {
    size_t elesize = (size_t)row * col;
    size_t ele256 = utils::padto_le(elesize, 256);
    size_t ele64 = utils::padto_le(elesize, 64);
    size_t i = 0;
    constexpr int LoadMask64 = (1 << (64 / 8)) - 1;
    for (; i < ele256; i += 256) {
      convert_s4_s8(dstptr + i + 0, (int8_t*)(srcptr + i / 2 + 0), zmm_mask, LoadMask64);
      convert_s4_s8(dstptr + i + 64, (int8_t*)(srcptr + i / 2 + 32), zmm_mask, LoadMask64);
      convert_s4_s8(dstptr + i + 128, (int8_t*)(srcptr + i / 2 + 64), zmm_mask, LoadMask64);
      convert_s4_s8(dstptr + i + 192, (int8_t*)(srcptr + i / 2 + 96), zmm_mask, LoadMask64);
    }
    if (i + 64 <= ele64) {
      for (; i < ele64; i += 64) {
        convert_s4_s8(dstptr + i, (int8_t*)(srcptr + i / 2), zmm_mask, LoadMask64);
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
  for (; i < col; i++) {
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

static inline JBLAS_CODE quantize_f32_s8_colblock(int row, int col, const float* srcptr, int ld_src, int8_t* dstptr,
                                                  int ld_dst, float* scales, int ld_scale, int blocksize) {
  int constexpr VLen = 16;
  if (blocksize % VLen != 0) {
    return JblasNotSupport;
  }
  auto vpos = _mm512_set1_epi32(127);
  auto vneg = _mm512_set1_epi32(-128);
  int i = 0;
  for (int i = 0; i < row; i += 1) {
    for (size_t j = 0; j < col; j += blocksize) {
      __m512 vscale;
      __m512 vmaxval = _mm512_set1_ps(std::numeric_limits<float>::min());
      for (size_t ij = 0; ij < blocksize; ij += VLen) {
        auto vsrc = _mm512_loadu_ps(&srcptr[(j + ij) + i * ld_src]);
        vsrc = _mm512_abs_ps(vsrc);
        vmaxval = _mm512_max_ps(vmaxval, vsrc);
      }
      auto maxval = _mm512_reduce_max_ps(vmaxval);
      float scale = maxval / 127;
      scales[j / blocksize + i * ld_scale] = scale;
      auto vrscale = _mm512_set1_ps(1.f / scale);
      for (size_t ij = 0; ij < blocksize; ij += VLen) {
        auto vsrc = _mm512_loadu_ps(&srcptr[(j + ij) + i * ld_src]);
        vsrc = _mm512_mul_ps(vsrc, vrscale);
        auto vdsrc = _mm512_cvtps_epi32(vsrc);
        vdsrc = _mm512_min_epi32(vdsrc, vpos);
        vdsrc = _mm512_max_epi32(vdsrc, vneg);
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

static inline JBLAS_CODE fp32_cvt_bf16_2D_write_back(void* raw_srcptr, void* raw_dstptr, int row, int col,
                                                     int srcstride, int dststride) {
  char* srcptr = (char*)raw_srcptr;
  char* dstptr = (char*)raw_dstptr;
#if CompileBF16()
  constexpr int simd_proc_elt = 32;
  auto col_body_loop = col / simd_proc_elt;
  auto col_tail = col % simd_proc_elt;
  const uint32_t tail_mask = (1U << col_tail) - 1;
  for (int i = 0; i < row; i++) {
    auto src = srcptr + i * srcstride;
    auto dst = dstptr + i * dststride;
    int j = 0;
    for (; j < col_body_loop; j++) {
      _mm512_storeu_epi16(
          (dst + (j * simd_proc_elt) * sizeof(jblas::utils::bf16)),
          (__m512i)_mm512_cvtne2ps_pbh(_mm512_loadu_ps(src + sizeof(float) * simd_proc_elt * j + sizeof(float) * 16),
                                       _mm512_loadu_ps(src + sizeof(float) * simd_proc_elt * j + sizeof(float) * 0)));
    }
    if (col_tail > 0) {
      _mm512_mask_storeu_epi16(
          (dst + (j * simd_proc_elt) * sizeof(jblas::utils::bf16)),  //
          tail_mask,
          (__m512i)_mm512_cvtne2ps_pbh(
              _mm512_maskz_loadu_ps(tail_mask >> 16, src + sizeof(float) * simd_proc_elt * j + sizeof(float) * 16),
              _mm512_maskz_loadu_ps(tail_mask >> 0, src + sizeof(float) * simd_proc_elt * j + sizeof(float) * 0)));
    }
  }
#else
  constexpr int simd_proc_elt = 16;
  auto col_body_loop = col / simd_proc_elt;
  auto col_tail = col % simd_proc_elt;
  auto tail_mask = _cvtu32_mask16(0xffff >> (16 - col_tail));
  for (int i = 0; i < row; i++) {
    auto src = srcptr + i * srcstride;
    auto dst = dstptr + i * dststride;
    int j = 0;
    for (; j < col_body_loop; j++) {
      auto pack_bf16_value =
          _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_loadu_si512(src + sizeof(float) * simd_proc_elt * j), 16));
      _mm256_storeu_si256((__m256i*)(dst + (j * simd_proc_elt) * sizeof(jblas::utils::bf16)), pack_bf16_value);
    }
    if (col_tail > 0) {
      auto pack_bf16_tail =
          _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_loadu_si512(src + sizeof(float) * simd_proc_elt * j), 16));
      _mm256_mask_storeu_epi16(dst + (j * simd_proc_elt) * sizeof(jblas::utils::bf16), tail_mask, pack_bf16_tail);
    }
  }
#endif
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
