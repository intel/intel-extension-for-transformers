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
#if CompileAVX2()
#include <immintrin.h>
#endif
namespace jblas {
namespace kernel {
namespace avx2 {
#if CompileAVX2()
#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target("avx2", "fma")
#else
#endif
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
#if 0
inline JBLAS_CODE decompress_avx2(utils::int4x2* srcptr, float* dstptr, int row,
                                  int col, int ld_src, int ld_dst,
                                  float* scales, int k_offset, int kblock) {
  uint32_t mask = 0xf0f0f0f0;
  auto vmask = _mm256_set1_epi32(*(int*)&mask);
  if (col == 24) {
    __m256 vscales[3];
    int constexpr UnrollRow = 8;
    int8_t tmpbuf[24 * UnrollRow];
    int row0 = kblock - k_offset % kblock;
    int row1 = row - row0;
    int irow = 0;
    if (row0) {
      for (int iv = 0; iv < 3; iv++) {
        vscales[iv] = _mm256_loadu_ps(
            scales + (k_offset + irow) / kblock * NPad + iv * 8);
      }
    }

    for (; irow < row0; irow++) {
      convert_s4_s8_48_avx2(tmpbuf, (int8_t*)(srcptr + irow * ld_src),
                                      vmask);
      dequant_s8_N_avx2<48>(dstptr + irow * ld_dst, tmpbuf, vscales);
    }

    int row1_blk = utils::padto_le(row1, kblock) + row0;
    assert(kblock % UnrollRow == 0);
    assert(ld_src == (48 / 2));  // no padding for unroll process

    for (; irow < row1_blk; irow += kblock) {
      for (int iv = 0; iv < 3; iv++) {
        vscales[iv] = _mm512_loadu_ps(
            scales + (k_offset + irow) / kblock * NPad + iv * 16);
      }

      int constexpr Loop64 = 48 * UnrollRow / 64;
      for (int irr = 0; irr < kblock; irr += UnrollRow) {
        for (int iter64 = 0; iter64 < Loop64; iter64++) {
          WeightS4::convert_s4_s8_64_avx512f(
              tmpbuf + iter64 * 64,
              (int8_t*)(srcptr + (irow + irr) * ld_src + 32 * iter64),
              zmm_mask);
        }
        for (int iterr = 0; iterr < UnrollRow; iterr++) {
          WeightS4::dequant_s8_N_avx512f<48>(
              dstptr + (irow + irr + iterr) * ld_dst, tmpbuf + iterr * 48,
              vscales);
        }
      }
    }
    if (irow < row) {
      for (int iv = 0; iv < 3; iv++) {
        vscales[iv] = _mm512_loadu_ps(
            scales + (k_offset + irow) / kblock * NPad + iv * 16);
      }
    }
    for (; irow < row; irow++) {
      WeightS4::convert_s4_s8_48_avx512f(
          tmpbuf, (int8_t*)(srcptr + irow * ld_src), zmm_mask);
      WeightS4::dequant_s8_N_avx512f<48>(dstptr + irow * ld_dst, tmpbuf,
                                         vscales);
    }
    return JblasSuccess;
  }
  return JblasNotSupport;
}
#endif
static inline JBLAS_CODE alphabeta_f32_f32(const float alpha, const float* srcptr, const int srcstep, const float beta,
                                           const float* src1ptr, const int src1step, float* dstptr, const int dststep,
                                           const int M, const int N) {
  int constexpr Vlen = 8;
  auto vN = utils::padto_le(N, Vlen);
  auto valpha = _mm256_set1_ps(alpha);
  auto vbeta = _mm256_set1_ps(beta);

  for (int i = 0; i < M; i++) {
    int j = 0;
    if (beta != 0.f) {
      for (; j < vN; j += Vlen) {
        auto vsrc = _mm256_loadu_ps(srcptr + i * srcstep + j);
        auto vsrc1 = _mm256_loadu_ps(src1ptr + i * src1step + j);
        auto vdst = _mm256_mul_ps(valpha, vsrc);
        vdst = _mm256_fmadd_ps(vbeta, vsrc1, vdst);
        _mm256_storeu_ps(dstptr + i * dststep + j, vdst);
      }
      for (; j < N; j += 1) {
        dstptr[i * dststep + j] = alpha * srcptr[i * srcstep + j] + beta * src1ptr[i * src1step + j];
      }
    } else {
      for (; j < vN; j += Vlen) {
        auto vsrc = _mm256_loadu_ps(srcptr + i * srcstep + j);
        auto vdst = _mm256_mul_ps(valpha, vsrc);
        _mm256_storeu_ps(dstptr + i * dststep + j, vdst);
      }
      for (; j < N; j += 1) {
        dstptr[i * dststep + j] = alpha * srcptr[i * srcstep + j];
      }
    }
  }
  return JblasSuccess;
}
#endif
}  // namespace avx2
}  // namespace kernel
}  // namespace jblas