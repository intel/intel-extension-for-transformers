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
#include <array>

#include "jit_blas_utils.h"
#include "kernel_avx2.h"
#include "kernel_avx512f.h"
#include "kernel_jit.h"
#include "kernel_ref.h"

namespace jblas {
namespace kernel {
namespace wrapper {
template <int NTile, int SrcBytes, int RowPack>
class PaddingInterleaveMN {
  // M x N ===> N/NTile x M/RowPack x NTile x RowPack (leading dim stride = NTile * dststride)
 public:
  template <JBLAS_ISA ISA_T>
  static JBLAS_CODE forward(void* srcptr, void* dstptr, int row, int col, int rowpad, int colpad, int srcstride,
                            int dststride) {
    return ref::padding_interleave(srcptr, dstptr, row, col, rowpad, colpad, srcstride, dststride, NTile, SrcBytes,
                                   RowPack);
  }
#if CompileBF16()
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"  // https://stackoverflow.com/a/49216021
#endif
  static void interleave_word(std::array<__m512i, 2>& dst) {
    static constexpr uint32_t perm_idx_a[16]{
        0 | 0,  1 | 0,  2 | 0,  3 | 0,   //
        0 | 16, 1 | 16, 2 | 16, 3 | 16,  //
        4 | 0,  5 | 0,  6 | 0,  7 | 0,   //
        4 | 16, 5 | 16, 6 | 16, 7 | 16,  //
    };
    static constexpr uint32_t perm_idx_b[16]{
        8 | 0,   9 | 0,   10 | 0,  11 | 0,   //
        8 | 16,  9 | 16,  10 | 16, 11 | 16,  //
        12 | 0,  13 | 0,  14 | 0,  15 | 0,   //
        12 | 16, 13 | 16, 14 | 16, 15 | 16,  //
    };
    static const auto v_perm_idx_a = _mm512_loadu_epi32(perm_idx_a);
    static const auto v_perm_idx_b = _mm512_loadu_epi32(perm_idx_b);

    __m512i tmp[2];
    tmp[0] = _mm512_unpacklo_epi16(dst[0], dst[1]);
    tmp[1] = _mm512_unpackhi_epi16(dst[0], dst[1]);
    dst[0] = _mm512_permutex2var_epi32(tmp[0], v_perm_idx_a, tmp[1]);
    dst[1] = _mm512_permutex2var_epi32(tmp[0], v_perm_idx_b, tmp[1]);
  }
  template <int tail>
  static std::array<__m512i, 2> load_fp16_bf16_interleave_word(const utils::fp16* a, size_t lda) {
    static_assert(tail > 0 && tail <= 2, "Unexpected tail value.");
    std::array<__m512i, 2> dst;
    for (int i = 0; i < tail; ++i) {
      dst[i] = (__m512i)(_mm512_cvtne2ps_pbh(                     //
          _mm512_cvtph_ps(_mm256_loadu_epi16(a + i * lda + 16)),  //
          _mm512_cvtph_ps(_mm256_loadu_epi16(a + i * lda + 0))));
    }
    for (int i = tail; i < 2; ++i) dst[i] = _mm512_setzero_epi32();
    interleave_word(dst);
    return dst;
  }
  template <int tail>
  static std::array<__m512i, 2> load_maskz_fp16_bf16_interleave_word(const utils::fp16* a, size_t lda, uint32_t mask) {
    static_assert(tail > 0 && tail <= 2, "Unexpected tail value.");

    const auto mask_lo = mask;
    const auto mask_hi = mask >> 16;
    std::array<__m512i, 2> dst;
    for (int i = 0; i < tail; ++i) {
      dst[i] = (__m512i)(_mm512_cvtne2ps_pbh(                                    //
          _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask_hi, a + i * lda + 16)),  //
          _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask_lo, a + i * lda + 0))));
    }
    for (int i = tail; i < 2; ++i) dst[i] = _mm512_setzero_epi32();
    interleave_word(dst);
    return dst;
  }
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#endif
  template <JBLAS_ISA ISA_T, typename T_SRC, typename T_DST = T_SRC>
  static JBLAS_CODE forward_cvt(const T_SRC* src, T_DST* dst, int row, int col, int row_pad, int col_pad, int src_step,
                                int dst_step) {
    // TODO(Yi): can forward_cvt of same i/o replace forward?
    static_assert(SrcBytes == sizeof(T_SRC), "SrcBytes should match T_SRC.");
#if CompileBF16()   // && 0
    if constexpr (  // TODO: avoid if constexpr
        std::is_same<T_SRC, utils::fp16>::value && std::is_same<T_DST, utils::bf16>::value && SrcBytes * RowPack == 4 &&
        RowPack == 2) {
      int i = 0;
      for (; i < row / RowPack * RowPack; i += RowPack) {
        int j = 0;
        for (; j < col / NTile * NTile; j += NTile) {
          static_assert(NTile % 32 == 0);
          for (int jj = 0; jj < NTile; jj += 32) {
            const auto xss = load_fp16_bf16_interleave_word<2>(src + i * src_step + j + jj, src_step);
            _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 0) * RowPack, xss[0]);
            _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 16) * RowPack, xss[1]);
          }
        }
        if (j < col) {  // j: tail processing
          int jj = 0;
          for (; j + jj < col / 32 * 32; jj += 32) {
            const auto xss = load_fp16_bf16_interleave_word<2>(src + i * src_step + j + jj, src_step);
            _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 0) * RowPack, xss[0]);
            _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 16) * RowPack, xss[1]);
          }
          if (j + jj < col) {  // jj: tail processing
            const uint32_t mask = (1U << (col - j - jj)) - 1;
            const auto xss = load_maskz_fp16_bf16_interleave_word<2>(src + i * src_step + j + jj, src_step, mask);
            _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 0) * RowPack, xss[0]);
            _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 16) * RowPack, xss[1]);
            jj += 32;
          }
          for (; jj < NTile; jj += 32) {  // jj: padding zero
            memset(dst + i * NTile + j * dst_step + jj * RowPack, 0, sizeof(T_DST) * 32 * RowPack);
          }
          j += NTile;
        }
        for (; j < col_pad; j += NTile) {  // j: padding zero
          memset(dst + i * NTile + j * dst_step, 0, sizeof(T_DST) * NTile * RowPack);
        }
      }
      if (i < row) {                      // i: tail processing
        static constexpr int tail_m = 1;  // must be 1
        int j = 0;
        for (; j < col / NTile * NTile; j += NTile) {
          static_assert(NTile % 32 == 0);
          for (int jj = 0; jj < NTile; jj += 32) {
            const auto xss = load_fp16_bf16_interleave_word<tail_m>(src + i * src_step + j + jj, src_step);
            _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 0) * RowPack, xss[0]);
            _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 16) * RowPack, xss[1]);
          }
        }
        if (j < col) {  // j: tail processing
          int jj = 0;
          for (; j + jj < col / 32 * 32; jj += 32) {
            const auto xss = load_fp16_bf16_interleave_word<tail_m>(src + i * src_step + j + jj, src_step);
            _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 0) * RowPack, xss[0]);
            _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 16) * RowPack, xss[1]);
          }
          if (j + jj < col) {  // jj: tail processing
            const uint32_t mask = (1U << (col - j - jj)) - 1;
            const auto xss = load_maskz_fp16_bf16_interleave_word<tail_m>(src + i * src_step + j + jj, src_step, mask);
            _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 0) * RowPack, xss[0]);
            _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 16) * RowPack, xss[1]);
            jj += 32;
          }
          for (; jj < NTile; jj += 32) {  // jj: padding zero
            memset(dst + i * NTile + j * dst_step + jj * RowPack, 0, sizeof(T_DST) * 32 * RowPack);
          }
          j += NTile;
        }
        for (; j < col_pad; j += NTile) {  // j: padding zero
          memset(dst + i * NTile + j * dst_step, 0, sizeof(T_DST) * NTile * RowPack);
        }
        i += RowPack;
      }
      for (; i < row_pad; i += RowPack) {  // i: padding zero
        for (int j = 0; j < col_pad; j += NTile) {
          memset(dst + i * NTile + j * dst_step, 0, sizeof(T_DST) * NTile * RowPack);
        }
      }
    }
    return JblasSuccess;
#endif
    return ref::padding_interleave(src, dst, row, col, row_pad, col_pad, src_step, dst_step, NTile, RowPack);
  }
};
template <int MTile, int SrcBytes, int colPack>
class PaddingTransInterleaveMN {
  // row and cols are in terms of src
  // M x N ===> M/MTile x N/colPack x MTile x colPack (leading dim stride = MTile * dststride)
 public:
  template <JBLAS_ISA ISA_T>
  static JBLAS_CODE forward(void* src, void* dst, int row, int col, int row_pad, int col_pad, int src_stride,
                            int dst_stride) {
    return ref::padding_trans_interleave(src, dst, row, col, row_pad, col_pad, src_stride, dst_stride, MTile, SrcBytes,
                                         colPack);
  }

#if CompileBF16()
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"  // https://stackoverflow.com/a/49216021
#endif
  static void tr_x16_dword(std::array<__m512i, 16>& dst) {
    __m512i tmp[16];

#pragma GCC unroll(8)
    for (int i = 0; i < 8; ++i) {
      tmp[2 * i] = _mm512_unpacklo_epi32(dst[2 * i], dst[2 * i + 1]);
      tmp[2 * i + 1] = _mm512_unpackhi_epi32(dst[2 * i], dst[2 * i + 1]);
    }

#pragma GCC unroll(4)
    for (int i = 0; i < 4; ++i) {
      dst[4 * i] = _mm512_unpacklo_epi64(tmp[4 * i], tmp[4 * i + 2]);
      dst[4 * i + 1] = _mm512_unpackhi_epi64(tmp[4 * i], tmp[4 * i + 2]);
      dst[4 * i + 2] = _mm512_unpacklo_epi64(tmp[4 * i + 1], tmp[4 * i + 3]);
      dst[4 * i + 3] = _mm512_unpackhi_epi64(tmp[4 * i + 1], tmp[4 * i + 3]);
    }

#pragma GCC unroll(2)
    for (int i = 0; i < 2; ++i) {
      tmp[8 * i + 0] = _mm512_shuffle_i32x4(dst[8 * i + 0], dst[8 * i + 4], 0x88);
      tmp[8 * i + 1] = _mm512_shuffle_i32x4(dst[8 * i + 1], dst[8 * i + 5], 0x88);
      tmp[8 * i + 2] = _mm512_shuffle_i32x4(dst[8 * i + 2], dst[8 * i + 6], 0x88);
      tmp[8 * i + 3] = _mm512_shuffle_i32x4(dst[8 * i + 3], dst[8 * i + 7], 0x88);
      tmp[8 * i + 4] = _mm512_shuffle_i32x4(dst[8 * i + 0], dst[8 * i + 4], 0xdd);
      tmp[8 * i + 5] = _mm512_shuffle_i32x4(dst[8 * i + 1], dst[8 * i + 5], 0xdd);
      tmp[8 * i + 6] = _mm512_shuffle_i32x4(dst[8 * i + 2], dst[8 * i + 6], 0xdd);
      tmp[8 * i + 7] = _mm512_shuffle_i32x4(dst[8 * i + 3], dst[8 * i + 7], 0xdd);
    }

    dst[0] = _mm512_shuffle_i32x4(tmp[0], tmp[8], 0x88);
    dst[1] = _mm512_shuffle_i32x4(tmp[1], tmp[9], 0x88);
    dst[2] = _mm512_shuffle_i32x4(tmp[2], tmp[10], 0x88);
    dst[3] = _mm512_shuffle_i32x4(tmp[3], tmp[11], 0x88);
    dst[4] = _mm512_shuffle_i32x4(tmp[4], tmp[12], 0x88);
    dst[5] = _mm512_shuffle_i32x4(tmp[5], tmp[13], 0x88);
    dst[6] = _mm512_shuffle_i32x4(tmp[6], tmp[14], 0x88);
    dst[7] = _mm512_shuffle_i32x4(tmp[7], tmp[15], 0x88);
    dst[8] = _mm512_shuffle_i32x4(tmp[0], tmp[8], 0xdd);
    dst[9] = _mm512_shuffle_i32x4(tmp[1], tmp[9], 0xdd);
    dst[10] = _mm512_shuffle_i32x4(tmp[2], tmp[10], 0xdd);
    dst[11] = _mm512_shuffle_i32x4(tmp[3], tmp[11], 0xdd);
    dst[12] = _mm512_shuffle_i32x4(tmp[4], tmp[12], 0xdd);
    dst[13] = _mm512_shuffle_i32x4(tmp[5], tmp[13], 0xdd);
    dst[14] = _mm512_shuffle_i32x4(tmp[6], tmp[14], 0xdd);
    dst[15] = _mm512_shuffle_i32x4(tmp[7], tmp[15], 0xdd);
  }
  template <int tail>
  static std::array<__m512i, 16> load_fp16_bf16_tr_x16_dword(const utils::fp16* a, size_t lda) {
    static_assert(tail > 0 && tail <= 16, "Unexpected tail value.");
    std::array<__m512i, 16> dst;
    for (int i = 0; i < tail; ++i) {
      dst[i] = (__m512i)(_mm512_cvtne2ps_pbh(                     //
          _mm512_cvtph_ps(_mm256_loadu_epi16(a + i * lda + 16)),  //
          _mm512_cvtph_ps(_mm256_loadu_epi16(a + i * lda + 0))));
    }
    for (int i = tail; i < 16; ++i) dst[i] = _mm512_setzero_epi32();
    tr_x16_dword(dst);
    return dst;
  }
  static constexpr decltype(load_fp16_bf16_tr_x16_dword<1>)* load_fp16_bf16_tr_x16_dword_tbl[17]{
      load_fp16_bf16_tr_x16_dword<1>,  load_fp16_bf16_tr_x16_dword<1>,  load_fp16_bf16_tr_x16_dword<2>,
      load_fp16_bf16_tr_x16_dword<3>,  load_fp16_bf16_tr_x16_dword<4>,  load_fp16_bf16_tr_x16_dword<5>,
      load_fp16_bf16_tr_x16_dword<6>,  load_fp16_bf16_tr_x16_dword<7>,  load_fp16_bf16_tr_x16_dword<8>,
      load_fp16_bf16_tr_x16_dword<9>,  load_fp16_bf16_tr_x16_dword<10>, load_fp16_bf16_tr_x16_dword<11>,
      load_fp16_bf16_tr_x16_dword<12>, load_fp16_bf16_tr_x16_dword<13>, load_fp16_bf16_tr_x16_dword<14>,
      load_fp16_bf16_tr_x16_dword<15>, load_fp16_bf16_tr_x16_dword<16>,
  };
  template <int tail>
  static std::array<__m512i, 16> load_maskz_fp16_bf16_tr_x16_dword(const utils::fp16* a, size_t lda, uint32_t mask) {
    static_assert(tail > 0 && tail <= 16, "Unexpected tail value.");
    std::array<__m512i, 16> dst;

    const auto mask_lo = mask;
    const auto mask_hi = mask >> 16;
    for (int i = 0; i < tail; ++i) {
      dst[i] = (__m512i)(_mm512_cvtne2ps_pbh(                                    //
          _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask_hi, a + i * lda + 16)),  //
          _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask_lo, a + i * lda + 0))));
    }
    for (int i = tail; i < 16; ++i) dst[i] = _mm512_setzero_epi32();
    tr_x16_dword(dst);
    return dst;
  }
  static constexpr decltype(load_maskz_fp16_bf16_tr_x16_dword<1>)* load_maskz_fp16_bf16_tr_x16_dword_tbl[17]{
      load_maskz_fp16_bf16_tr_x16_dword<1>,  load_maskz_fp16_bf16_tr_x16_dword<1>,
      load_maskz_fp16_bf16_tr_x16_dword<2>,  load_maskz_fp16_bf16_tr_x16_dword<3>,
      load_maskz_fp16_bf16_tr_x16_dword<4>,  load_maskz_fp16_bf16_tr_x16_dword<5>,
      load_maskz_fp16_bf16_tr_x16_dword<6>,  load_maskz_fp16_bf16_tr_x16_dword<7>,
      load_maskz_fp16_bf16_tr_x16_dword<8>,  load_maskz_fp16_bf16_tr_x16_dword<9>,
      load_maskz_fp16_bf16_tr_x16_dword<10>, load_maskz_fp16_bf16_tr_x16_dword<11>,
      load_maskz_fp16_bf16_tr_x16_dword<12>, load_maskz_fp16_bf16_tr_x16_dword<13>,
      load_maskz_fp16_bf16_tr_x16_dword<14>, load_maskz_fp16_bf16_tr_x16_dword<15>,
      load_maskz_fp16_bf16_tr_x16_dword<16>,
  };

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#endif

  // Note: rows/cols and i/j are in terms of src
  template <JBLAS_ISA ISA_T, typename T_SRC, typename T_DST = T_SRC>
  static JBLAS_CODE forward_cvt(const T_SRC* src, T_DST* dst, int row, int col, int row_pad, int col_pad, int src_step,
                                int dst_step) {
    // TODO(Yi): can forward_cvt of same i/o replace forward?
    static_assert(SrcBytes == sizeof(T_SRC), "SrcBytes should match T_SRC.");
#if CompileBF16()   // && 0
    if constexpr (  // TODO: avoid if constexpr
        std::is_same<T_SRC, utils::fp16>::value && std::is_same<T_DST, utils::bf16>::value && SrcBytes * colPack == 4) {
      assert(row_pad % 16 == 0 && col_pad % 32 == 0);
      int i = 0;
      for (; i < row / MTile * MTile; i += MTile) {
        assert(MTile % 16 == 0);
        int j = 0;
        for (; j < col / 32 * 32; j += 32) {
          for (int ii = 0; ii < MTile; ii += 16) {
            assert(MTile % 16 == 0);
            const auto xss = load_fp16_bf16_tr_x16_dword<16>(src + (i + ii) * src_step + j, src_step);
            for (int jj = 0; jj < 32; jj += 2) {
              _mm512_storeu_si512(dst + i * dst_step + ii * colPack + (j + jj) * MTile, xss[jj / 2]);
            }
          }
        }
        if (j < col) {  // j: tail processing
          for (int ii = 0; ii < MTile; ii += 16) {
            assert(MTile % 16 == 0);
            const uint32_t mask = (1U << (col - j)) - 1;
            const auto xss = load_maskz_fp16_bf16_tr_x16_dword<16>(src + (i + ii) * src_step + j, src_step, mask);
            for (int jj = 0; jj < 32; jj += 2) {
              _mm512_storeu_si512(dst + i * dst_step + ii * colPack + (j + jj) * MTile, xss[jj / 2]);
            }
          }
          j += 32;
        }
        for (; j < col_pad; j += 2) {  // j: padding zero
          memset(dst + i * dst_step + j * MTile, 0, 2 * sizeof(T_DST) * MTile);
        }
      }
      if (i < row) {  // i: tail processing
        int ii = 0;
        for (; i + ii < row / 16 * 16; ii += 16) {
          int j = 0;
          for (; j < col / 32 * 32; j += 32) {
            assert(MTile % 16 == 0);
            const auto xss = load_fp16_bf16_tr_x16_dword<16>(src + (i + ii) * src_step + j, src_step);
            for (int jj = 0; jj < 32; jj += 2) {
              _mm512_storeu_si512(dst + i * dst_step + ii * colPack + (j + jj) * MTile, xss[jj / 2]);
            }
          }
          if (j < col) {  // j: tail processing
            assert(MTile % 16 == 0);
            const uint32_t mask = (1U << (col - j)) - 1;
            const auto xss = load_maskz_fp16_bf16_tr_x16_dword<16>(src + (i + ii) * src_step + j, src_step, mask);
            for (int jj = 0; jj < 32; jj += 2) {
              _mm512_storeu_si512(dst + i * dst_step + ii * colPack + (j + jj) * MTile, xss[jj / 2]);
            }
            j += 32;
          }
          for (; j < col_pad; j += 2) {  // j: padding zero
            memset(dst + i * dst_step + ii * colPack + j * MTile, 0, 2 * sizeof(T_DST) * 16);
          }
        }
        if (i + ii < row) {  // ii: tail processing
          const int tbl_idx = row - i - ii;
          int j = 0;
          for (; j < col / 32 * 32; j += 32) {
            assert(MTile % 16 == 0);
            const auto xss = load_fp16_bf16_tr_x16_dword_tbl[tbl_idx](src + (i + ii) * src_step + j, src_step);
            for (int jj = 0; jj < 32; jj += 2) {
              _mm512_storeu_si512(dst + i * dst_step + ii * colPack + (j + jj) * MTile, xss[jj / 2]);
            }
          }
          if (j < col) {  // j: tail processing
            assert(MTile % 16 == 0);
            const uint32_t mask = (1U << (col - j)) - 1;
            const auto xss =
                load_maskz_fp16_bf16_tr_x16_dword_tbl[tbl_idx](src + (i + ii) * src_step + j, src_step, mask);
            for (int jj = 0; jj < 32; jj += 2) {
              _mm512_storeu_si512(dst + i * dst_step + ii * colPack + (j + jj) * MTile, xss[jj / 2]);
            }
            j += 32;
          }
          for (; j < col_pad; j += 2) {  // j: padding zero
            memset(dst + i * dst_step + ii * colPack + j * MTile, 0, 2 * sizeof(T_DST) * 16);
          }
          ii += 16;
        }
        for (; ii < MTile; ii += 16) {  // ii: padding zero
          for (int j = 0; j < col_pad; j += 2) {
            memset(dst + i * dst_step + ii * colPack + j * MTile, 0, 2 * sizeof(T_DST) * 16);
          }
        }
        assert(ii == MTile);
        i += MTile;
      }
      assert(row_pad % MTile == 0);
      for (; i < row_pad; i += MTile) {  // i: padding zero
        for (int j = 0; j < col_pad; j += 2) {
          memset(dst + i * dst_step + j * MTile, 0, 2 * sizeof(T_DST) * MTile);
        }
      }
      return JblasSuccess;
    }
#endif
    return ref::padding_trans_interleave(src, dst, row, col, row_pad, col_pad, src_step, dst_step, MTile, colPack);
  }
};

class Memcpy2D {
 public:
  template <JBLAS_ISA ISA_T>
  static JBLAS_CODE forward(void* srcptr, void* dstptr, int row, int col, int srcstride, int dststride) {
#if CompileAVX512F()
    if (utils::isa_base<ISA_T>::avx512f) {
      return kernel::jit::JitMemcpy2DAvx512f::forward(srcptr, dstptr, row, col, srcstride, dststride);
    }
#endif
    return kernel::ref::memcpy2d(srcptr, dstptr, row, col, srcstride, dststride);
  }

  template <JBLAS_ISA ISA_T>
  static JBLAS_CODE forward_with_gelu(void* srcptr, void* dstptr, int row, int col, int srcstride, int dststride) {
    if (utils::isa_base<ISA_T>::avx512f) {
      return kernel::jit::JitMemcpy2DAvx512f::forward_with_gelu(srcptr, dstptr, row, col, srcstride, dststride);
    }
  }
};

class Memcpy2DFp32CvtBf16 {
 public:
  template <JBLAS_ISA ISA_T>
  static JBLAS_CODE forward(void* srcptr, void* dstptr, int row, int col, int srcstride, int dststride) {
#if 0
    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {
        ((utils::bf16*)((char*)dstptr + i * dststride))[j] =
            static_cast<utils::bf16>(((float*)((char*)srcptr + i * srcstride))[j]);
      }
    }
    return JblasSuccess;
#elif CompileAVX512F()
    if (utils::isa_base<ISA_T>::avx512f) {
      return kernel::avx512f::fp32_cvt_bf16_2D_write_back(srcptr, dstptr, row, col, srcstride, dststride);
    }
#else
    return kernel::ref::memcpy2d_dw2highw(srcptr, dstptr, row, col, srcstride, dststride);
#endif
  }
};

template <int NTILE>
class CompressS8S4 {
 public:
  template <JBLAS_ISA ISA_T>
  static inline JBLAS_CODE forward(int8_t* srcptr, jblas::utils::int4x2* dstptr, int row, int col, int ld_src,
                                   int ld_dst) {
    return ref::compress_s8_s4<NTILE>(srcptr, dstptr, row, col, ld_src, ld_dst);
  }
};

template <int NTILE>
class CompressFp4 {
 public:
  template <JBLAS_ISA ISA_T>
  static inline JBLAS_CODE forward(int8_t* srcptr, jblas::utils::fp4x2* dstptr, int row, int col, int ld_src,
                                   int ld_dst) {
    return ref::compress_fp4<NTILE>(srcptr, dstptr, row, col, ld_src, ld_dst);
  }
};

template <typename _T>
class Transpose2D {
 public:
  template <JBLAS_ISA ISA_T>
  static inline JBLAS_CODE forward(const _T* srcptr, _T* dstptr, int row, int col, int ld_src, int ld_dst) {
    return ref::transpose2d(srcptr, dstptr, row, col, ld_src, ld_dst);
  }
};

class QuantizeS8RowBlock {
 public:
  template <JBLAS_ISA ISA_T>
  static inline JBLAS_CODE forward(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                                   float* scales, int blocksize) {
    if (row % blocksize != 0) {
      return JblasNotSupport;
    }
#if CompileAVX512F()
    if (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::quantize_f32_s8_rowblock(srcptr, dstptr, row, col, ld_src, ld_dst, scales, blocksize);
    }
#endif
    return ref::quantize_f32_s8_rowblock(srcptr, dstptr, row, col, ld_src, ld_dst, scales, blocksize);
  }
};

class QuantizeFp4RowBlock {
 public:
  template <JBLAS_ISA ISA_T>
  static inline JBLAS_CODE forward(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                                   float* scales, int blocksize) {
    if (row % blocksize != 0) {
      return JblasNotSupport;
    }
    return ref::quantize_f32_fp4_rowblock(srcptr, dstptr, row, col, ld_src, ld_dst, scales, blocksize);
  }
};
class QuantizeU8ColBlock {
 public:
  template <JBLAS_ISA ISA_T>
  static inline JBLAS_CODE forward(int row, int col, const float* srcptr, int ld_src, uint8_t* dstptr, int ld_dst,
                                   float* scales, int ld_scale, uint8_t* zps, int blocksize) {
#if CompileAVX512F()
    if (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::quantize_f32_u8_colblock(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale, zps,
                                               blocksize);
    }
#endif
    return ref::quantize_f32_u8_colblock(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale, zps, blocksize);
  }
};

class QuantizeS8ColBlock {
 public:
  template <JBLAS_ISA ISA_T>
  static inline JBLAS_CODE forward(int row, int col, const float* srcptr, int ld_src, int8_t* dstptr, int ld_dst,
                                   float* scales, int ld_scale, int blocksize) {
#if CompileAVX512F()
    if (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::quantize_f32_s8_colblock(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale, blocksize);
    }
#endif
    return ref::quantize_f32_s8_colblock(row, col, srcptr, ld_src, dstptr, ld_dst, scales, ld_scale, blocksize);
  }
};

class Broadcast {
 public:
  template <JBLAS_ISA ISA_T>
  static inline JBLAS_CODE forward(int num, const uint8_t& srcval, uint8_t* dstptr) {
#if CompileAVX512F()
    if (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::broadcast_u8(num, srcval, dstptr);
    }
#endif
    return ref::broadcast_u8(num, srcval, dstptr);
  }
};

class AccumulateDequantizeS32F32 {
 public:
  template <JBLAS_ISA ISA_T>
  static inline JBLAS_CODE forward(const int32_t* srcptr, float* dstptr, float alpha, float beta, int row, int col,
                                   int ld_src, int ld_dst, float* ascales, int ldas, float* wscales) {
#if CompileAVX512F()
    if (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::accumulate_dequantize_s32_f32(srcptr, dstptr, alpha, beta, row, col, ld_src, ld_dst, ascales,
                                                    ldas, wscales);
    }
#endif
    return ref::accumulate_dequantize_s32_f32(srcptr, dstptr, alpha, beta, row, col, ld_src, ld_dst, ascales, ldas,
                                              wscales);
  }
};

template <typename _DST_T>
class DecompressKBlockS4FP {
 public:
  template <JBLAS_ISA ISA_T, typename _T>
  static inline JBLAS_CODE forward(utils::int4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                   _T* scales, int k_offset, int kblock, int NPad) {
#if CompileAVX512F()
    if (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::decompress_kblock_s4_fp(srcptr, dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad);
    }
#endif
    return ref::decompress_kblock_s4_fp(srcptr, dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad);
  }
};

template <typename _DST_T>
class DecompressKBlockFp4Fp {
 public:
  template <JBLAS_ISA ISA_T, typename _T>
  static inline JBLAS_CODE forward(utils::fp4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                   _T* scales, int k_offset, int kblock, int NPad) {
#if CompileAVX512F()
    if (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::decompress_kblock_fp4_fp(srcptr, dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock,
                                               NPad);
    }
#endif
    return ref::decompress_kblock_fp4_fp(srcptr, dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad);
  }
};

class DecompressKBlockS4S8 {
 public:
  template <JBLAS_ISA ISA_T>
  static inline JBLAS_CODE forward(utils::int4x2* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst) {
    if (utils::isa_base<ISA_T>::avx512f) {
      return jit::decompress_s4_s8(srcptr, dstptr, row, col, ld_src, ld_dst);
    }
#if CompileAVX512F()
    if (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::decompress_s4_s8(srcptr, dstptr, row, col, ld_src, ld_dst);
    }
#endif
    return ref::decompress_s4_s8(srcptr, dstptr, row, col, ld_src, ld_dst);
  }
};

class DecompressKBlockS8F32 {
 public:
  template <JBLAS_ISA ISA_T, typename _T>
  static inline JBLAS_CODE forward(int8_t* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst, _T* scales,
                                   int k_offset, int kblock, int NPad) {
#if CompileAVX512F()
    if (utils::isa_base<ISA_T>::avx512f) {
      return jit::DequanKBlockS8F32::forward_avx512f(srcptr, dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock,
                                                     NPad);
    }
#endif
    return ref::decompress_kblock_s8_f32(srcptr, dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad);
  }
};

class AlphaBetaF32F32 {
 public:
  template <JBLAS_ISA ISA_T>
  static JBLAS_CODE forward(const float alpha, const float* srcptr, const int srcstep, const float beta,
                            const float* src1ptr, const int src1step, float* dstptr, const int dststep, const int M,
                            const int N) {
#if CompileAVX512F()
    if (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::alphabeta_f32_f32(alpha, srcptr, srcstep, beta, src1ptr, src1step, dstptr, dststep, M, N);
    }
#endif
#if CompileAVX2()
    if (utils::isa_base<ISA_T>::avx2) {
      return avx2::alphabeta_f32_f32(alpha, srcptr, srcstep, beta, src1ptr, src1step, dstptr, dststep, M, N);
    }
#endif
    return ref::alphabeta_f32_f32(alpha, srcptr, srcstep, beta, src1ptr, src1step, dstptr, dststep, M, N);
  }
};

class QuanOutS32U32 {
 public:
  template <JBLAS_ISA ISA_T>
  static JBLAS_CODE forward(const float alpha, const int32_t* srcptr, const int srcstep, uint8_t* dstptr,
                            const int dststep, const int M, const int N, float scaleSrc, float scaleDst, int zpDst) {
#if CompileAVX512F()
    if (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::quanout_s32_u32(alpha, srcptr, srcstep, dstptr, dststep, M, N, scaleSrc, scaleDst, zpDst);
    }
#endif
    return ref::quanout_s32_u32(alpha, srcptr, srcstep, dstptr, dststep, M, N, scaleSrc, scaleDst, zpDst);
  }
};

class MinMaxKBlock {
 public:
  template <JBLAS_ISA ISA_T>
  static inline JBLAS_CODE forward(const float* srcptr, int row, int col, int ld_src, float* minmaxptr, int ld_minmax,
                                   int fsize_minmax, int blocksize) {
    return ref::minmax_f32_kblock(srcptr, row, col, ld_src, minmaxptr, ld_minmax, fsize_minmax, blocksize);
  }
};

}  // namespace wrapper
}  // namespace kernel
}  // namespace jblas
