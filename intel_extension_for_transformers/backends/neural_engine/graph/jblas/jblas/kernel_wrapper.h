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
#include "kernel_avx2.h"
#include "kernel_avx512f.h"
#include "kernel_jit.h"
#include "kernel_ref.h"

namespace jblas {
namespace kernel {
namespace wrapper {
template <int NTile, int SrcBytes, int RowPack>
class PaddingInterleaveMN {
 public:
  template <JBLAS_ISA ISA_T>
  static JBLAS_CODE forward(void* srcptr, void* dstptr, int row, int col, int rowpad, int colpad, int srcstride,
                            int dststride) {
    return ref::padding_interleave(srcptr, dstptr, row, col, rowpad, colpad, srcstride, dststride, NTile, SrcBytes,
                                   RowPack);
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

class DecompressKBlockS4F32 {
 public:
  template <JBLAS_ISA ISA_T, typename _T>
  static inline JBLAS_CODE forward(utils::int4x2* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst,
                                   _T* scales, int k_offset, int kblock, int NPad) {
#if CompileAVX512F()
    if (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::decompress_kblock_s4_f32(srcptr, dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock,
                                               NPad);
    }
#endif
    return ref::decompress_kblock_s4_f32(srcptr, dstptr, row, col, ld_src, ld_dst, scales, k_offset, kblock, NPad);
  }
};

class DecompressKBlockS4S8 {
 public:
  template <JBLAS_ISA ISA_T>
  static inline JBLAS_CODE forward(utils::int4x2* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst) {
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
