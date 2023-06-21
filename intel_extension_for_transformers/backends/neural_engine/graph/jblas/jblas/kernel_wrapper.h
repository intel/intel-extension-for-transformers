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
  static JBLAS_CODE forward(void *srcptr, void *dstptr, int row, int col,
                            int rowpad, int colpad, int srcstride,
                            int dststride) {
    return ref::padding_interleave(srcptr, dstptr, row, col, rowpad, colpad,
                                   srcstride, dststride, NTile, SrcBytes,
                                   RowPack);
  }
};

class Memcpy2D {
 public:
  template <JBLAS_ISA ISA_T>
  static JBLAS_CODE forward(void *srcptr, void *dstptr, int row, int col,
                            int srcstride, int dststride) {
#if CompileAVX512F()
    if (utils::isa_base<ISA_T>::avx512f) {
      return kernel::jit::JitMemcpy2DAvx512f::forward(srcptr, dstptr, row, col,
                                                      srcstride, dststride);
    }
#endif
    return kernel::ref::memcpy2d(srcptr, dstptr, row, col, srcstride,
                                 dststride);
  }
};
template <int NTILE>
class CompressS8S4 {
 public:
  template <JBLAS_ISA ISA_T>
  static inline JBLAS_CODE forward(int8_t *srcptr, jblas::utils::int4x2 *dstptr,
                                   int row, int col, int ld_src, int ld_dst) {
    return ref::compress_s8_s4<NTILE>(srcptr, dstptr, row, col, ld_src, ld_dst);
  }
};

template <typename _T>
class Transpose2D {
 public:
  template <JBLAS_ISA ISA_T>
  static inline JBLAS_CODE forward(const _T *srcptr, _T *dstptr, int row,
                                   int col, int ld_src, int ld_dst) {
    return ref::transpose2d(srcptr, dstptr, row, col, ld_src, ld_dst);
  }
};

class QuantizeS8KBlock {
 public:
  template <JBLAS_ISA ISA_T>
  static inline JBLAS_CODE forward(const float *srcptr, int8_t *dstptr, int row,
                                   int col, int ld_src, int ld_dst,
                                   float *scales, int blocksize) {
    if (row % blocksize != 0) {
      return JblasNotSupport;
    }
#if CompileAVX512F()
    if (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::quantize_f32_s8_kblock(srcptr, dstptr, row, col, ld_src,
                                             ld_dst, scales, blocksize);
    }
#endif
    return ref::quantize_f32_s8_kblock(srcptr, dstptr, row, col, ld_src, ld_dst,
                                       scales, blocksize);
  }
};

class DecompressKBlockS4F32 {
 public:
  template <JBLAS_ISA ISA_T,typename _T>
  static inline JBLAS_CODE forward(utils::int4x2 *srcptr, float *dstptr,
                                   int row, int col, int ld_src, int ld_dst,
                                   _T *scales, int k_offset, int kblock,
                                   int NPad) {
#if CompileAVX512F()
    if (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::decompress_kblock_s4_f32(srcptr, dstptr, row, col, ld_src,
                                               ld_dst, scales, k_offset, kblock,
                                               NPad);
    }
#endif
    return ref::decompress_kblock_s4_f32(srcptr, dstptr, row, col, ld_src,
                                         ld_dst, scales, k_offset, kblock,
                                         NPad);
  }
};

class AlphaBetaF32F32 {
 public:
  template <JBLAS_ISA ISA_T>
  static JBLAS_CODE forward(const float alpha, const float *srcptr,
                            const int srcstep, const float beta,
                            const float *src1ptr, const int src1step,
                            float *dstptr, const int dststep, const int M,
                            const int N) {
#if CompileAVX512F()
    if (utils::isa_base<ISA_T>::avx512f) {
      return avx512f::alphabeta_f32_f32(alpha, srcptr, srcstep, beta, src1ptr,
                                        src1step, dstptr, dststep, M, N);
    }
#endif
#if CompileAVX2()
    if (utils::isa_base<ISA_T>::avx2) {
      return avx2::alphabeta_f32_f32(alpha, srcptr, srcstep, beta, src1ptr,
                                     src1step, dstptr, dststep, M, N);
    }
#endif
    return ref::alphabeta_f32_f32(alpha, srcptr, srcstep, beta, src1ptr,
                                  src1step, dstptr, dststep, M, N);
  }
};
}  // namespace wrapper
}  // namespace kernel
}  // namespace jblas