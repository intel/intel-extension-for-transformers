#pragma once
#include "jit_base.hpp"
#include "jit_blas.h"
#include "jit_blas_utils.h"
#include "kernel_wrapper.h"

namespace jblas {
namespace epilogue {
namespace gemm {
class AlphaBetaProcessBase {};
class AlphaBetaProcessFp32 : protected AlphaBetaProcessBase {
 public:
  struct Param {
    float *C, *D;
    int ldc, ldd;
    float alpha, beta;
  };

  template <JBLAS_ISA ISA_T>
  JBLAS_CODE forward(const float *cacheptr, const int cachestep,
                     const int M_offset, const int N_offset, const int M,
                     const int N, const Param &_param) {
    auto DOffset = M_offset * _param.ldd + N_offset;
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    auto dptr = _param.D + DOffset;
    return kernel::wrapper::AlphaBetaF32F32::forward<ISA_T>(
        _param.alpha, cacheptr, cachestep, _param.beta, dptr, _param.ldd, cptr,
        _param.ldc, M, N);
  }
};

}  // namespace gemm
}  // namespace epilogue
}  // namespace jblas