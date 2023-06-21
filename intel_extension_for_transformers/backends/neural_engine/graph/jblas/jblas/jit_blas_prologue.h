#pragma once
#include "jit_base.hpp"
#include "jit_blas.h"
#include "jit_blas_gemm.h"
#include "jit_blas_utils.h"
#include "kernel_wrapper.h"
#include <immintrin.h>

namespace jblas {
namespace prologue {
class PrologueBase {};

namespace gemm {

template <class _GemmCore_T> class ActivationBase {
public:
    using AType = typename _GemmCore_T::AType;
  struct Param {
      const AType *A;
    int lda;
  };
  ActivationBase() {}
  template <JBLAS_ISA ISA_T>
  JBLAS_CODE getActivation(AType **dstptr, int *dststep, const Param &_param,
                           int m_size, int k_size, int m_offset, int k_offset) {
    auto aptr = const_cast<AType *>(_param.A);
    if (k_size % _GemmCore_T::KTILE == 0) {
      *dstptr = aptr + m_offset * _param.lda + k_offset;
      *dststep = _param.lda;
      return JblasSuccess;
    } else {
      auto k_pad = utils::padto(k_size, _GemmCore_T::KTILE);
      *dststep = k_pad;
      return kernel::wrapper::Memcpy2D::forward<ISA_T>(
          aptr + m_offset * _param.lda + k_offset, *dstptr, m_size,
          k_size * sizeof(AType), _param.lda * sizeof(AType),
          k_pad * sizeof(AType));
    }
    return JblasSuccess;
  }
};
} // namespace gemm
} // namespace prologue
} // namespace jblas