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
#include "jit_base.hpp"
#include "jit_blas.h"
#include "jit_blas_utils.h"
#include "kernel_wrapper.h"

namespace jblas {
namespace epilogue {
namespace gemm {
template <typename _SRC_T, typename _DST_T>
class AccumulatorWriteBack {
 public:
  struct Param {
    _DST_T* C;
    int ldc;
  };

  template <JBLAS_ISA ISA_T>
  JBLAS_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    static_assert(std::is_same<_SRC_T, float>::value,
                  "src data type must be float in AccumulatorWriteBack epilogue now.");
    if (std::is_same<_DST_T, jblas::utils::bf16>::value) {
      return kernel::wrapper::Memcpy2DFp32CvtBf16::template forward<ISA_T>(
          (void*)cacheptr, (void*)cptr, M, N, cachestep * sizeof(_SRC_T), _param.ldc * sizeof(_DST_T));
    } else if (sizeof(_SRC_T) == sizeof(_DST_T)) {
      return kernel::wrapper::Memcpy2D::template forward<ISA_T>(
          (void*)cacheptr, (void*)cptr, M, N * sizeof(_DST_T), cachestep * sizeof(_SRC_T), _param.ldc * sizeof(_DST_T));
    } else {
      assert(false);
    }
  }
};

template <typename _SRC_T, typename _DST_T>
class AccumulatorWriteBackWithGelu {
 public:
   struct Param {
    _DST_T* C;
    int ldc;
  };

  template <JBLAS_ISA ISA_T>
  JBLAS_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    static_assert(std::is_same<_SRC_T, float>::value && std::is_same<_DST_T, float>::value,
                  "src/dst data type must be float in AccumulatorWriteBackWithGelu epilogue now.");

    return kernel::wrapper::Memcpy2D::template forward_with_gelu<ISA_T>(
        (void*)cacheptr, (void*)cptr, M, N * sizeof(_DST_T), cachestep * sizeof(_SRC_T), _param.ldc * sizeof(_DST_T));
  }
};

class AlphaBetaProcessFp32 {
 public:
  struct Param {
    float *C, *D;
    int ldc, ldd;
    float alpha, beta;
  };

  template <JBLAS_ISA ISA_T>
  JBLAS_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    auto DOffset = M_offset * _param.ldd + N_offset;
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    auto dptr = _param.D + DOffset;
    return kernel::wrapper::AlphaBetaF32F32::template forward<ISA_T>(_param.alpha, cacheptr, cachestep, _param.beta,
                                                                     dptr, _param.ldd, cptr, _param.ldc, M, N);
  }
};

class AlphaBetaProcessS32U8 {
 public:
  struct Param {
    uint8_t* C;
    int ldc;
    float alpha;
    float scaleAcc, scaleC;
    int zpC;
  };

  template <JBLAS_ISA ISA_T>
  JBLAS_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    return kernel::wrapper::QuanOutS32U32::template forward<ISA_T>(_param.alpha, cacheptr, cachestep, cptr, _param.ldc,
                                                                   M, N, _param.scaleAcc, _param.scaleC, _param.zpC);
  }
};

}  // namespace gemm
}  // namespace epilogue
}  // namespace jblas
