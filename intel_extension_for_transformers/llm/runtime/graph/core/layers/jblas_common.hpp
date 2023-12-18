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
#include <utility>
#include "ne_jblas.h"
#include "jblas/jit_blas_prologue_b.h"
#include "jblas/jit_blas_device.h"
#include "jblas/jit_blas_utils.h"
#include "layers/ele_wise.h"
#include "jblas_defs.h"
#include "jblas/jit_blas_parallel.h"

namespace ne_jblas {

class ne_threading {
 public:
  static jblas::parallel::IThreading* get() {
#ifdef _OPENMP
    static jblas::parallel::OMPThreading DefaultThreading(4);
#else
    static jblas::parallel::StdThreading DefaultThreading(4);
#endif  // _OPNEMP
    return &DefaultThreading;
  }

  static void set_threads(int n_thread) { get()->set_threads(n_thread); }
};

template <typename T>
static inline void safe_delete(T* ptr) {
  if (ptr) {
    delete ptr;
  }
}

template <typename T>
static bool contains(const T& val, const T* set, size_t len) {
  for (size_t i = 0; i < len; i++) {
    if (val == set[i]) {
      return true;
    }
  }
  return false;
}

static bool hasISA(const uint64_t* coreset, size_t len) {
  GetCPUDevice();
  bool support = false;
  for (size_t i = 0; i < len; i++) {
    auto isa = jblas::gemm::CoreAttr::get_ISA(coreset[i]);
    switch (isa) {
      case JblasAVX:
        support |= _cd->AVX();
        break;
      case JblasAVX2:
        support |= _cd->AVX2();
        break;
      case JblasAMX_BF16:
        support |= _cd->AMX_BF16();
        break;
      case JblasAMX_INT8:
        support |= _cd->AMX_INT8();
        break;
      case JblasAVX512F:
        support |= _cd->AVX512F();
        break;
      case JblasAVX512_VNNI:
        support |= _cd->AVX512_VNNI();
        break;
      case JblasAVX512_FP16:
        support |= _cd->AVX512_FP16();
        break;
      case JblasAVX_VNNI:
        support |= _cd->AVX_VNNI();
        break;
      default:
        break;
    }
    if (support) {
      break;
    }
  }
  return support;
}

static inline bool samePackedWeight(jblas::storage::gemm::IWeightBase* ptr0, jblas::storage::gemm::IWeightBase* ptr1) {
  return ptr0->mCoreId == ptr1->mCoreId && ptr0->mPrologueID == ptr1->mPrologueID;
}

static inline bool samePackedWeight(jblas::storage::gemm::IWeightBase** ptrs, size_t len) {
  bool sameKernel = samePackedWeight(ptrs[0], ptrs[1]);
  if (sameKernel) {
    for (size_t i = 2; i < len; i++) {
      sameKernel &= samePackedWeight(ptrs[0], ptrs[i]);
    }
  }
  return sameKernel;
}

namespace custom {
namespace epilogue {
template <typename _T>
struct ParamAdd {
  _T *C, *D;
  int ldc, ldd;
  ParamAdd<_T> offset(int offset_) { return {C + offset_ * ldc, D + offset_ * ldd, ldc, ldd}; }
};
template <JBLAS_ISA ISA_T, typename _T>
class Add {
 public:
  using Param = ParamAdd<_T>;

  JBLAS_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto DOffset = M_offset * _param.ldd + N_offset;
    auto cptr = _param.C + COffset;
    auto dptr = _param.D + DOffset;
    // for (int i = 0; i < M; i++) {
    //   ne_vec_add_f32(N, cptr + i * _param.ldc,dptr + i * _param.ldd, cacheptr + i * cachestep);
    // }
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        cptr[i * _param.ldc + j] = dptr[i * _param.ldd + j] + cacheptr[i * cachestep + j];
      }
    }
    return JblasSuccess;
  }
};
template <JBLAS_ISA ISA_T>
using AddFp32 = Add<ISA_T, float>;

template <typename _T>
struct ParamMul {
  _T *C, *D;
  int ldc, ldd;
  ParamMul<_T> offset(int offset_) { return {C + offset_ * ldc, D + offset_ * ldd, ldc, ldd}; }
};
template <JBLAS_ISA ISA_T, typename _T>
class Mul {
 public:
  using Param = ParamMul<_T>;
  JBLAS_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto DOffset = M_offset * _param.ldd + N_offset;
    auto cptr = _param.C + COffset;
    auto dptr = _param.D + DOffset;
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        cptr[i * _param.ldc + j] = dptr[i * _param.ldd + j] * cacheptr[i * cachestep + j];
      }
    }
    return JblasSuccess;
  }
};
template <JBLAS_ISA ISA_T>
using MulFp32 = Mul<ISA_T, float>;

template <typename _T>
struct ParamAdd_Gelu {
  _T *C, *D;
  int ldc, ldd;
  ParamAdd_Gelu<_T> offset(int offset_) { return {C + offset_ * ldc, D + offset_ * ldd, ldc, ldd}; }
};
template <JBLAS_ISA ISA_T, typename _T>
class Add_Gelu {
 public:
  using Param = ParamAdd_Gelu<_T>;
  JBLAS_CODE forward(  // NOLINT [build/include_what_you_use]
      const float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M, const int N,
      const Param& _param, void* tmpcache, size_t cachesize) {
    auto COffset = M_offset * _param.ldc + N_offset;
    auto DOffset = M_offset * _param.ldd + N_offset;
    auto cptr = _param.C + COffset;
    auto dptr = _param.D + DOffset;
    for (int i = 0; i < M; i++) {
      ne_vec_add_f32(N, cptr + i * _param.ldc, dptr + i * _param.ldd, cacheptr + i * cachestep);
      // ne_vec_gelu_f32(N, cptr + i * _param.ldc, cptr + i * _param.ldc);
    }
    using GeluKernel = jblas::epilogue::gemm::AccumulatorWriteBackWithGeluFp32<ISA_T>;
    static GeluKernel ker;
    typename GeluKernel::Param param{_param.C, _param.ldc, NULL};
    auto ret = ker.forward(cptr, _param.ldc, M_offset, N_offset, M, N, param, tmpcache, cachesize);
    return ret;
  }
};
template <JBLAS_ISA ISA_T>
using Add_GeluFp32 = Add_Gelu<ISA_T, float>;

}  // namespace epilogue
}  // namespace custom
}  // namespace ne_jblas
