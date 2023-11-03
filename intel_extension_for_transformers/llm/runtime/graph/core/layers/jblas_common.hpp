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
#include "ne_jblas.h"
#include "jblas/jit_blas_weight_compression.h"
#include "jblas/jit_blas_transformer.h"
#include "layers/ele_wise.h"

namespace ne_jblas {
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

static bool hasISA(const jblas::gemm::GemmCoreType* set, size_t len) {
  GetCPUDevice();
  bool support = false;
  for (size_t i = 0; i < len; i++) {
    switch (set[i]) {
      case jblas::gemm::GemmCoreType::AMX_INT8_16x48_KBLOCK:
      case jblas::gemm::GemmCoreType::AMX_INT8_16x48:
      case jblas::gemm::GemmCoreType::AMX_INT8_16x64:
      case jblas::gemm::GemmCoreType::AMX_INT8_16x48_SS:
        support |= _cd->AMX_INT8();
        break;
      case jblas::gemm::GemmCoreType::AMX_BF16_16x48:
      case jblas::gemm::GemmCoreType::AMX_BF16_16x64:
        support |= _cd->AMX_BF16();
        break;
      case jblas::gemm::GemmCoreType::AVX512_VNNI_3x48_KBLOCK:
      case jblas::gemm::GemmCoreType::AVX512_VNNI_4x48_KBLOCK:
      case jblas::gemm::GemmCoreType::AVX512_VNNI_8x48:
        support |= _cd->AVX512_VNNI();
        break;
      case jblas::gemm::GemmCoreType::AVX512_FP16_8x64:
      case jblas::gemm::GemmCoreType::AVX512_FP16_8x96:
        support |= _cd->AVX512_FP16();
        break;
      case jblas::gemm::GemmCoreType::AVX512F_8x48:
        support |= _cd->AVX512F();
        break;
      case jblas::gemm::GemmCoreType::AVX2_4X24:
      case jblas::gemm::GemmCoreType::AVX2_2X48:
        support |= _cd->AVX2();
        break;
      case jblas::gemm::GemmCoreType::AVX_VNNI_1x48_KBLOCK:
      case jblas::gemm::GemmCoreType::AVX_VNNI_2x48:
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

static inline bool samePackedWeight(jblas::prologue::gemm::WeightBase* ptr0, jblas::prologue::gemm::WeightBase* ptr1) {
  return ptr0->mCoreType == ptr1->mCoreType && ptr0->mPrologueID == ptr1->mPrologueID;
}

static inline bool samePackedWeight(jblas::prologue::gemm::WeightBase** ptrs, size_t len) {
  assert(len >= 2);
  bool sameKernel = samePackedWeight(ptrs[0], ptrs[1]);
  if (sameKernel) {
    for (size_t i = 2; i < len; i++) {
      sameKernel &= samePackedWeight(ptrs[0], ptrs[i]);
    }
  }
  return sameKernel;
}

template <class T, JBLAS_ISA ISA>
using WeiS4ClipFp32 = jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32<T, ISA>;

template <class T, JBLAS_ISA ISA>
using WeiS8Fp32 = jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32<T, ISA>;

template <class T, JBLAS_ISA ISA>
using WeiS8Fp32PerN = jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32PerChannelN<T, ISA>;

template <class T, JBLAS_ISA ISA>
using WeiS4ClipFp32PerN = jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32PerN<T, ISA>;

using WeightCompType = jblas::prologue::weight_comp::gemm_kblcok::PrologueBIDs;

using SS4Fp32 = jblas::prologue::weight_comp::gemm_kblcok::StorageWeightS4ScaleFp32;
using SS8Fp32 = jblas::prologue::weight_comp::gemm_kblcok::StorageWeightS8ScaleFp32;
using SS8Fp32PerN = jblas::prologue::weight_comp::gemm_kblcok::StorageWeightS8ScaleFp32PerChannelN;
using SS4Fp32PerN = jblas::prologue::weight_comp::gemm_kblcok::StorageWeightS4ScaleFp32PerChannelN;

using GcCompAVX2 = jblas::gemm::GemmCore_Row_NN_4x24_AVX2;
using GcCompFp32 = jblas::gemm::GemmCore_Row_NN_8x48_AVX512F;
using GcCompInt8KBlock = jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK;
using GcCompBf16 = jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16;
using GcCompFp16 = jblas::gemm::GemmCore_Row_NN_8x64_AVX512_FP16;
using GcCompInt8 = jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI;

constexpr jblas::gemm::GemmCoreType GcCompInt8KBlockSet[] = {jblas::gemm::GemmCoreType::AMX_INT8_16x48_KBLOCK,
                                                             jblas::gemm::GemmCoreType::AVX512_VNNI_3x48_KBLOCK};

constexpr jblas::gemm::GemmCoreType GcCompInt8Set[] = {jblas::gemm::GemmCoreType::AMX_INT8_16x48_SS,
                                                       jblas::gemm::GemmCoreType::AVX512_VNNI_8x48};

namespace custom {
namespace epilogue {
template <JBLAS_ISA ISA_T, typename _T>
class Silu {
 public:
  struct Param {
    _T* C;
    int ldc;
  };
  using SiluKernel = jblas::epilogue::gemm::CustomAccumulatorWriteBackWithEltop<ISA_T, float, float, SWISH>;

  JBLAS_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    float alpha = -1.f;
    typename SiluKernel::Param param{_param.C, _param.ldc, &alpha};
    static SiluKernel ker;
    auto ret = ker.forward(cacheptr, cachestep, M_offset, N_offset, M, N, param);
    return JblasSuccess;
  }
};
template <JBLAS_ISA ISA_T>
using SiluFp32 = Silu<ISA_T, float>;

template <JBLAS_ISA ISA_T>
class DequantSiluFp32 : protected jblas::epilogue::gemm::DequantInt32ToFp32<ISA_T> {
 public:
  using Parent = jblas::epilogue::gemm::DequantInt32ToFp32<ISA_T>;
  using Param = typename Parent::Param;
  using SiluKernel = jblas::epilogue::gemm::CustomAccumulatorWriteBackWithEltop<ISA_T, float, float, SWISH>;

  JBLAS_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    float alpha = -1.f;
    typename SiluKernel::Param param{_param.C, _param.ldc, &alpha};
    Parent::forward(cacheptr, cachestep, M_offset, N_offset, M, N, _param);
    static SiluKernel ker;
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    auto ret = ker.forward(cptr, _param.ldc, M_offset, N_offset, M, N, param);
    return JblasSuccess;
  }
};

template <JBLAS_ISA ISA_T>
class ZpDequantSiluFp32 : protected jblas::epilogue::gemm::ZpDequantInt32ToFp32<ISA_T> {
 public:
  using Parent = jblas::epilogue::gemm::ZpDequantInt32ToFp32<ISA_T>;
  using Param = typename Parent::Param;
  using SiluKernel = jblas::epilogue::gemm::CustomAccumulatorWriteBackWithEltop<ISA_T, float, float, SWISH>;

  JBLAS_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    float alpha = -1.f;
    typename SiluKernel::Param param{_param.C, _param.ldc, &alpha};
    Parent::forward(cacheptr, cachestep, M_offset, N_offset, M, N, _param);
    static SiluKernel ker;
    auto COffset = M_offset * _param.ldc + N_offset;
    auto cptr = _param.C + COffset;
    auto ret = ker.forward(cptr, _param.ldc, M_offset, N_offset, M, N, param);
    return JblasSuccess;
  }
};

template <JBLAS_ISA ISA_T, typename _T>
class Gelu {
 public:
  struct Param {
    _T* C;
    int ldc;
  };

  JBLAS_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    using GeluKernel = jblas::epilogue::gemm::AccumulatorWriteBackWithGeluFp32<ISA_T>;
    static GeluKernel ker;
    typename GeluKernel::Param param{_param.C, _param.ldc, NULL};
    auto ret = ker.forward(cacheptr, cachestep, M_offset, N_offset, M, N, param);
    return JblasSuccess;
  }
};

template <JBLAS_ISA ISA_T>
using GeluFp32 = Gelu<ISA_T, float>;

template <JBLAS_ISA ISA_T, typename _T>
class Add {
 public:
  struct Param {
    _T *C, *D;
    int ldc, ldd;
  };

  JBLAS_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
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

template <JBLAS_ISA ISA_T, typename _T>
class ZpDequantAdd : protected jblas::epilogue::gemm::ZpDequantInt32ToFp32<ISA_T> {
 public:
  using Parent = jblas::epilogue::gemm::ZpDequantInt32ToFp32<ISA_T>;
  using PParam = typename Parent::Param;
  struct Param {
    PParam ZpDequant;
    _T* D;
    int ldd;
  };

  JBLAS_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    Parent::forward(cacheptr, cachestep, M_offset, N_offset, M, N, _param.ZpDequant);
    auto COffset = M_offset * _param.ZpDequant.ldc + N_offset;
    auto DOffset = M_offset * _param.ldd + N_offset;
    auto cptr = _param.ZpDequant.C + COffset;
    auto dptr = _param.D + DOffset;
    for (int i = 0; i < M; i++) {
      ne_vec_add_f32(N, cptr + i * _param.ZpDequant.ldc, dptr + i * _param.ldd, cptr + i * _param.ZpDequant.ldc);
    }
    return JblasSuccess;
  }
};
template <JBLAS_ISA ISA_T>
using ZpDequantAddFp32 = ZpDequantAdd<ISA_T, float>;

template <JBLAS_ISA ISA_T, typename _T>
class DequantAdd : protected jblas::epilogue::gemm::DequantInt32ToFp32<ISA_T> {
 public:
  using Parent = jblas::epilogue::gemm::DequantInt32ToFp32<ISA_T>;
  using PParam = typename Parent::Param;
  struct Param {
    PParam Dequant;
    _T* D;
    int ldd;
  };

  JBLAS_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    Parent::forward(cacheptr, cachestep, M_offset, N_offset, M, N, _param.Dequant);
    auto COffset = M_offset * _param.Dequant.ldc + N_offset;
    auto DOffset = M_offset * _param.ldd + N_offset;
    auto cptr = _param.Dequant.C + COffset;
    auto dptr = _param.D + DOffset;
    for (int i = 0; i < M; i++) {
      ne_vec_add_f32(N, cptr + i * _param.Dequant.ldc, dptr + i * _param.ldd, cptr + i * _param.Dequant.ldc);
    }
    return JblasSuccess;
  }
};
template <JBLAS_ISA ISA_T>
using DequantAddFp32 = DequantAdd<ISA_T, float>;

template <JBLAS_ISA ISA_T, typename _T>
class Add_Gelu {
 public:
  struct Param {
    _T *C, *D;
    int ldc, ldd;
  };

  JBLAS_CODE forward(const float* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
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
    auto ret = ker.forward(cptr, _param.ldc, M_offset, N_offset, M, N, param);
    return ret;
  }
};
template <JBLAS_ISA ISA_T>
using Add_GeluFp32 = Add_Gelu<ISA_T, float>;

template <JBLAS_ISA ISA_T, typename _T>
class Dequant_Add_Gelu : protected DequantAdd<ISA_T, _T> {
 public:
  using Parent = DequantAdd<ISA_T, _T>;
  using GeluKernel = jblas::epilogue::gemm::AccumulatorWriteBackWithGeluFp32<ISA_T>;
  using Param = typename Parent::Param;

  JBLAS_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    Parent::forward(cacheptr, cachestep, M_offset, N_offset, M, N, _param);
    auto COffset = M_offset * _param.Dequant.ldc + N_offset;
    auto cptr = _param.Dequant.C + COffset;
    auto ret = gelufp32.forward(cptr, _param.Dequant.ldc, M_offset, N_offset, M, N,
                                {_param.Dequant.C, _param.Dequant.ldc, NULL});
    return ret;
  }
  GeluKernel gelufp32;
};

template <JBLAS_ISA ISA_T>
using Dequant_Add_GeluFp32 = Dequant_Add_Gelu<ISA_T, float>;

template <JBLAS_ISA ISA_T, typename _T>
class ZpDequant_Add_Gelu : protected ZpDequantAdd<ISA_T, _T> {
 public:
  using Parent = ZpDequantAdd<ISA_T, _T>;
  using GeluKernel = jblas::epilogue::gemm::AccumulatorWriteBackWithGeluFp32<ISA_T>;
  using Param = typename Parent::Param;

  JBLAS_CODE forward(const int32_t* cacheptr, const int cachestep, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& _param) {
    Parent::forward(cacheptr, cachestep, M_offset, N_offset, M, N, _param);
    auto COffset = M_offset * _param.ZpDequant.ldc + N_offset;
    auto cptr = _param.ZpDequant.C + COffset;
    auto ret = gelufp32.forward(cptr, _param.ZpDequant.ldc, M_offset, N_offset, M, N,
                                {_param.ZpDequant.C, _param.ZpDequant.ldc, NULL});
    return ret;
  }
  GeluKernel gelufp32;
};

template <JBLAS_ISA ISA_T>
using ZpDequant_Add_GeluFp32 = ZpDequant_Add_Gelu<ISA_T, float>;
}  // namespace epilogue
namespace wrapper {
namespace transformer {
template <class _SiluLauncher_T, class _Launcher_T>
class FFNFusedInterface {
 public:
  static_assert(std::is_same<typename _Launcher_T::AParam, typename _SiluLauncher_T::AParam>::value,
                "Prologue A param of the 2 Launcher (w/wo SILU) should be the same.");
  struct Arguments {
    const int Seq, Fin, FMid, FOut;
    const typename _Launcher_T::AParam paramA;
    const typename _Launcher_T::AParam paramA2;
    const typename _SiluLauncher_T::BParam paramW1;
    const typename _Launcher_T::BParam paramW2, paramW3;
    const typename _SiluLauncher_T::EpiParam param1;
    const typename _Launcher_T::EpiParam param2, param3;
  };
  using Config = typename _Launcher_T::ParallelConfig;
  using ActConfig = typename _SiluLauncher_T::ParallelConfig;
  using ActivationType = typename _Launcher_T::PrologueA;
  using WeightType = typename _Launcher_T::PrologueB;
  using GemmCore = typename _Launcher_T::GemmCore;
  using LArguments = typename _Launcher_T::Param;
  using CParam = typename _Launcher_T::EpiParam;
  using Parallel = jblas::utils::parallel::Parallel2DGemmKBlockFixed<GemmCore>;
  ActivationType* getActivationPtr() { return &mLauncher.mProA; }
  // forward=packB+compute
  JBLAS_CODE compute(const Arguments& _param) {
    auto bptr = (jblas::prologue::weight_comp::gemm_kblcok::WeightBase*)(_param.paramW1.packedW);
    // dynamic quantization: Seq*Fin
    auto cb = jblas::utils::CpuBase();
    auto paraA = mLauncher.mProA.createParallel(_param.Seq, _param.Fin, bptr->mBlockSize);
    auto paraA2 = mLauncher.mProA.createParallel(_param.Seq, _param.FMid, bptr->mBlockSize);

    Parallel _paral = Parallel();   // w1&w3 from Seq* Fin=>FMid
    Parallel _paral2 = Parallel();  // w2 from Seq* FMid=>Fout
    _paral.update(_param.Seq, _param.FMid, _param.Fin, bptr->mBlockSize, cb.mNumThreads);
    _paral2.update(_param.Seq, _param.FOut, _param.FMid, bptr->mBlockSize, cb.mNumThreads);

    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      mLauncher.mProA.launch(_param.paramA, tidx, paraA);
#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          ActConfig _actconfig{
              rowidx, colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(), cb.mL2Cache};
          Config _config{rowidx,     colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(),
                         cb.mL2Cache};
          mActLauncher.launch(
              _actconfig, {_param.Seq, _param.FMid, _param.Fin, _param.paramA, _param.paramW1, _param.param1, NULL});
          mLauncher.launch(_config,
                           {_param.Seq, _param.FMid, _param.Fin, _param.paramA, _param.paramW3, _param.param3, NULL});
          int row_r = jblas::utils::remainsize(rowidx, _paral.mRows, rowsize);
          int col_r = jblas::utils::remainsize(colidx, _paral.mCols, colsize);

          // TODO(Yu): replace the naive inplace eltwise mul
          for (int i = 0; i < row_r; i++) {
            for (int j = 0; j < col_r; j++) {
              _param.param1.C[(rowidx + i) * _param.param1.ldc + colidx + j] *=
                  _param.param3.C[(rowidx + i) * _param.param3.ldc + colidx + j];
            }
          }
        }
      }
#pragma omp barrier
      mLauncher.mProA.launch(_param.paramA2, tidx, paraA2);
#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral2.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          Config _config{
              rowidx,     colidx, rowsize, colsize, _paral2.getMStep(), _paral2.getNStep(), _paral2.getKStep(),
              cb.mL2Cache};
          mLauncher.launch(_config,
                           {_param.Seq, _param.FOut, _param.FMid, _param.paramA2, _param.paramW2, _param.param2, NULL});
        }
      }
    }
    return JblasSuccess;
  }

 protected:
  _Launcher_T mLauncher;
  _SiluLauncher_T mActLauncher;
};

template <class _SiluLauncher_T, class _Launcher_T>
class FPFFNFusedInterface {
 public:
  static_assert(std::is_same<typename _Launcher_T::AParam, typename _SiluLauncher_T::AParam>::value,
                "Prologue A param of the 2 Launcher (w/wo SILU) should be the same.");
  struct Arguments {
    const int Seq, Fin, FMid, FOut;
    const typename _Launcher_T::AParam paramA;
    const typename _SiluLauncher_T::BParam paramW1;
    const typename _Launcher_T::BParam paramW2, paramW3;
    const typename _SiluLauncher_T::EpiParam param1;
    const typename _Launcher_T::EpiParam param2, param3;
  };
  using Config = typename _Launcher_T::ParallelConfig;
  using ActConfig = typename _SiluLauncher_T::ParallelConfig;
  using ActivationType = typename _Launcher_T::PrologueA;
  using WeightType = typename _Launcher_T::PrologueB;
  using GemmCore = typename _Launcher_T::GemmCore;
  using LArguments = typename _Launcher_T::Param;
  using CParam = typename _Launcher_T::EpiParam;
  using Parallel = jblas::utils::parallel::Parallel2DGemmKBlockFixed<GemmCore>;
  ActivationType* getActivationPtr() { return &mLauncher.mProA; }
  // forward=packB+compute
  JBLAS_CODE compute(const Arguments& _param) {
    auto bptr = (jblas::prologue::weight_comp::gemm_kblcok::WeightBase*)(_param.paramW1.packedW);
    if (bptr == nullptr) {
      return JblasInvalidParam;
    }
    // dynamic quantization: Seq*Fin
    auto cb = jblas::utils::CpuBase();

    Parallel _paral = Parallel();   // w1&w3 from Seq* Fin=>FMid
    Parallel _paral2 = Parallel();  // w2 from Seq* FMid=>Fout
    _paral.update(_param.Seq, _param.FMid, _param.Fin, bptr->mBlockSize, cb.mNumThreads);
    _paral2.update(_param.Seq, _param.FOut, _param.FMid, bptr->mBlockSize, cb.mNumThreads);

    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      {
        int colidx, rowidx, rowsize, colsize;
        _paral.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          ActConfig _actconfig{
              rowidx, colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(), cb.mL2Cache};
          Config _config{rowidx,     colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(),
                         cb.mL2Cache};
          mActLauncher.launch(
              _actconfig, {_param.Seq, _param.FMid, _param.Fin, _param.paramA, _param.paramW1, _param.param1, NULL});
          mLauncher.launch(_config,
                           {_param.Seq, _param.FMid, _param.Fin, _param.paramA, _param.paramW3, _param.param3, NULL});
          int row_r = jblas::utils::remainsize(rowidx, _paral.mRows, rowsize);
          int col_r = jblas::utils::remainsize(colidx, _paral.mCols, colsize);

          // TODO(Yu): replace the naive inplace eltwise mul
          for (int i = 0; i < row_r; i++) {
            for (int j = 0; j < col_r; j++) {
              _param.param1.C[(rowidx + i) * _param.param1.ldc + colidx + j] *=
                  _param.param3.C[(rowidx + i) * _param.param3.ldc + colidx + j];
            }
          }
        }
      }
#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral2.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          Config _config{
              rowidx,     colidx, rowsize, colsize, _paral2.getMStep(), _paral2.getNStep(), _paral2.getKStep(),
              cb.mL2Cache};
          mLauncher.launch(_config, {_param.Seq,
                                     _param.FOut,
                                     _param.FMid,
                                     {_param.param1.C, _param.param1.ldc},
                                     _param.paramW2,
                                     _param.param2,
                                     NULL});
        }
      }
    }
    return JblasSuccess;
  }

 protected:
  _Launcher_T mLauncher;
  _SiluLauncher_T mActLauncher;
};

template <class _SiluLauncher_T, class _Launcher_T>
class FFNFusedInterfacePerN {
 public:
  static_assert(std::is_same<typename _Launcher_T::AParam, typename _SiluLauncher_T::AParam>::value,
                "Prologue A param of the 2 Launcher (w/wo SILU) should be the same.");
  struct Arguments {
    const int Seq, Fin, FMid, FOut;
    const typename _Launcher_T::AParam paramA;
    const typename _Launcher_T::AParam paramA2;
    const typename _SiluLauncher_T::BParam paramW1;
    const typename _Launcher_T::BParam paramW2, paramW3;
    const typename _SiluLauncher_T::EpiParam param1;
    const typename _Launcher_T::EpiParam param2, param3;
  };
  using Config = typename _Launcher_T::ParallelConfig;
  using ActConfig = typename _SiluLauncher_T::ParallelConfig;
  using ActivationType = typename _Launcher_T::PrologueA;
  using WeightType = typename _Launcher_T::PrologueB;
  using GemmCore = typename _Launcher_T::GemmCore;
  using LArguments = typename _Launcher_T::Param;
  using CParam = typename _Launcher_T::EpiParam;
  using Parallel = jblas::utils::parallel::Parallel2DGemm<GemmCore>;
  ActivationType* getActivationPtr() { return &mLauncher.mProA; }
  // forward=packB+compute
  JBLAS_CODE compute(const Arguments& _param) {
    auto bptr = (jblas::prologue::weight_comp::gemm_kblcok::WeightBase*)(_param.paramW1.packedW);
    // dynamic quantization: Seq*Fin
    auto cb = jblas::utils::CpuBase();
    auto paraA = mLauncher.mProA.createParallel(_param.Seq, _param.Fin);
    auto paraA2 = mLauncher.mProA.createParallel(_param.Seq, _param.FMid);

    Parallel _paral = Parallel();   // w1&w3 from Seq* Fin=>FMid
    Parallel _paral2 = Parallel();  // w2 from Seq* FMid=>Fout
    _paral.update(_param.Seq, _param.FMid, _param.Fin, cb.mNumThreads);
    _paral2.update(_param.Seq, _param.FOut, _param.FMid, cb.mNumThreads);

    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      mLauncher.mProA.launch(_param.paramA, tidx, paraA);
#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          ActConfig _actconfig{
              rowidx, colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(), cb.mL2Cache};
          Config _config{rowidx,     colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(),
                         cb.mL2Cache};
          mActLauncher.launch(
              _actconfig, {_param.Seq, _param.FMid, _param.Fin, _param.paramA, _param.paramW1, _param.param1, NULL});
          mLauncher.launch(_config,
                           {_param.Seq, _param.FMid, _param.Fin, _param.paramA, _param.paramW3, _param.param3, NULL});
          int row_r = jblas::utils::remainsize(rowidx, _paral.mRows, rowsize);
          int col_r = jblas::utils::remainsize(colidx, _paral.mCols, colsize);

          // TODO(Yu): replace the naive inplace eltwise mul
          for (int i = 0; i < row_r; i++) {
            for (int j = 0; j < col_r; j++) {
              _param.param1.C[(rowidx + i) * _param.param1.ldc + colidx + j] *=
                  _param.param3.C[(rowidx + i) * _param.param3.ldc + colidx + j];
            }
          }
        }
      }
#pragma omp barrier
      mLauncher.mProA.launch(_param.paramA2, tidx, paraA2);
#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral2.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          Config _config{
              rowidx,     colidx, rowsize, colsize, _paral2.getMStep(), _paral2.getNStep(), _paral2.getKStep(),
              cb.mL2Cache};
          mLauncher.launch(_config,
                           {_param.Seq, _param.FOut, _param.FMid, _param.paramA2, _param.paramW2, _param.param2, NULL});
        }
      }
    }
    return JblasSuccess;
  }

 protected:
  _Launcher_T mLauncher;
  _SiluLauncher_T mActLauncher;
};

template <class _GeluLauncher_T, class _Launcher_T>
class GeluFusedInterface {
 public:
  struct Arguments {
    const int Seq, Fin, FMid, FOut;
    const typename _GeluLauncher_T::AParam paramA;
    const typename _Launcher_T::AParam paramA2;
    const typename _GeluLauncher_T::BParam paramW1;
    const typename _Launcher_T::BParam paramW2;
    const typename _GeluLauncher_T::EpiParam param1;
    const typename _Launcher_T::EpiParam param2;
  };
  using Config = typename _Launcher_T::ParallelConfig;
  using ActConfig = typename _GeluLauncher_T::ParallelConfig;
  using ActivationType = typename _Launcher_T::PrologueA;
  using GemmCore = typename _Launcher_T::GemmCore;
  using LArguments = typename _Launcher_T::Param;
  using CParam = typename _Launcher_T::EpiParam;
  using Parallel = jblas::utils::parallel::Parallel2DGemmKBlockFixed<GemmCore>;
  ActivationType* getActivationPtr() { return &mLauncher.mProA; }

  // forward=packB+compute
  JBLAS_CODE compute(const Arguments& _param) {
    auto bptr = (jblas::prologue::weight_comp::gemm_kblcok::WeightBase*)(_param.paramW1.packedW);
    // dynamic quantization: Seq*Fin
    auto paraA = mActLauncher.mProA.createParallel(_param.Seq, _param.Fin, bptr->mBlockSize);
    auto paraA2 = mLauncher.mProA.createParallel(_param.Seq, _param.FMid, bptr->mBlockSize);
    auto cb = jblas::utils::CpuBase();
    Parallel _paral = Parallel();   // w1 from Seq* Fin=>FMid
    Parallel _paral2 = Parallel();  // w2 from Seq* FMid=>Fout
    _paral.update(_param.Seq, _param.FMid, _param.Fin, bptr->mBlockSize, cb.mNumThreads);
    _paral2.update(_param.Seq, _param.FOut, _param.FMid, bptr->mBlockSize, cb.mNumThreads);

    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      mActLauncher.mProA.launch(_param.paramA, tidx, paraA);
#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          ActConfig _actconfig{
              rowidx, colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(), cb.mL2Cache};
          mActLauncher.launch(
              _actconfig, {_param.Seq, _param.FMid, _param.Fin, _param.paramA, _param.paramW1, _param.param1, NULL});
        }
      }
#pragma omp barrier
      mLauncher.mProA.launch(_param.paramA2, tidx, paraA2);
#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral2.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          Config _config{
              rowidx,     colidx, rowsize, colsize, _paral2.getMStep(), _paral2.getNStep(), _paral2.getKStep(),
              cb.mL2Cache};
          mLauncher.launch(_config,
                           {_param.Seq, _param.FOut, _param.FMid, _param.paramA2, _param.paramW2, _param.param2, NULL});
        }
      }
    }
    return JblasSuccess;
  }

 protected:
  _Launcher_T mLauncher;
  _GeluLauncher_T mActLauncher;
};

template <class _GeluLauncher_T, class _Launcher_T>
class GeluFusedInterfacePerN {
 public:
  struct Arguments {
    const int Seq, Fin, FMid, FOut;
    const typename _GeluLauncher_T::AParam paramA;
    const typename _Launcher_T::AParam paramA2;
    const typename _GeluLauncher_T::BParam paramW1;
    const typename _Launcher_T::BParam paramW2;
    const typename _GeluLauncher_T::EpiParam param1;
    const typename _Launcher_T::EpiParam param2;
  };
  using Config = typename _Launcher_T::ParallelConfig;
  using ActConfig = typename _GeluLauncher_T::ParallelConfig;
  using ActivationType = typename _Launcher_T::PrologueA;
  using GemmCore = typename _Launcher_T::GemmCore;
  using LArguments = typename _Launcher_T::Param;
  using CParam = typename _Launcher_T::EpiParam;
  using Parallel = jblas::utils::parallel::Parallel2DGemm<GemmCore>;
  ActivationType* getActivationPtr() { return &mLauncher.mProA; }

  // forward=packB+compute
  JBLAS_CODE compute(const Arguments& _param) {
    auto bptr = (jblas::prologue::weight_comp::gemm_kblcok::WeightBase*)(_param.paramW1.packedW);
    // dynamic quantization: Seq*Fin
    auto paraA = mActLauncher.mProA.createParallel(_param.Seq, _param.Fin);
    auto paraA2 = mLauncher.mProA.createParallel(_param.Seq, _param.FMid);
    auto cb = jblas::utils::CpuBase();
    Parallel _paral = Parallel();   // w1 from Seq* Fin=>FMid
    Parallel _paral2 = Parallel();  // w2 from Seq* FMid=>Fout
    _paral.update(_param.Seq, _param.FMid, _param.Fin, cb.mNumThreads);
    _paral2.update(_param.Seq, _param.FOut, _param.FMid, cb.mNumThreads);

    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      mActLauncher.mProA.launch(_param.paramA, tidx, paraA);
#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          ActConfig _actconfig{
              rowidx, colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(), cb.mL2Cache};
          mActLauncher.launch(
              _actconfig, {_param.Seq, _param.FMid, _param.Fin, _param.paramA, _param.paramW1, _param.param1, NULL});
        }
      }
#pragma omp barrier
      mLauncher.mProA.launch(_param.paramA2, tidx, paraA2);
#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral2.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          Config _config{
              rowidx,     colidx, rowsize, colsize, _paral2.getMStep(), _paral2.getNStep(), _paral2.getKStep(),
              cb.mL2Cache};
          mLauncher.launch(_config,
                           {_param.Seq, _param.FOut, _param.FMid, _param.paramA2, _param.paramW2, _param.param2, NULL});
        }
      }
    }
    return JblasSuccess;
  }

 protected:
  _Launcher_T mLauncher;
  _GeluLauncher_T mActLauncher;
};

template <class _GeluLauncher_T, class _Launcher_T>
class FpGeluFusedInterface {
 public:
  struct Arguments {
    const int Seq, Fin, FMid, FOut;
    const typename _GeluLauncher_T::AParam paramA;
    const typename _GeluLauncher_T::BParam paramW1;
    const typename _Launcher_T::BParam paramW2;
    const typename _GeluLauncher_T::EpiParam param1;
    const typename _Launcher_T::EpiParam param2;
  };
  using Config = typename _Launcher_T::ParallelConfig;
  using ActConfig = typename _GeluLauncher_T::ParallelConfig;
  using GemmCore = typename _Launcher_T::GemmCore;
  using LArguments = typename _Launcher_T::Param;
  using CParam = typename _Launcher_T::EpiParam;
  using Parallel = jblas::utils::parallel::Parallel2DGemmKBlockFixed<GemmCore>;

  JBLAS_CODE compute(const Arguments& _param) {
    auto bptr = (jblas::prologue::weight_comp::gemm_kblcok::WeightBase*)(_param.paramW1.packedW);
    auto cb = jblas::utils::CpuBase();
    Parallel _paral = Parallel();   // w1 from Seq* Fin=>FMid
    Parallel _paral2 = Parallel();  // w2 from Seq* FMid=>Fout
    _paral.update(_param.Seq, _param.FMid, _param.Fin, bptr->mBlockSize, cb.mNumThreads);
    _paral2.update(_param.Seq, _param.FOut, _param.FMid, bptr->mBlockSize, cb.mNumThreads);

    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      {
        int colidx, rowidx, rowsize, colsize;
        _paral.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          ActConfig _actconfig{
              rowidx, colidx, rowsize, colsize, _paral.getMStep(), _paral.getNStep(), _paral.getKStep(), cb.mL2Cache};
          mActLauncher.launch(
              _actconfig, {_param.Seq, _param.FMid, _param.Fin, _param.paramA, _param.paramW1, _param.param1, NULL});
        }
      }

#pragma omp barrier
      {
        int colidx, rowidx, rowsize, colsize;
        _paral2.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
        if (rowsize > 0 && colsize > 0) {
          Config _config{
              rowidx,     colidx, rowsize, colsize, _paral2.getMStep(), _paral2.getNStep(), _paral2.getKStep(),
              cb.mL2Cache};
          mLauncher.launch(_config, {_param.Seq,
                                     _param.FOut,
                                     _param.FMid,
                                     {_param.param1.C, _param.param1.ldc},
                                     _param.paramW2,
                                     _param.param2,
                                     NULL});
        }
      }
    }
    return JblasSuccess;
  }

 protected:
  _Launcher_T mLauncher;
  _GeluLauncher_T mActLauncher;
};

}  // namespace transformer
namespace kblock {
namespace avx512_vnni {
template <template <class GC, JBLAS_ISA ISA> class ProB, template <JBLAS_ISA ISA> class Epi>
using DynamicGemm =
    jblas::wrapper::gemm_kblock::GemmLauncherKBlock<JblasAVX512_VNNI,
                                                    jblas::gemm::kblock::GemmCore_Row_NN_4x48_AVX512_VNNI_KBLOCK,
                                                    jblas::prologue::gemm::ActivationF32U8KBlockQuantize, ProB, Epi>;

template <template <class GC, JBLAS_ISA ISA> class ProB, template <JBLAS_ISA ISA> class Epi>
using DynamicGemmNext = jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
    JblasAVX512_VNNI, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
    jblas::prologue::gemm::ActivationF32U8KBlockQuantize, ProB, Epi>;

template <template <class GC, JBLAS_ISA ISA> class ProB, template <JBLAS_ISA ISA> class Epi>
using DynamicGemmPerN = jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
    JblasAVX512_VNNI, jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI,
    jblas::prologue::gemm::ActivationFp32AsymU8Quantize, ProB, Epi>;

using GemmSKernelDynamicS4KBlock = DynamicGemm<WeiS4ClipFp32, jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
using SiluGemmSKernelDynamicS4KBlock = DynamicGemm<WeiS4ClipFp32, custom::epilogue::SiluFp32>;
using GeluGemmSKernelDynamicS4KBlock = DynamicGemm<WeiS4ClipFp32, custom::epilogue::GeluFp32>;
using AddGeluGemmSKernelDynamicS4KBlock = DynamicGemm<WeiS4ClipFp32, custom::epilogue::Add_GeluFp32>;
using AddGemmSKernelDynamicS4KBlock = DynamicGemm<WeiS4ClipFp32, custom::epilogue::AddFp32>;

using GemmSKernelDynamicS4KBlockNext = DynamicGemmNext<WeiS4ClipFp32, jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
using SiluGemmSKernelDynamicS4KBlockNext = DynamicGemmNext<WeiS4ClipFp32, custom::epilogue::SiluFp32>;
using GeluGemmSKernelDynamicS4KBlockNext = DynamicGemmNext<WeiS4ClipFp32, custom::epilogue::GeluFp32>;
using AddGeluGemmSKernelDynamicS4KBlockNext = DynamicGemmNext<WeiS4ClipFp32, custom::epilogue::Add_GeluFp32>;
using AddGemmSKernelDynamicS4KBlockNext = DynamicGemmNext<WeiS4ClipFp32, custom::epilogue::AddFp32>;

using GemmSKernelDynamicS8KBlock = DynamicGemm<WeiS8Fp32, jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
using SiluGemmSKernelDynamicS8KBlock = DynamicGemm<WeiS8Fp32, custom::epilogue::SiluFp32>;
using GeluGemmSKernelDynamicS8KBlock = DynamicGemm<WeiS8Fp32, custom::epilogue::GeluFp32>;
using AddGeluGemmSKernelDynamicS8KBlock = DynamicGemm<WeiS8Fp32, custom::epilogue::Add_GeluFp32>;
using AddGemmSKernelDynamicS8KBlock = DynamicGemm<WeiS8Fp32, custom::epilogue::AddFp32>;

using GemmDynamicS8PerN = DynamicGemmPerN<WeiS8Fp32PerN, jblas::epilogue::gemm::ZpDequantInt32ToFp32>;
using SiluGemmDynamicS8PerN = DynamicGemmPerN<WeiS8Fp32PerN, custom::epilogue::ZpDequantSiluFp32>;
using AddGeluGemmDynamicS8PerN = DynamicGemmPerN<WeiS8Fp32PerN, custom::epilogue::ZpDequant_Add_GeluFp32>;
using AddGemmDynamicS8PerN = DynamicGemmPerN<WeiS8Fp32PerN, custom::epilogue::ZpDequantAddFp32>;

using GemmDynamicS4ClipPerN = DynamicGemmPerN<WeiS4ClipFp32PerN, jblas::epilogue::gemm::ZpDequantInt32ToFp32>;
using SiluGemmDynamicS4ClipPerN = DynamicGemmPerN<WeiS4ClipFp32PerN, custom::epilogue::ZpDequantSiluFp32>;
using AddGeluGemmDynamicS4ClipPerN = DynamicGemmPerN<WeiS4ClipFp32PerN, custom::epilogue::ZpDequant_Add_GeluFp32>;
using AddGemmDynamicS4ClipPerN = DynamicGemmPerN<WeiS4ClipFp32PerN, custom::epilogue::ZpDequantAddFp32>;

}  // namespace avx512_vnni
namespace amx_int8 {
template <template <class GC, JBLAS_ISA ISA> class ProB, template <JBLAS_ISA ISA> class Epi>
using DynamicGemm = jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
    JblasAMX_INT8, jblas::gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK,
    jblas::prologue::gemm::ActivationF32S8KBlockQuantize, ProB, Epi>;

template <template <class GC, JBLAS_ISA ISA> class ProB, template <JBLAS_ISA ISA> class Epi>
using DynamicGemmPerN =
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<JblasAMX_INT8, jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8,
                                                             jblas::prologue::gemm::ActivationFp32SymS8Quantize, ProB,
                                                             Epi>;

using GemmSKernelDynamicS4KBlock = DynamicGemm<WeiS4ClipFp32, jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
using SiluGemmSKernelDynamicS4KBlock = DynamicGemm<WeiS4ClipFp32, custom::epilogue::SiluFp32>;
using GeluGemmSKernelDynamicS4KBlock = DynamicGemm<WeiS4ClipFp32, custom::epilogue::GeluFp32>;
using AddGeluGemmSKernelDynamicS4KBlock = DynamicGemm<WeiS4ClipFp32, custom::epilogue::Add_GeluFp32>;
using AddGemmSKernelDynamicS4KBlock = DynamicGemm<WeiS4ClipFp32, custom::epilogue::AddFp32>;

using GemmSKernelDynamicS8KBlock = DynamicGemm<WeiS8Fp32, jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
using SiluGemmSKernelDynamicS8KBlock = DynamicGemm<WeiS8Fp32, custom::epilogue::SiluFp32>;
using GeluGemmSKernelDynamicS8KBlock = DynamicGemm<WeiS8Fp32, custom::epilogue::GeluFp32>;
using AddGeluGemmSKernelDynamicS8KBlock = DynamicGemm<WeiS8Fp32, custom::epilogue::Add_GeluFp32>;
using AddGemmSKernelDynamicS8KBlock = DynamicGemm<WeiS8Fp32, custom::epilogue::AddFp32>;

using GemmDynamicS8PerN = DynamicGemmPerN<WeiS8Fp32PerN, jblas::epilogue::gemm::DequantInt32ToFp32>;
using SiluGemmDynamicS8PerN = DynamicGemmPerN<WeiS8Fp32PerN, custom::epilogue::DequantSiluFp32>;
using AddGeluGemmDynamicS8PerN = DynamicGemmPerN<WeiS8Fp32PerN, custom::epilogue::Dequant_Add_GeluFp32>;
using AddGemmDynamicS8PerN = DynamicGemmPerN<WeiS8Fp32PerN, custom::epilogue::DequantAddFp32>;

using GemmDynamicS4ClipPerN = DynamicGemmPerN<WeiS4ClipFp32PerN, jblas::epilogue::gemm::DequantInt32ToFp32>;
using SiluGemmDynamicS4ClipPerN = DynamicGemmPerN<WeiS4ClipFp32PerN, custom::epilogue::DequantSiluFp32>;
using AddGeluGemmDynamicS4ClipPerN = DynamicGemmPerN<WeiS4ClipFp32PerN, custom::epilogue::Dequant_Add_GeluFp32>;
using AddGemmDynamicS4ClipPerN = DynamicGemmPerN<WeiS4ClipFp32PerN, custom::epilogue::DequantAddFp32>;
}  // namespace amx_int8
namespace avx512f {
template <template <class GC, JBLAS_ISA ISA> class ProB, template <JBLAS_ISA ISA> class Epi>
using DefaultGemmFp32 =
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<JblasAVX512F, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F,
                                                             jblas::prologue::gemm::ActivationBase, ProB, Epi>;
using AddGeluGemmS8KBlock = DefaultGemmFp32<WeiS8Fp32, custom::epilogue::Add_GeluFp32>;
using AddGemmS8KBlock = DefaultGemmFp32<WeiS8Fp32, custom::epilogue::AddFp32>;
using GemmS8KBlock = DefaultGemmFp32<WeiS8Fp32, jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
using SiluGemmS8KBlock = DefaultGemmFp32<WeiS8Fp32, custom::epilogue::SiluFp32>;

using AddGeluGemmS4KBlock = DefaultGemmFp32<WeiS4ClipFp32, custom::epilogue::Add_GeluFp32>;
using AddGemmS4KBlock = DefaultGemmFp32<WeiS4ClipFp32, custom::epilogue::AddFp32>;
using GemmS4KBlock = DefaultGemmFp32<WeiS4ClipFp32, jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
using SiluGemmS4KBlock = DefaultGemmFp32<WeiS4ClipFp32, custom::epilogue::SiluFp32>;
}  // namespace avx512f
namespace avx2 {
template <template <class GC, JBLAS_ISA ISA> class ProB, template <JBLAS_ISA ISA> class Epi>
using DefaultGemmFp32 =
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<JblasAVX2, jblas::gemm::GemmCore_Row_NN_2x48_AVX2,
                                                             jblas::prologue::gemm::ActivationBase, ProB, Epi>;
using AddGeluGemmS8KBlock = DefaultGemmFp32<WeiS8Fp32, custom::epilogue::Add_GeluFp32>;
using AddGemmS8KBlock = DefaultGemmFp32<WeiS8Fp32, custom::epilogue::AddFp32>;
using GemmS8KBlock = DefaultGemmFp32<WeiS8Fp32, jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
using SiluGemmS8KBlock = DefaultGemmFp32<WeiS8Fp32, custom::epilogue::SiluFp32>;

using AddGeluGemmS4KBlock = DefaultGemmFp32<WeiS4ClipFp32, custom::epilogue::Add_GeluFp32>;
using AddGemmS4KBlock = DefaultGemmFp32<WeiS4ClipFp32, custom::epilogue::AddFp32>;
using GemmS4KBlock = DefaultGemmFp32<WeiS4ClipFp32, jblas::epilogue::gemm::AccumulatorWriteBackFp32>;
using SiluGemmS4KBlock = DefaultGemmFp32<WeiS4ClipFp32, custom::epilogue::SiluFp32>;
}  // namespace avx2
namespace amx_bf16 {
template <template <class GC, JBLAS_ISA ISA> class ProB, template <JBLAS_ISA ISA> class Epi>
using DefaultGemmFp32 =
    jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<JblasAMX_BF16, jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,
                                                             jblas::prologue::gemm::ActivationConverterFp32, ProB, Epi>;
using AddGeluGemmS8KBlock = DefaultGemmFp32<WeiS8Fp32, custom::epilogue::Add_GeluFp32>;
using AddGemmS8KBlock = DefaultGemmFp32<WeiS8Fp32, custom::epilogue::AddFp32>;

using AddGeluGemmS4KBlock = DefaultGemmFp32<WeiS4ClipFp32, custom::epilogue::Add_GeluFp32>;
using AddGemmS4KBlock = DefaultGemmFp32<WeiS4ClipFp32, custom::epilogue::AddFp32>;
}  // namespace amx_bf16
}  // namespace kblock
}  // namespace wrapper
}  // namespace custom
}  // namespace ne_jblas
